from abc import ABC, abstractmethod
import k2
import torch

from diactc.config import WILDCARD_TOKEN, NO_DIAC_TOKEN, UNK_DIAC_TOKEN, UNK_TOKEN
from diactc.lexicon import allowed_diacritics
from diactc.utils.text import MERGED_DIAC_TO_TOKEN_MAP


class DiacritizationModel(ABC):
    '''
    Base class for diacritization models. implements
    - load_model
    - get_logits
    - diacritize ctc
    - diacritize wfst
    '''

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def get_logits(self):
        pass

    @property
    def token2id(self):
        pass

    @property
    def constrained_wildcard_ids(self):
        pass

    @property
    def unconstrained_wildcard_ids(self):
        pass

    @property
    def word_delimiter_token(self):
        pass

    @property
    def ctc_topo(self):
        # Keep CTC topology on CPU for k2 operations
        return k2.arc_sort(k2.ctc_topo(len(self.token2id) - 1))

    def to(self, device):
        if hasattr(self, 'model'):
            self.model.to(device)
        self.device = device

    def build_pattern_fsa(self, pattern, constrained=True):
        '''
        Build pattern FSA.

        If the model has a lexicon and a positive lexicon_weight, a word the lexicon
        knows can be decoded through its lexicon variants at no cost, while decoding
        it freely costs lexicon_weight. The last letter of a word carries the case
        ending and is never restricted.
        Args:
            pattern: pattern to build
            wildcard_ids: wildcard ids
            token2id: token to id mapping
        Returns:
            pattern FSA
        '''
        wildcard_ids = self.constrained_wildcard_ids if constrained else self.unconstrained_wildcard_ids
        token2id = self.token2id
        num_tokens = len(token2id)
        lexicon = getattr(self, 'lexicon', None)
        lexicon_weight = getattr(self, 'lexicon_weight', 0.0)
        arcs = []
        lexicon_arcs = []

        def add_arc(arcs, src, dst, position, tid, cost=0.0):
            # pattern_arc identifies the (pattern position, token) pair an arc stands for, so that
            # lattice arcs can be traced back to it through compose/intersect (0 = none)
            pattern_arc = 1 + position * num_tokens + tid
            is_wildcard = int(pattern[position] == WILDCARD_TOKEN)
            arcs.append(f"{src} {dst} {tid} {tid} {pattern_arc} {is_wildcard} {cost}")

        def letter_id(ch):
            return token2id[ch] if ch in token2id else token2id[UNK_TOKEN]

        # paths through the lexicon variants of each known word, as a trie over the vowel classes of its letters
        penalized_positions = set()
        next_state = len(pattern) + 1
        if lexicon is not None and lexicon_weight > 0 and constrained:
            for start, end in self._word_spans(pattern):
                letters = [pattern[i] for i in range(start, end, 2)]
                is_diacritizable = end - start == 2 * len(letters) and all(pattern[i] == WILDCARD_TOKEN for i in range(start + 1, end, 2))
                if len(letters) < 2 or not is_diacritizable or WILDCARD_TOKEN in letters:
                    continue
                stems = {variant[:-1] for variant in lexicon.variants("".join(letters)) if len(variant) == len(letters)}
                if not stems:
                    continue

                penalized_positions.add(start)
                # states before / after a letter, by the vowel classes of the letters before it
                before_letter_state = {(): start}
                after_letter_state = {}
                for stem in sorted(stems):
                    for k, ch in enumerate(letters):
                        prefix = stem[:k]
                        position = start + 2 * k
                        if prefix not in after_letter_state:
                            after_letter_state[prefix] = next_state
                            next_state += 1
                            add_arc(lexicon_arcs, before_letter_state[prefix], after_letter_state[prefix], position, letter_id(ch))
                            if k == len(letters) - 1:
                                for wid in wildcard_ids:
                                    add_arc(lexicon_arcs, after_letter_state[prefix], end, position + 1, wid)
                        if k < len(stem) and stem[:k + 1] not in before_letter_state:
                            before_letter_state[stem[:k + 1]] = next_state
                            next_state += 1
                            for wid in self._class_token_ids(stem[k], wildcard_ids):
                                add_arc(lexicon_arcs, after_letter_state[prefix], before_letter_state[stem[:k + 1]], position + 1, wid)

        state = 0
        for i, ch in enumerate(pattern):
            if ch == WILDCARD_TOKEN:
                for wid in wildcard_ids:
                    add_arc(arcs, state, state + 1, i, wid)
            else:
                # openfst format: the last column is a cost
                add_arc(arcs, state, state + 1, i, letter_id(ch), cost=lexicon_weight if i in penalized_positions else 0.0)
            state += 1
        # openfst format: the source state of the first arc is the start state, so the lexicon arcs come last
        arcs.extend(lexicon_arcs)
        arcs.append(f"{state} 0.0")
        txt = "\n".join(arcs)
        fsa = k2.Fsa.from_str(txt, aux_label_names=['aux_labels', 'pattern_arc', 'is_wildcard'], openfst=True)
        # the final arc carries -1 in every label
        fsa.pattern_arc = fsa.pattern_arc.clamp(min=0)
        fsa.is_wildcard = fsa.is_wildcard.clamp(min=0)

        return k2.arc_sort(fsa), txt

    def _word_spans(self, pattern):
        '''
        Yield the (start, end) positions of each word of the pattern.
        '''
        start = 0
        for i, ch in enumerate(pattern):
            if ch == self.word_delimiter_token:
                if i > start:
                    yield start, i
                start = i + 1
        if len(pattern) > start:
            yield start, len(pattern)

    def _class_token_ids(self, cls, wildcard_ids):
        '''
        Wildcard ids a letter of the given lexicon vowel class may be followed by.
        '''
        token2id = self.token2id
        # blank and the tokens that stand for no / unknown diacritic are always allowed
        allowed = {0} | {token2id[t] for t in (NO_DIAC_TOKEN, UNK_DIAC_TOKEN) if t in token2id}
        for diacritics in allowed_diacritics(cls):
            for token in (diacritics, MERGED_DIAC_TO_TOKEN_MAP.get(diacritics)):
                if token in token2id:
                    allowed.add(token2id[token])

        return [wid for wid in wildcard_ids if wid in allowed]

    def _wfst_lattice(self, logits, pattern, constrained, search_beam, output_beam, min_active_states, max_active_states):
        fsa, _ = self.build_pattern_fsa(pattern, constrained=constrained)

        # Move logits to CPU for k2 operations (k2.compose with epsilons is CPU-only)
        log_probs = torch.log_softmax(logits, dim=-1).cpu()
        T, V = logits.shape
        dense = k2.DenseFsaVec(log_probs.unsqueeze(0), torch.tensor([[0, 0, T]], dtype=torch.int32))
        decoding_graph = k2.arc_sort(k2.compose(self.ctc_topo, fsa))

        lattice = k2.intersect_dense_pruned(
            decoding_graph, dense,
            search_beam=search_beam,
            output_beam=output_beam,
            min_active_states=min_active_states,
            max_active_states=max_active_states
        )

        return lattice

    def decode_wfst(self, logits, pattern, constrained=True, search_beam=102400.0, output_beam=100000.0, min_active_states=12000, max_active_states=100000, confidence_threshold=None):
        '''
        Decode WFST.
        Args:
            logits: logits from the model
            pattern: pattern to decode
            wildcard_ids: wildcard ids
            constrained: whether to use constrained wildcard ids
            token2id: token to id mapping
            confidence_threshold: if set, a wildcard token whose lattice posterior
                (probability among the options allowed at its position) is below this
                value is replaced by <no_diac>, or dropped if the vocab has no <no_diac>
        Returns:
            decoded text
        '''
        if confidence_threshold is not None:
            ids, confidences = self.decode_wfst_with_confidence(logits, pattern, constrained, search_beam, output_beam, min_active_states, max_active_states)
            return self.apply_confidence_threshold(ids, confidences, confidence_threshold)

        lattice = self._wfst_lattice(logits, pattern, constrained, search_beam, output_beam, min_active_states, max_active_states)
        best_path = k2.shortest_path(lattice, use_double_scores=False)
        aux = k2.get_aux_labels(best_path)[0]

        return aux

    def decode_wfst_with_confidence(self, logits, pattern, constrained=True, search_beam=102400.0, output_beam=100000.0, min_active_states=12000, max_active_states=100000):
        '''
        Decode WFST and score each decoded wildcard token.
        Args:
            same as decode_wfst
        Returns:
            ids: decoded ids (same as decode_wfst without a threshold)
            confidences: one entry per id. For a wildcard token, its lattice posterior,
                i.e. its probability among the options allowed at its position;
                None for the fixed tokens of the pattern
        '''
        lattice = self._wfst_lattice(logits, pattern, constrained, search_beam, output_beam, min_active_states, max_active_states)
        best_path = k2.shortest_path(lattice, use_double_scores=False)

        # posterior of each pattern arc = total posterior of the lattice arcs that came from it
        arc_post = lattice.get_arc_post(use_double_scores=True, log_semiring=True).exp()
        pattern_arc = lattice.pattern_arc.long()
        pattern_arc_post = torch.zeros(int(pattern_arc.max()) + 1, dtype=arc_post.dtype)
        pattern_arc_post.scatter_add_(0, pattern_arc, arc_post)
        pattern_arc_post = pattern_arc_post.tolist()

        ids, confidences = [], []
        for tid, arc, is_wildcard in zip(best_path.aux_labels.tolist(), best_path.pattern_arc.tolist(), best_path.is_wildcard.tolist()):
            # same as k2.get_aux_labels: skip 0's and -1's
            if tid <= 0:
                continue
            ids.append(tid)
            confidences.append(pattern_arc_post[arc] if is_wildcard else None)

        return ids, confidences

    def apply_confidence_threshold(self, ids, confidences, confidence_threshold):
        '''
        Replace low-confidence wildcard tokens by <no_diac>, or drop them if the vocab has no <no_diac>.
        Args:
            ids, confidences: output of decode_wfst_with_confidence
            confidence_threshold: tokens with a confidence below this value are replaced
        Returns:
            decoded ids
        '''
        token2id = self.token2id
        no_diac_id = token2id.get(NO_DIAC_TOKEN)
        # tokens that already mean "no / unknown diacritic" are never replaced
        keep_ids = {token2id[t] for t in (NO_DIAC_TOKEN, UNK_DIAC_TOKEN) if t in token2id}

        aux = []
        for tid, confidence in zip(ids, confidences):
            if confidence is not None and tid not in keep_ids and confidence < confidence_threshold:
                if no_diac_id is None:
                    continue
                tid = no_diac_id
            aux.append(tid)

        return aux

    def decode_ctc(self, logits, search_beam=20.0, output_beam=8.0, min_active_states=30, max_active_states=10000):
        '''
        Decode CTC.
        Args:
            logits: logits from the model
        Returns:
            decoded ids
        '''
        # Move logits to CPU for k2 operations
        log_probs = torch.log_softmax(logits, dim=-1).cpu()
        T, V = log_probs.shape
        dense = k2.DenseFsaVec(log_probs.unsqueeze(0), torch.tensor([[0, 0, T]], dtype=torch.int32))
        lattice = k2.intersect_dense_pruned(
            self.ctc_topo, dense,
            search_beam=search_beam,
            output_beam=output_beam,
            min_active_states=min_active_states,
            max_active_states=max_active_states
        )
        best_path = k2.shortest_path(lattice, use_double_scores=False)

        aux = k2.get_aux_labels(best_path)[0]
        hyp_ids = [x for x in aux if x >= 0]

        return hyp_ids

    def decode_ctc_greedy(self, logits):
        '''
        Decode CTC greedy.
        Args:
            logits: logits from the model
        Returns:
            decoded text
        '''
        log_probs = torch.log_softmax(logits, dim=-1)
        greedy_ids = log_probs.argmax(dim=-1).cpu().numpy().tolist()
        # ignore padding and blank tokens
        greedy_ids = [x for x in greedy_ids if x > 0 ]
        
        return greedy_ids

    @abstractmethod
    def diacritize(self, text, audio_path, constrained=True, method="wfst", confidence_threshold=None):
        raise NotImplementedError("Subclasses must implement this method")
