from abc import ABC, abstractmethod
import k2
import torch

from diactc.config import WILDCARD_TOKEN, NO_DIAC_TOKEN, UNK_DIAC_TOKEN, UNK_TOKEN


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
        Args:
            pattern: pattern to build
            wildcard_ids: wildcard ids
            token2id: token to id mapping
        Returns:
            pattern FSA
        '''
        wildcard_ids = self.constrained_wildcard_ids if constrained else self.unconstrained_wildcard_ids
        token2id = self.token2id
        arcs = []
        state = 0
        for i, ch in enumerate(pattern):
            if ch == WILDCARD_TOKEN:
                for wid in wildcard_ids:
                    arcs.append(f"{state} {state+1} {wid} {wid} 0.0")
            else:
                if ch not in token2id:
                    arcs.append(f"{state} {state+1} {token2id[UNK_TOKEN]} {token2id[UNK_TOKEN]} 0.0")
                else:
                    tid = token2id[ch]
                    arcs.append(f"{state} {state+1} {tid} {tid} 0.0")
            state += 1
        arcs.append(f"{state} 0.0")
        txt = "\n".join(arcs)
        fsa = k2.Fsa.from_str(txt, acceptor=False, openfst=True)

        # Arc attributes that k2 carries through compose/intersect, so that lattice arcs
        # can be traced back to the pattern arc they came from (0 = no pattern arc).
        wildcard_states = torch.tensor([i for i, ch in enumerate(pattern) if ch == WILDCARD_TOKEN], dtype=torch.int32)
        fsa.pattern_arc = torch.arange(1, fsa.num_arcs + 1, dtype=torch.int32)
        fsa.is_wildcard = torch.isin(fsa.arcs.values()[:, 0], wildcard_states).to(torch.int32)

        return k2.arc_sort(fsa), txt

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
