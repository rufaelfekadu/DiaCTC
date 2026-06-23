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
        return k2.arc_sort(k2.ctc_topo(len(self.token2id) - 1)).to(self.device)

    def to(self, device):
        if hasattr(self, 'model'):
            self.model.to(device)
        if hasattr(self, 'ctc_topo'):
            self.ctc_topo.to(device)
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
        return k2.arc_sort(fsa).to(self.device), txt

    def decode_wfst(self, logits, pattern, constrained=True, search_beam=102400.0, output_beam=100000.0, min_active_states=12000, max_active_states=100000):
        '''
        Decode WFST.
        Args:
            logits: logits from the model
            pattern: pattern to decode
            wildcard_ids: wildcard ids
            constrained: whether to use constrained wildcard ids
            token2id: token to id mapping
        Returns:
            decoded text
        '''
        fsa, _ = self.build_pattern_fsa(pattern, constrained=constrained)

        log_probs = torch.log_softmax(logits, dim=-1)
        T, V = logits.shape
        dense = k2.DenseFsaVec(log_probs.unsqueeze(0), torch.tensor([[0, 0, T]], dtype=torch.int32)).to(self.device)
        decoding_graph = k2.arc_sort(k2.compose(self.ctc_topo, fsa)).to(self.device)

        lattice = k2.intersect_dense_pruned(
            decoding_graph, dense,
            search_beam=search_beam,
            output_beam=output_beam,
            min_active_states=min_active_states,
            max_active_states=max_active_states
        )

        best_path = k2.shortest_path(lattice, use_double_scores=False)
        aux = k2.get_aux_labels(best_path)[0]

        return aux

    def decode_ctc(self, logits, search_beam=20.0, output_beam=8.0, min_active_states=30, max_active_states=10000):
        '''
        Decode CTC.
        Args:
            logits: logits from the model
        Returns:
            decoded ids
        '''
        log_probs = torch.log_softmax(logits, dim=-1)
        T, V = log_probs.shape
        dense = k2.DenseFsaVec(log_probs.unsqueeze(0), torch.tensor([[0, 0, T]], dtype=torch.int32)).to(self.device)
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
    def diacritize(self, text, audio_path, constrained=True, method="wfst"):
        raise NotImplementedError("Subclasses must implement this method")
