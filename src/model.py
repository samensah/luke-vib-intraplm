import torch
import torch.nn as nn
from transformers import AutoModel, RobertaPreTrainedModel, LukePreTrainedModel
from torch.amp import autocast
import torch.nn.functional as F
from torch.autograd import Variable
import math

import numpy as np
from scipy.linalg import eigvalsh

from repitl import matrix_itl, difference_of_entropies

from plm_vib.luke import LukeModel
from plm_vib.roberta import RobertaModel

# roberta-large
# studio-ousia/luke-large


def entropy_normalization(entropy, normalization, N, D):
    """
    Normalize the entropy based on the specified normalization method.

    Args:
        entropy (float): The entropy value to be normalized.
        normalization (str): The normalization method to use.
        N (int): The number of samples.
        D (int): The dimensionality of the data.

    Returns:
        float: The normalized entropy value.
    """
    assert normalization in ['maxEntropy', 'logN', 'logD', 'logNlogD', 'raw', 'length']

    if normalization == 'maxEntropy':
        entropy /= min(math.log(N), math.log(D))
    elif normalization == 'logN':
        entropy /= math.log(N)
    elif normalization == 'logD':
        entropy /= math.log(D)
    elif normalization == 'logNlogD':
        entropy /= (math.log(N) * math.log(D))
    elif normalization == 'raw':
        pass
    elif normalization == 'length':
        entropy = N

    return entropy





def compute_entity_entropy(hidden_states, 
                           entity_mask, 
                           alpha=1.0, 
                           normalizations=['maxEntropy']):
    """
    Compute Renyi entropy for entity tokens per sample.
    
    Args:
        hidden_states: (batch_size, seq_len, hidden_dim)
        entity_mask: (batch_size, seq_len) with 1s for entity tokens
        alpha: Renyi entropy parameter (default 1 = Shannon entropy)
    
    Returns:
        entropies: (batch_size,) - entropy value per sample
    """
    batch_size, N, D = hidden_states.shape
    entropies = []
    
    for b in range(batch_size):
        # Extract only entity tokens
        entity_idx = entity_mask[b] == 1
        entity_tokens = hidden_states[b, entity_idx]  # (num_entities, hidden_dim)
        
        if entity_tokens.shape[0] == 0:
            entropies.append(torch.nan)
            continue
        
        # Gram matrix: K = Z @ Z.T
        K = torch.matmul(entity_tokens, entity_tokens.t())
        K = torch.clamp(K, min=0)
        K = K.double() / (torch.trace(K.double()) + 1e-8)
        try:
            entropies.append(matrix_itl.matrixAlphaEntropy(K, alpha=alpha).item())
        except Exception as e:
            entropies.append(np.nan)
    return {norm: np.mean([entropy_normalization(x, norm, N, D) for x in entropies]) for norm in normalizations}
 

class REModel(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # Determine which encoder to use based on config
        self.encoder_type = getattr(config, 'encoder_type', 'luke')  # default to luke for backward compatibility
        if self.encoder_type == 'luke':
            self.luke = LukeModel(config=config)
        elif self.encoder_type == 'roberta':
            self.roberta = RobertaModel(config=config)
        else:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}. Must be 'luke' or 'roberta'")
        

        hidden_size = config.hidden_size
        self.loss_fnt = nn.CrossEntropyLoss()
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=config.dropout_prob),
            nn.Linear(hidden_size, config.num_class)
        )

    @autocast(device_type='cuda')
    def forward(self, input_ids=None, attention_mask=None, labels=None, ss=None, os=None, entity_mask=None):
        # import pdb; pdb.set_trace()
        encoder = self.luke if self.encoder_type == 'luke' else self.roberta
        outputs, vib_loss = encoder(
            input_ids,
            attention_mask=attention_mask,
            entity_mask=entity_mask,
            output_hidden_states=True,
        )
        pooled_output = outputs.last_hidden_state

        # Compute entropy only during evaluation (when labels is None)
        if labels is None:
            layer_entropies = []
            for layer_hidden_states in outputs.hidden_states:
                ent = compute_entity_entropy(layer_hidden_states, entity_mask)
                layer_entropies.append(ent)
            self.layer_entropies = layer_entropies
        
        idx = torch.arange(input_ids.size(0)).to(input_ids.device)
        ss_emb = pooled_output[idx, ss]
        os_emb = pooled_output[idx, os]
        h = torch.cat((ss_emb, os_emb), dim=-1)
        logits = self.classifier(h)
        outputs = (logits,)
        if labels is not None:
            loss = self.loss_fnt(logits.float(), labels)
            outputs = (loss, vib_loss) + outputs
        else:
            outputs = outputs + (None,)
        return outputs