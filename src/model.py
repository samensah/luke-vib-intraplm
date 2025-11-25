import torch
import torch.nn as nn
from transformers import AutoModel, RobertaPreTrainedModel, LukePreTrainedModel
from torch.amp import autocast
import torch.nn.functional as F
from torch.autograd import Variable
import math

import numpy as np
from scipy.linalg import eigvalsh

# from repitl import matrix_itl, difference_of_entropies

from plm_vib.luke import LukeModel
from plm_vib.roberta import RobertaModel

# roberta-large
# studio-ousia/luke-large


# def compute_dataset_entity_entropy(hidden_states, entity_mask, alpha=1.0,):
#     """
#     Compute dataset-level matrix-based entropy using mean entity embeddings per prompt.

#     Args:
#         hidden_states: (batch_size, seq_len, hidden_dim)
#             Token embeddings for each prompt.
#         entity_mask: (batch_size, seq_len)
#             Binary mask where 1 marks entity tokens.
#         alpha: float
#             Rényi entropy order (default 1.0 = von Neumann/Shannon entropy).

#     Returns:
#         entropy: float
#             Dataset-level matrix-based entropy.
#     """
#     batch_size, seq_len, hidden_dim = hidden_states.shape
#     mean_entity_embeddings = []

#     for b in range(batch_size):
#         # Extract entity tokens for this prompt
#         entity_idx = entity_mask[b] == 1
#         entity_tokens = hidden_states[b, entity_idx]  # (num_entities, hidden_dim)

#         if entity_tokens.shape[0] == 0:
#             continue
#         mean_entity_embeddings.append(entity_tokens.mean(dim=0))

#     if len(mean_entity_embeddings) == 0:
#         return np.nan  # no entities found at all
#     Z = torch.stack(mean_entity_embeddings, dim=0)
#     K = torch.matmul(Z, Z.T)
#     K = K.double() / (torch.trace(K.double()) + 1e-8)
#     try:
#         entropy = matrix_itl.matrixAlphaEntropy(K, alpha=alpha).item()
#     except Exception as e:
#         print("Error computing dataset entropy:", e)
#         entropy = np.nan

#     # normalization: maxEntropy
#     entropy /= min(math.log(batch_size), math.log(hidden_dim))
#     return {'maxEntropy': entropy}



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

        # # Compute entropy only during evaluation (when labels is None)
        # if labels is None:
        #     layer_entropies = []
        #     for layer_hidden_states in outputs.hidden_states:
        #         ent = compute_dataset_entity_entropy(layer_hidden_states, entity_mask)
        #         layer_entropies.append(ent)
        #     self.layer_entropies = layer_entropies
        
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
    

class REModelForMetrics(REModel):
    """
    Subclass for inference only. Exposes hidden_states and adds 
    helper methods required by the metric library.
    """
    def forward(self, input_ids=None, attention_mask=None, labels=None, ss=None, os=None, entity_mask=None, return_dict=True):
        # 1. Run the encoder exactly as the original parent class does
        encoder = self.luke if self.encoder_type == 'luke' else self.roberta
        
        # Force output_hidden_states=True so we can analyze them
        outputs, vib_loss = encoder(
            input_ids,
            attention_mask=attention_mask,
            entity_mask=entity_mask,
            output_hidden_states=True,
        )
        
        pooled_output = outputs.last_hidden_state
        hidden_states = outputs.hidden_states 

        # 2. Run the classifier logic (Same as original)
        idx = torch.arange(input_ids.size(0)).to(input_ids.device)
        ss_emb = pooled_output[idx, ss]
        os_emb = pooled_output[idx, os]
        h = torch.cat((ss_emb, os_emb), dim=-1)
        logits = self.classifier(h)
        
        # 3. Return Dictionary with Hidden States (Critical for metrics)
        return {
            "logits": logits,
            "hidden_states": hidden_states,
            "loss": self.loss_fnt(logits.float(), labels) if labels is not None else None
        }

    def prepare_inputs(self, batch):
        """Standard input mover."""
        device = next(self.parameters()).device
        out = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        return out

    def _get_pooled_hidden_states(self, layer_hidden_states, mask, method='mean'):
        """
        Helper required by metric_calling.py for Dataset Entropy.
        """
        if mask is None:
             return layer_hidden_states.mean(dim=1)
             
        # Simplify: mask (B, S) -> (B, S, 1). PyTorch broadcasts this against (B, S, D) automatically.
        mask = mask.unsqueeze(-1)
        # Sum valid tokens and divide by count of valid tokens
        return (layer_hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)