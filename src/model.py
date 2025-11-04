import torch
import torch.nn as nn
from transformers import AutoModel, RobertaPreTrainedModel, LukePreTrainedModel
from torch.amp import autocast
import torch.nn.functional as F
from torch.autograd import Variable
import math

from plm_vib.luke import LukeModel
from plm_vib.roberta import RobertaModel

# roberta-large
# studio-ousia/luke-large


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
        )
        pooled_output = outputs.last_hidden_state
        
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