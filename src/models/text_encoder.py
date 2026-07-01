import torch.nn as nn
from transformers import BertModel

class TextEncoder(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.output_dim = 768

        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.pooler_output