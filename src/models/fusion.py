import torch
import torch.nn as nn

class ConcatFusion(nn.Module):
    """Simple baseline: just concat image + text features."""
    def __init__(self, image_dim, text_dim):
        super().__init__()
        self.output_dim = image_dim + text_dim

    def forward(self, image_feat, text_feat):
        return torch.cat([image_feat, text_feat], dim=1)


class CrossAttentionFusion(nn.Module):
    """Upgrade: text attends over image features."""
    def __init__(self, image_dim, text_dim, hidden_dim=256, num_heads=4):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.output_dim = hidden_dim

    def forward(self, image_feat, text_feat):
        img = self.image_proj(image_feat).unsqueeze(1)   # (batch, 1, hidden)
        txt = self.text_proj(text_feat).unsqueeze(1)      # (batch, 1, hidden)
        attended, _ = self.attn(query=txt, key=img, value=img)
        return attended.squeeze(1)