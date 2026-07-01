import torch.nn as nn
from src.models.image_encoder import ImageEncoder
from src.models.text_encoder import TextEncoder
from src.models.fusion import ConcatFusion, CrossAttentionFusion

class MultimodalEmotionModel(nn.Module):
    def __init__(self, num_classes, fusion_type="concat"):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()

        if fusion_type == "concat":
            self.fusion = ConcatFusion(self.image_encoder.output_dim, self.text_encoder.output_dim)
        else:
            self.fusion = CrossAttentionFusion(self.image_encoder.output_dim, self.text_encoder.output_dim)

        self.classifier = nn.Sequential(
            nn.Linear(self.fusion.output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, input_ids, attention_mask):
        image_feat = self.image_encoder(image)
        text_feat = self.text_encoder(input_ids, attention_mask)
        fused = self.fusion(image_feat, text_feat)
        return self.classifier(fused)