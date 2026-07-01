import torch
from src.models.image_encoder import ImageEncoder
from src.models.text_encoder import TextEncoder
from src.models.fusion import ConcatFusion, CrossAttentionFusion
from src.models.classifier import MultimodalEmotionModel

def test_image_encoder_output_shape():
    encoder = ImageEncoder()
    dummy_image = torch.randn(2, 3, 224, 224)
    output = encoder(dummy_image)
    assert output.shape == (2, 512)

def test_text_encoder_output_shape():
    encoder = TextEncoder()
    input_ids = torch.randint(0, 1000, (2, 32))
    attention_mask = torch.ones(2, 32, dtype=torch.long)
    output = encoder(input_ids, attention_mask)
    assert output.shape == (2, 768)

def test_concat_fusion_shape():
    fusion = ConcatFusion(image_dim=512, text_dim=768)
    image_feat = torch.randn(2, 512)
    text_feat = torch.randn(2, 768)
    output = fusion(image_feat, text_feat)
    assert output.shape == (2, 1280)

def test_full_model_forward_pass():
    model = MultimodalEmotionModel(num_classes=3, fusion_type="concat")
    image = torch.randn(2, 3, 224, 224)
    input_ids = torch.randint(0, 1000, (2, 32))
    attention_mask = torch.ones(2, 32, dtype=torch.long)
    logits = model(image, input_ids, attention_mask)
    assert logits.shape == (2, 3)