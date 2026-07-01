import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from src.dataset import MemeDataset
from src.models.classifier import MultimodalEmotionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_df = pd.read_csv("data/processed/test.csv")
test_loader = DataLoader(MemeDataset(test_df), batch_size=16, shuffle=False)

model = MultimodalEmotionModel(num_classes=test_df['label'].nunique()).to(device)
model.load_state_dict(torch.load("experiments/checkpoints/baseline_model.pt"))
model.eval()

all_preds, all_labels = [], []

with torch.no_grad():
    for image, input_ids, attention_mask, label in test_loader:
        image, input_ids, attention_mask = image.to(device), input_ids.to(device), attention_mask.to(device)
        logits = model(image, input_ids, attention_mask)
        preds = logits.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(label.numpy())

print(classification_report(all_labels, all_preds))
print(confusion_matrix(all_labels, all_preds))