import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

from src.dataset import MemeDataset
from src.models.classifier import MultimodalEmotionModel
from src.utils import set_seed, log

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df = pd.read_csv("data/processed/train.csv")
val_df = pd.read_csv("data/processed/val.csv")
NUM_CLASSES = train_df['label'].nunique()

train_loader = DataLoader(MemeDataset(train_df), batch_size=16, shuffle=True)
val_loader = DataLoader(MemeDataset(val_df), batch_size=16, shuffle=False)

model = MultimodalEmotionModel(num_classes=NUM_CLASSES, fusion_type="concat").to(device)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)
criterion = nn.CrossEntropyLoss()

EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for image, input_ids, attention_mask, label in train_loader:
        image, input_ids, attention_mask, label = (
            image.to(device), input_ids.to(device), attention_mask.to(device), label.to(device)
        )
        optimizer.zero_grad()
        logits = model(image, input_ids, attention_mask)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(1) == label).sum().item()
        total += label.size(0)

    log(f"Epoch {epoch+1}/{EPOCHS} - loss: {total_loss/len(train_loader):.4f} - train_acc: {correct/total:.4f}")

torch.save(model.state_dict(), "experiments/checkpoints/baseline_model.pt")
log("model saved")
