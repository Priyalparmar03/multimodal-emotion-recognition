import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from transformers import BertTokenizer
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_DIR = "data/raw/memotion_dataset_7k/images"

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class MemeDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        try:
            image = Image.open(os.path.join(IMAGE_DIR, row['image_name'])).convert("RGB")
            image = image_transform(image)
        except Exception as e:
            print(f"Skipping bad image: {row['image_name']} ({e})")
            # fall back to a black image instead of crashing
            image = torch.zeros(3, 224, 224)

        encoded = tokenizer(
            str(row['text_clean']),
            padding='max_length', truncation=True, max_length=32,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        label = torch.tensor(row['label'], dtype=torch.long)
        return image, input_ids, attention_mask, label