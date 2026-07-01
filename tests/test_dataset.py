import pandas as pd
from src.dataset import MemeDataset

def test_dataset_length():
    df = pd.read_csv("data/processed/train.csv")
    dataset = MemeDataset(df)
    assert len(dataset) == len(df)

def test_dataset_item_shape():
    df = pd.read_csv("data/processed/train.csv")
    dataset = MemeDataset(df)
    image, input_ids, attention_mask, label = dataset[0]
    assert image.shape == (3, 224, 224)
    assert input_ids.shape == (32,)
    assert attention_mask.shape == (32,)
    assert label.dim() == 0