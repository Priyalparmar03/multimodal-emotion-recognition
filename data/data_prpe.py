import pandas as pd
from sklearn.model_selection import train_test_split

LABELS_CSV = "data/raw/memotion_dataset_7k/labels.csv"


df = pd.read_csv(LABELS_CSV)
df = df[['image_name', 'text_corrected', 'overall_sentiment']].dropna()

def clean_text(t):
    t = str(t).lower().replace("\n", " ")
    return " ".join(t.split())

df['text_clean'] = df['text_corrected'].apply(clean_text)

label2id = {label: idx for idx, label in enumerate(df['overall_sentiment'].unique())}
df['label'] = df['overall_sentiment'].map(label2id)

train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

train_df.to_csv("data/processed/train.csv", index=False)
val_df.to_csv("data/processed/val.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

print("label map:", label2id)
print("train/val/test sizes:", len(train_df), len(val_df), len(test_df))