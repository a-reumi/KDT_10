import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, XLMRobertaForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ✅ 1. 설정
MODEL_NAME = "xlm-roberta-base"
NUM_LABELS = 7
BATCH_SIZE = 4
EPOCHS = 1
MAX_LEN = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 2. 데이터 불러오기
df = pd.read_csv("../../../data/open/train_data.csv")
df = df[["title", "topic_idx"]]
df = df.sample(n=1000, random_state=42)

# ✅ 3. 토크나이저
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ✅ 4. 커스텀 Dataset
class NewsDataset(Dataset):
    def __init__(self, titles, labels):
        self.titles = titles
        self.labels = labels

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        title = str(self.titles[idx])
        label = int(self.labels[idx])
        encoding = tokenizer(title,
                             padding="max_length",
                             truncation=True,
                             max_length=MAX_LEN,
                             return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label)
        }

# ✅ 5. 학습/검증 데이터 분할
train_texts, val_texts, train_labels, val_labels = train_test_split(df["title"], df["topic_idx"], test_size=0.1, random_state=42)

train_dataset = NewsDataset(train_texts.tolist(), train_labels.tolist())
val_dataset = NewsDataset(val_texts.tolist(), val_labels.tolist())

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ✅ 6. 모델 정의
model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=2e-5)

# ✅ 7. 학습 루프
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

# ✅ 8. 모델 저장
model.save_pretrained("model/")
tokenizer.save_pretrained("model/")
print("✅ 학습 완료 및 model/ 폴더에 저장되었습니다.")
