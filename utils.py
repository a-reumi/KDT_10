import torch

def preprocess_text(text, tokenizer, max_len=32):
    # HuggingFace 토크나이저 방식 사용: 자동 패딩, 텐서 반환
    inputs = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    return inputs  # 딕셔너리 형태 {'input_ids': tensor, 'attention_mask': tensor}

def predict_topic(inputs, model):
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return pred
