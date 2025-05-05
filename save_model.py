import torch
from transformers import AutoTokenizer, XLMRobertaForSequenceClassification

# ✅ 1. 사용할 사전학습 모델 이름
model_name = "xlm-roberta-base"

# ✅ 2. 사전학습 모델과 토크나이저 불러오기 (또는 학습한 모델 불러와도 됨)
model = XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=7)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ✅ 3. 학습된 가중치가 있다면 여기에 불러오기
# model.load_state_dict(torch.load("your_trained_model.pt"))
# model.eval()

# ✅ 4. 모델과 토크나이저를 model/ 폴더에 저장 (Streamlit에서 바로 쓸 수 있음)
save_dir = "model/"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print("✅ 모델과 토크나이저가 model/ 폴더에 저장되었습니다. 이제 Streamlit에서 바로 사용할 수 있어요!")
