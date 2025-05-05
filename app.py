import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, XLMRobertaForSequenceClassification
from utils import preprocess_text, predict_topic

@st.cache_resource
def load_model_and_tokenizer():
    # model 디렉토리 내부에 저장된 HuggingFace 형식 모델 불러오기
    model = XLMRobertaForSequenceClassification.from_pretrained("model/model")
    tokenizer = AutoTokenizer.from_pretrained("model/model")
    model.eval()
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

label_map = {
    0: "IT과학",
    1: "경제",
    2: "사회",
    3: "생활문화",
    4: "세계",
    5: "스포츠",
    6: "정치"
}


st.title("뉴스 토픽 분류 예측기")
st.write("뉴스 제목을 입력하면 해당 토픽을 예측해줍니다.")

user_input = st.text_input("뉴스 제목을 입력하세요")


# 예측 후 출력
if st.button("예측하기", key="predict_button") and user_input:
    tokens = preprocess_text(user_input, tokenizer)
    pred_label = predict_topic(tokens, model)
    topic_name = label_map.get(pred_label, "알 수 없음")
    st.success(f"예측된 토픽 번호: **{pred_label}**\n\n예측된 카테고리: **{topic_name}**")

