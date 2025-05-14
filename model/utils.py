import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# ✅ Define theme columns
theme_cols = ['nature', 'urban', 'healing', 'activity', 'traditional', 'new',
              'spectating', 'experience', 'popular', 'hidden', 'quiet', 'lively']

ALL_KEYWORDS = theme_cols.copy()

# ✅ 임베딩 모델 로드 함수
def load_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

# ✅ 키워드 임베딩 함수
@torch.no_grad()
def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embedding.numpy()

# ✅ 전체 키워드 임베딩 딕셔너리 생성 함수
def build_embedding_dict(keywords=ALL_KEYWORDS, tokenizer=None, model=None):
    if tokenizer is None or model is None:
        tokenizer, model = load_embedding_model()
    return {kw: get_embedding(kw, tokenizer, model) for kw in keywords}

# ✅ soft condition vector (임베딩 기반 RBF)
def soft_condition_vector(selected_keywords, embedding_dict, all_keywords=ALL_KEYWORDS, sigma=0.5):
    """
    선택된 키워드들과 전체 키워드 간의 임베딩 거리 기반 RBF 유사도로 소프트 벡터 생성
    """
    vec = []
    for kw in all_keywords:
        # selected_keywords 중 가장 가까운 키워드와의 거리 기준으로 weight 계산
        dists = [np.linalg.norm(embedding_dict[kw] - embedding_dict[sk]) for sk in selected_keywords]
        min_dist = min(dists)
        weight = np.exp(-min_dist**2 / (2 * sigma**2))
        vec.append(weight)
    return torch.tensor(vec, dtype=torch.float32)