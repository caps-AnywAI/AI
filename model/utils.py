
import torch

# 영어 키워드 → 한글 키워드 매핑
EN_TO_KR = {
    "healing": "힐링",
    "nature": "자연",
    "food": "맛집",
    "history": "역사",
    "tradition": "전통",
    "art": "예술",
    "experience": "체험",
    "leisure": "레저",
    "exhibition": "전시",
    "relax": "휴식",
    "shopping": "쇼핑",
    "activity": "액티비티"
}

# 고정된 한글 키워드 순서
ALL_KEYWORDS = list(EN_TO_KR.values())

def make_condition_vector(english_keywords):
    """
    영어 키워드 리스트를 받아서 한글 매핑 후 binary multi-hot 벡터로 변환
    Args:
        english_keywords (list of str): 예: ['healing', 'food']
    Returns:
        torch.Tensor: 12차원 이진 벡터
    """
    selected_korean = [EN_TO_KR[k] for k in english_keywords if k in EN_TO_KR]
    return torch.tensor([1 if keyword in selected_korean else 0 for keyword in ALL_KEYWORDS], dtype=torch.float32)
