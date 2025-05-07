# ✅ utils.py 안에 추가할 코드 (soft_condition_vector 함수)

import numpy as np
import torch
# ✅ Define theme columns
theme_cols = ['nature', 'urban', 'healing', 'activity', 'traditional', 'new',
              'spectating', 'experience', 'popular', 'hidden', 'quiet', 'lively']

ALL_KEYWORDS = theme_cols.copy()
# ✅ soft condition vector (RBF 기반)
def soft_condition_vector(selected_keywords, all_keywords=ALL_KEYWORDS, sigma=0.5):
    """
    선택된 키워드들과 전체 키워드 간의 RBF 유사도 기반 소프트 벡터 생성
    """
    def keyword_distance(k1, k2):
        return 0.0 if k1 == k2 else 1.0

    vec = []
    for kw in all_keywords:
        min_dist = min([keyword_distance(kw, sk) for sk in selected_keywords])
        weight = np.exp(-min_dist**2 / (2 * sigma**2))
        vec.append(weight)
    return torch.tensor(vec, dtype=torch.float32)
