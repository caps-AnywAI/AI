
# 🎉 CVAE 기반 지역 축제 추천 시스템

이 프로젝트는 **CVAE(Conditional Variational Autoencoder)** 모델을 기반으로,  
선택한 축제와 키워드를 바탕으로 반경 10km 이내 관광지를 추천해주는 시스템입니다.

---

## 📁 프로젝트 구조

```
├── model/                   # 모델 정의 및 유틸
│   ├── cvae.py
│   └── utils.py
├── train/
│   └── cvae_train.py        # 모델 학습 및 가중치 저장
├── eval/
│   └── cvae_eval.py         # Train vs Validation RMSE 시각화
├── eval.py                  # Test set RMSE, nDCG, Accuracy 평가
├── weights/
│   └── cvae_final_12dim.pt  # 저장된 모델 가중치
├── data/
│   ├── festivals_with_soft_vectors_final_adjusted_utf8.csv
│   └── tourist_places_from_tourapi.csv
```

---

일단 계속 보는 중인데 추후에 바뀔 수 있음