from fastapi import FastAPI, HTTPException,Query
from pydantic import BaseModel
from typing import List
from result_inference import recommend_festivals_by_themes, recommend_places_by_festival,get_k_random_festivals  # 위 코드에서 함수 불러오기

app = FastAPI()

# 요청 데이터를 받을 Pydantic 모델 정의
class RecommendFestivalsRequest(BaseModel):
    themes: List[str]
    top_k: int = 5

class RecommendPlacesRequest(BaseModel):
    festival_title: str
    radius_km: float = 10
    top_k: int = 5

# 축제 추천 API
@app.post("/api/v1/festivals", response_model=List[dict])
async def recommend_festivals(request: RecommendFestivalsRequest):
    print(request)
    selected_themes = request.themes
    top_k = request.top_k
    
    if not selected_themes:
        raise HTTPException(status_code=400, detail="No themes provided")
    
    try:
        festivals = recommend_festivals_by_themes(selected_themes, top_k)
        return festivals
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 관광지 추천 API
@app.post("/api/v1/places", response_model=List[dict])
async def recommend_places(request: RecommendPlacesRequest):
    festival_title = request.festival_title
    radius_km = request.radius_km
    top_k = request.top_k
    
    if not festival_title:
        raise HTTPException(status_code=400, detail="No festival title provided")
    
    try:
        places = recommend_places_by_festival(festival_title, radius_km, top_k)
        return places
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/api/v1/random")
def get_random_festivals(k: int = Query(3, ge=1, le=10)):
    random_festivals = get_k_random_festivals(k)
    return random_festivals.to_dict(orient="records")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)