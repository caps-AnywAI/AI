
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import BallTree

# ===== CVAE 정의 =====
class CVAE(nn.Module):
    def __init__(self, cond_dim, item_dim, hidden_dim=128, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(cond_dim + item_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, item_dim),
            nn.Sigmoid()
        )
    def encode(self, cond, item):
        x = torch.cat([cond, item], dim=1)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def decode(self, z, cond):
        x = torch.cat([z, cond], dim=1)
        return self.decoder(x)
    def forward(self, cond, item):
        mu, logvar = self.encode(cond, item)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, cond), mu, logvar

THEMES_KR = ['자연', '도시', '힐링', '활동', '전통', '신규', '관람', '체험', '인기', '숨은', '조용한', '활기찬']
THEMES = ['nature', 'urban', 'healing', 'activity', 'traditional', 'new',
          'spectating', 'experience', 'popular', 'hidden', 'quiet', 'lively']
THEME_MAP = dict(zip(THEMES_KR, THEMES))

def selected_tags_to_vector(selected_kr_tags):
    vec = np.zeros(len(THEMES))
    for i, theme_kr in enumerate(THEMES_KR):
        if theme_kr in selected_kr_tags:
            vec[i] = 1.0
    if vec.sum() > 0:
        vec /= vec.sum()
    return vec.tolist()

@st.cache_data
def load_data():
    fdf = pd.read_csv("data/festivals_with_soft_vectors_final_adjusted_utf8.csv").rename(columns={"mapx": "longitude", "mapy": "latitude"})
    pdf = pd.read_csv("data/tourist_places_from_tourapi.csv").rename(columns={"mapx": "longitude", "mapy": "latitude"})
    return fdf, pdf

@st.cache_resource
def load_model():
    model = CVAE(cond_dim=12, item_dim=12)
    model.load_state_dict(torch.load("weights/cvae_final_12dim.pt", map_location=torch.device("cpu")))
    model.eval()
    return model

st.set_page_config(page_title="축제 기반 관광지 추천", layout="wide")
st.title("🎯 CVAE 기반 축제 추천 → 거리순 관광지 추천 시스템")

festival_df, place_df = load_data()
model = load_model()

# Step 1. 테마 선택
st.subheader("1️⃣ 관심 있는 테마를 선택하세요")
selected_themes = st.multiselect("여행 테마", THEMES_KR, max_selections=6)

# Step 2. 축제 추천
if st.button("🔍 축제 추천하기") and selected_themes:
    cond_vec = torch.tensor(selected_tags_to_vector(selected_themes), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        recon_vec, _, _ = model(cond_vec, torch.zeros((1, 12)))

    festival_df["score"] = festival_df[THEMES].apply(
        lambda row: F.cosine_similarity(torch.tensor(row.values, dtype=torch.float32), recon_vec.squeeze(0), dim=0).item(),
        axis=1
    )
    top_fests = festival_df.sort_values(by="score", ascending=False).head(5)
    st.session_state.top_fests = top_fests
    st.write("🎉 추천된 축제:")
    st.dataframe(top_fests[["title", "score"]].reset_index(drop=True))

# Step 3. 축제 선택 후 관광지 추천
if "top_fests" in st.session_state:
    selected_festival = st.selectbox("📍 관광지를 보고 싶은 축제를 선택하세요", st.session_state.top_fests["title"].tolist())
    selected_row = st.session_state.top_fests[st.session_state.top_fests["title"] == selected_festival].iloc[0]
    lat, lon = selected_row["latitude"], selected_row["longitude"]

    coords = np.radians(place_df[["latitude", "longitude"]].values)
    tree = BallTree(coords, metric="haversine")
    query = np.radians([[lat, lon]])
    idxs = tree.query_radius(query, r=10 / 6371.0)[0]
    nearby = place_df.iloc[idxs].copy()

    def haversine(lat1, lon1, lat2, lon2):
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return 6371.0 * c

    nearby["distance_km"] = nearby.apply(lambda row: haversine(lat, lon, row["latitude"], row["longitude"]), axis=1)
    top_places = nearby.sort_values("distance_km").head(5).reset_index(drop=True)

    st.subheader("📌 추천 관광지 (최대 5개, 거리순)")
    st.dataframe(top_places[["title", "distance_km"]].style.format({"distance_km": "{:.2f} km"}))
