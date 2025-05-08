import torch
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from model.cvae import CVAE
from model.utils import soft_condition_vector, ALL_KEYWORDS

def load_model(weight_path="weights/cvae_final_12dim.pt"):
    model = CVAE(cond_dim=12, item_dim=12)
    model.load_state_dict(torch.load(weight_path, map_location=torch.device("cpu")))
    model.eval()
    return model

def load_festival_data():
    fdf = pd.read_csv("data/festivals_with_soft_vectors_final_adjusted_utf8.csv").rename(columns={"mapx": "longitude", "mapy": "latitude"})
    return fdf

def load_place_data():
    pdf = pd.read_csv("data/tourist_places_from_tourapi.csv").rename(columns={"mapx": "longitude", "mapy": "latitude"})
    return pdf

def recommend_festivals_by_themes(selected_themes, top_k=5):
    model = load_model()
    festival_df = load_festival_data()
    cond_vec = soft_condition_vector(selected_themes, all_keywords=ALL_KEYWORDS)
    cond_tensor = cond_vec.unsqueeze(0)  # [1, 12]

    with torch.no_grad():
        recon_vec, _, _ = model(cond_tensor, torch.zeros((1, 12)))

    def cosine_sim(row):
        v1 = torch.tensor(row.values, dtype=torch.float32)
        return torch.nn.functional.cosine_similarity(v1, recon_vec.squeeze(0), dim=0).item()

    festival_df = festival_df.copy()
    festival_df["score"] = festival_df[ALL_KEYWORDS].apply(cosine_sim, axis=1)
    top_fests = festival_df.sort_values(by="score", ascending=False).head(top_k)

    return top_fests[["title", "score", "latitude", "longitude"]].to_dict(orient="records")

def recommend_places_by_festival(festival_title, radius_km=10, top_k=5):
    festival_df = load_festival_data()
    place_df = load_place_data()

    if festival_title not in festival_df["title"].values:
        return []

    selected = festival_df[festival_df["title"] == festival_title].iloc[0]
    lat, lon = selected["latitude"], selected["longitude"]

    coords = np.radians(place_df[["latitude", "longitude"]].values)
    tree = BallTree(coords, metric="haversine")
    query = np.radians([[lat, lon]])
    idxs = tree.query_radius(query, r=radius_km / 6371.0)[0]
    nearby = place_df.iloc[idxs].copy()

    def haversine(lat1, lon1, lat2, lon2):
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return 6371.0 * c

    nearby["distance_km"] = nearby.apply(lambda row: haversine(lat, lon, row["latitude"], row["longitude"]), axis=1)
    return nearby.sort_values("distance_km").head(top_k)[["title", "latitude", "longitude", "distance_km"]].to_dict(orient="records")
