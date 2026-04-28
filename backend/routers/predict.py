from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import json
import os

from utils.finbert import get_impact_score, get_sentiment_label

router = APIRouter()

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ── Load Gold/Silver artifacts ────────────────────────────────
gs_scaler   = joblib.load(os.path.join(MODELS_DIR, "gold_silver_scaler.pkl"))
gs_gold_mdl = joblib.load(os.path.join(MODELS_DIR, "gold_silver_gold_model.pkl"))
gs_silv_mdl = joblib.load(os.path.join(MODELS_DIR, "gold_silver_silver_model.pkl"))

with open(os.path.join(MODELS_DIR, "gold_silver_feature_cols.json")) as f:
    gs_features = json.load(f)

# Vol ratios — optional, defaults to 1.0 if not found
try:
    with open(os.path.join(MODELS_DIR, "gold_silver_vol_ratios.json")) as f:
        vol_ratios = json.load(f)
    print(f"✅ Volatility ratios loaded: {vol_ratios}")
except FileNotFoundError:
    vol_ratios = {"gold": 1.0, "silver": 1.0}
    print("⚠️  vol_ratios.json not found — using 1.0x default")

# ── Load Nifty/Sensex artifacts ───────────────────────────────
ns_scaler     = joblib.load(os.path.join(MODELS_DIR, "nifty_sensex_scaler.pkl"))
ns_nifty_mdl  = joblib.load(os.path.join(MODELS_DIR, "nifty_sensex_nifty_model.pkl"))
ns_sensex_mdl = joblib.load(os.path.join(MODELS_DIR, "nifty_sensex_sensex_model.pkl"))

with open(os.path.join(MODELS_DIR, "nifty_sensex_feature_cols.json")) as f:
    ns_features = json.load(f)

print("✅ All models loaded")

# ── Request schema ────────────────────────────────────────────
class PredictRequest(BaseModel):
    news: str
    gold_price: float       # INR per 10gms
    silver_price: float     # INR per 1kg
    nifty_price: float
    sensex_price: float
    date: str               # YYYY-MM-DD


# ── Gold/Silver feature builder ───────────────────────────────
def build_gs_features(gold: float, silver: float,
                       impact: float, date_str: str) -> pd.DataFrame:
    dt  = pd.Timestamp(date_str)
    row = {}

    for col, val in [
        ("Gold Price (INR / 10gms)", gold),
        ("Silver Price (INR / 1kg)", silver)
    ]:
        for lag in [1, 2, 3, 5, 7, 10]:
            row[f"{col}_lag{lag}"] = val
        for w in [3, 5, 7, 14, 21]:
            row[f"{col}_MA{w}"]  = val
            row[f"{col}_std{w}"] = 0.0
        for r in [1, 3, 5, 10]:
            row[f"{col}_ret{r}"] = 0.0
        row[f"{col}_EMA5"]     = val
        row[f"{col}_EMA10"]    = val
        row[f"{col}_EMA21"]    = val
        row[f"{col}_range5"]   = 0.0
        row[f"{col}_RSI14"]    = 50.0
        row[f"{col}_dev_MA7"]  = 0.0
        row[f"{col}_dev_MA21"] = 0.0

    row["day_of_week"]       = dt.dayofweek
    row["month"]             = dt.month
    row["quarter"]           = dt.quarter
    row["week_of_year"]      = int(dt.isocalendar()[1])
    row["gold_silver_ratio"] = gold / silver if silver != 0 else 0.0
    row["Impact_lag1"]       = impact
    row["Impact_MA3"]        = impact

    return pd.DataFrame([row]).reindex(columns=gs_features, fill_value=0.0)


# ── Nifty/Sensex feature builder ──────────────────────────────
# Exact column order matches training notebook:
# All Price_Nifty features first, then all Price_Sensex features,
# then calendar features, then spread_NS
def build_ns_features(nifty: float, sensex: float,
                       date_str: str) -> pd.DataFrame:
    dt  = pd.Timestamp(date_str)
    row = {}

    # ── Price_Nifty features (cols 00-17) ──
    for lag in [1, 2, 3, 5]:
        row[f"Price_Nifty_lag{lag}"] = nifty
    for w in [3, 5, 7, 14]:
        row[f"Price_Nifty_MA{w}"]  = nifty
        row[f"Price_Nifty_std{w}"] = 0.0
    for r in [1, 3, 5]:
        row[f"Price_Nifty_ret{r}"] = 0.0
    row["Price_Nifty_EMA5"]   = nifty
    row["Price_Nifty_EMA10"]  = nifty
    row["Price_Nifty_range5"] = 0.0

    # ── Price_Sensex features (cols 18-35) ──
    for lag in [1, 2, 3, 5]:
        row[f"Price_Sensex_lag{lag}"] = sensex
    for w in [3, 5, 7, 14]:
        row[f"Price_Sensex_MA{w}"]  = sensex
        row[f"Price_Sensex_std{w}"] = 0.0
    for r in [1, 3, 5]:
        row[f"Price_Sensex_ret{r}"] = 0.0
    row["Price_Sensex_EMA5"]   = sensex
    row["Price_Sensex_EMA10"]  = sensex
    row["Price_Sensex_range5"] = 0.0

    # ── Calendar features (cols 36-39) ──
    row["day_of_week"]  = dt.dayofweek
    row["month"]        = dt.month
    row["quarter"]      = dt.quarter
    row["week_of_year"] = int(dt.isocalendar()[1])

    # ── Cross-index spread (col 40) ──
    row["spread_NS"] = nifty - (sensex / 3.3)

    # Enforce exact training column order via ns_features JSON
    df = pd.DataFrame([row]).reindex(columns=ns_features, fill_value=0.0)
    return df


# ── POST /predict ─────────────────────────────────────────────
@router.post("/predict")
def predict(req: PredictRequest):

    # 1. FinBERT sentiment scoring
    impact    = get_impact_score(req.news)
    sentiment = get_sentiment_label(impact)

    # 2. Gold & Silver prediction
    gs_df     = build_gs_features(
                    req.gold_price, req.silver_price,
                    impact, req.date)
    gs_scaled = gs_scaler.transform(gs_df)

    gold_ret  = gs_gold_mdl.predict(gs_scaled)[0] * vol_ratios.get("gold", 1.0)
    silv_ret  = gs_silv_mdl.predict(gs_scaled)[0] * vol_ratios.get("silver", 1.0)

    pred_gold   = req.gold_price   * (1 + gold_ret)
    pred_silver = req.silver_price * (1 + silv_ret)

    # 3. Nifty & Sensex prediction
    ns_df     = build_ns_features(
                    req.nifty_price, req.sensex_price,
                    req.date)
    ns_scaled = ns_scaler.transform(ns_df)

    raw_nifty  = float(ns_nifty_mdl.predict(ns_scaled)[0])
    raw_sensex = float(ns_sensex_mdl.predict(ns_scaled)[0])

    # ── Swap detection & correction ──────────────────────────
    # Nifty is always ~3.3x smaller than Sensex
    # If values are clearly swapped, correct them automatically
    if raw_nifty > raw_sensex:
        pred_nifty  = raw_sensex
        pred_sensex = raw_nifty
    else:
        pred_nifty  = raw_nifty
        pred_sensex = raw_sensex

    # 4. Build and return response
    return {
        "input_news":           req.news,
        "finbert_impact_score": impact,
        "sentiment":            sentiment,
        "today": {
            "gold":   round(req.gold_price,   2),
            "silver": round(req.silver_price, 2),
            "nifty":  round(req.nifty_price,  2),
            "sensex": round(req.sensex_price, 2),
        },
        "predicted_next_day": {
            "gold":   round(pred_gold,   2),
            "silver": round(pred_silver, 2),
            "nifty":  round(pred_nifty,  2),
            "sensex": round(pred_sensex, 2),
        },
        "expected_change": {
            "gold":   round(pred_gold   - req.gold_price,   2),
            "silver": round(pred_silver - req.silver_price, 2),
            "nifty":  round(pred_nifty  - req.nifty_price,  2),
            "sensex": round(pred_sensex - req.sensex_price, 2),
        },
        "direction": {
            "gold":   "UP ▲" if pred_gold   > req.gold_price   else "DOWN ▼",
            "silver": "UP ▲" if pred_silver > req.silver_price else "DOWN ▼",
            "nifty":  "UP ▲" if pred_nifty  > req.nifty_price  else "DOWN ▼",
            "sensex": "UP ▲" if pred_sensex > req.sensex_price else "DOWN ▼",
        }
    }