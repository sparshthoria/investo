"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  LATEST METAL & INDEX PRICES — Day-to-Day Dataset                          ║
║  Output : latest_metal_prices.csv                                           ║
║                                                                             ║
║  COLUMNS:                                                                   ║
║    Date | Gold Price (INR / 10gms) | Silver Price (INR / 1kg)             ║
║    Price_Nifty | Price_Sensex                                               ║
║                                                                             ║
║  SOURCES (via yfinance):                                                    ║
║    Gold   → GC=F  (USD/troy oz)  × USD/INR × (10g / 31.1035g)            ║
║    Silver → SI=F  (USD/troy oz)  × USD/INR × (1000g / 31.1035g)          ║
║    Nifty  → ^NSEI  (direct INR)                                            ║
║    Sensex → ^BSESN (direct INR)                                            ║
║                                                                             ║
║  NOTE: Uses actual USD→INR conversion rates per day (USDINR=X)            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import subprocess, sys, importlib

# ── Auto-install ──────────────────────────────────────────────────────────────
REQUIRED = {"yfinance": "yfinance", "pandas": "pandas", "requests": "requests"}

print(f"\n  Python : {sys.executable}")
print(f"  Version: {sys.version.split()[0]}\n")

for mod, pkg in REQUIRED.items():
    try:
        importlib.import_module(mod)
        print(f"  ✔  {pkg}")
    except ImportError:
        print(f"  ⬇  Installing {pkg} …")
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg,
             "--quiet", "--no-warn-script-location"],
            capture_output=True, text=True
        )
        print(f"  {'✔' if r.returncode == 0 else '✘'}  {pkg}")

print("\n  All packages ready.\n")

# ── Imports ───────────────────────────────────────────────────────────────────
import os, platform, warnings
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_FILE = "latest_metal_prices.csv"
START_DATE  = "2018-01-01"
END_DATE    = datetime.now(timezone.utc).strftime("%Y-%m-%d")

# Troy ounce → grams
TROY_OZ_TO_GRAM = 31.1034768

# ── Tickers ───────────────────────────────────────────────────────────────────
# GC=F   : Gold Futures (USD per troy oz)
# SI=F   : Silver Futures (USD per troy oz)
# USDINR=X: USD to INR spot rate
# ^NSEI  : Nifty 50 (INR)
# ^BSESN : BSE Sensex (INR)

TICKERS = {
    "Gold_USD_oz"  : "GC=F",
    "Silver_USD_oz": "SI=F",
    "USDINR"       : "USDINR=X",
    "Nifty"        : "^NSEI",
    "Sensex"       : "^BSESN",
}

SEP  = "═" * 68
sep2 = "─" * 68

print(SEP)
print(f"  METAL & INDEX PRICE FETCHER")
print(f"  Date range : {START_DATE}  →  {END_DATE}")
print(f"  Output     : {OUTPUT_FILE}")
print(f"  Columns    : Date | Gold (INR/10gm) | Silver (INR/1kg)")
print(f"               Price_Nifty | Price_Sensex")
print(SEP)

# ── Step 1: Download all tickers ─────────────────────────────────────────────
print(f"\n▌ Downloading prices …")

raw = {}
for label, ticker in TICKERS.items():
    print(f"  [{ticker:12s}] {label} …", end=" ", flush=True)
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE,
                         interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            print("✘  no data")
            raw[label] = None
            continue
        # yfinance may return MultiIndex columns; flatten
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(c).strip("_") for c in df.columns]
        # Prefer "Close" column
        close_col = next((c for c in df.columns if "close" in c.lower()), None)
        if close_col is None:
            print(f"✘  no Close column (got {list(df.columns)})")
            raw[label] = None
            continue
        s = df[close_col].rename(label)
        s.index = pd.to_datetime(s.index).tz_localize(None)  # strip tz
        raw[label] = s
        print(f"✔  {len(s):,} rows  ({s.index[0].date()} → {s.index[-1].date()})")
    except Exception as e:
        print(f"✘  {e}")
        raw[label] = None

# ── Step 2: Align on a common date index ─────────────────────────────────────
print(f"\n▌ Aligning dates …")

# Collect available series
series_list = [v for v in raw.values() if v is not None]
if not series_list:
    print("  ✘  No data fetched at all — check your internet connection.")
    sys.exit(1)

# Outer join so we keep all trading days across all sources
merged = pd.concat(series_list, axis=1, join="outer")
merged.index.name = "Date"
merged.sort_index(inplace=True)

# Forward-fill gaps (weekends/holidays — last known price carries forward)
# but only within a 5-day window to avoid stale data
merged = merged.ffill(limit=5)

print(f"  Combined date range : {merged.index[0].date()}  →  {merged.index[-1].date()}")
print(f"  Total rows          : {len(merged):,}")

# ── Step 3: Handle missing USD/INR ───────────────────────────────────────────
# If USDINR fetch failed, use a static fallback rate
FALLBACK_USDINR = 84.5   # approximate as of early 2026

if raw.get("USDINR") is None or merged["USDINR"].isna().all():
    print(f"\n  ⚠  USD/INR data unavailable — using static rate ₹{FALLBACK_USDINR}")
    merged["USDINR"] = FALLBACK_USDINR
else:
    missing_fx = merged["USDINR"].isna().sum()
    if missing_fx > 0:
        merged["USDINR"] = merged["USDINR"].ffill().bfill()
        print(f"  ⚠  Filled {missing_fx} missing USD/INR values by forward/back fill")

# ── Step 4: Convert to INR ───────────────────────────────────────────────────
print(f"\n▌ Converting to INR …")

# Gold: USD/troy-oz → INR/10g
#   1 troy oz = 31.1035 g
#   Price (INR/10g) = Price_USD_oz × USDINR × (10 / 31.1035)
if raw.get("Gold_USD_oz") is not None:
    merged["Gold Price (INR / 10gms)"] = (
        merged["Gold_USD_oz"] * merged["USDINR"] * (10.0 / TROY_OZ_TO_GRAM)
    ).round(2)
    print(f"  Gold  (INR/10gm) — sample latest: "
          f"₹{merged['Gold Price (INR / 10gms)'].dropna().iloc[-1]:,.2f}")
else:
    merged["Gold Price (INR / 10gms)"] = float("nan")
    print("  Gold  ✘  no source data")

# Silver: USD/troy-oz → INR/1kg = INR/1000g
#   Price (INR/1kg) = Price_USD_oz × USDINR × (1000 / 31.1035)
if raw.get("Silver_USD_oz") is not None:
    merged["Silver Price (INR / 1kg)"] = (
        merged["Silver_USD_oz"] * merged["USDINR"] * (1000.0 / TROY_OZ_TO_GRAM)
    ).round(2)
    print(f"  Silver(INR/1kg ) — sample latest: "
          f"₹{merged['Silver Price (INR / 1kg)'].dropna().iloc[-1]:,.2f}")
else:
    merged["Silver Price (INR / 1kg)"] = float("nan")
    print("  Silver✘  no source data")

# Nifty and Sensex are already in INR points
if raw.get("Nifty") is not None:
    merged["Price_Nifty"] = merged["Nifty"].round(2)
    print(f"  Nifty            — sample latest: "
          f"{merged['Price_Nifty'].dropna().iloc[-1]:,.2f}")
else:
    merged["Price_Nifty"] = float("nan")
    print("  Nifty ✘  no source data")

if raw.get("Sensex") is not None:
    merged["Price_Sensex"] = merged["Sensex"].round(2)
    print(f"  Sensex           — sample latest: "
          f"{merged['Price_Sensex'].dropna().iloc[-1]:,.2f}")
else:
    merged["Price_Sensex"] = float("nan")
    print("  Sensex✘  no source data")

# ── Step 5: Build final DataFrame ────────────────────────────────────────────
print(f"\n▌ Building final dataset …")

out_cols = [
    "Gold Price (INR / 10gms)",
    "Silver Price (INR / 1kg)",
    "Price_Nifty",
    "Price_Sensex",
]

df_out = merged[out_cols].copy()

# Format Date as DD-MM-YYYY to match the existing nifty_sensex_gold_silver.csv
df_out.index = pd.to_datetime(df_out.index)
df_out.index = df_out.index.strftime("%d-%m-%Y")
df_out.index.name = "Date"

# Drop rows where every price column is NaN
df_out.dropna(how="all", inplace=True)

# Drop rows before 2018 (safety check)
df_out = df_out[pd.to_datetime(df_out.index, dayfirst=True) >= pd.Timestamp("2018-01-01")]

print(f"  Final rows    : {len(df_out):,}")
print(f"  Date range    : {df_out.index[0]}  →  {df_out.index[-1]}")
print(f"  Columns       : {list(df_out.columns)}")

# ── Step 6: Save ─────────────────────────────────────────────────────────────
abs_path = os.path.abspath(OUTPUT_FILE)
df_out.to_csv(abs_path, encoding="utf-8-sig")
print(f"\nSaved → {abs_path}")

# ── Step 7: Print sample ─────────────────────────────────────────────────────
print(f"\n{sep2}")
print(f"  SAMPLE — last 5 rows:")
print(sep2)
print(df_out.tail(5).to_string())
print(sep2)

print(f"\n{SEP}")
print(f"  Done!  {len(df_out):,} rows saved to {OUTPUT_FILE}")
print(f"  Columns:")
print(f"    Date                       — DD-MM-YYYY format")
print(f"    Gold Price (INR / 10gms)  — e.g. ₹89,000")
print(f"    Silver Price (INR / 1kg)  — e.g. ₹95,000")
print(f"    Price_Nifty               — e.g. 22,500")
print(f"    Price_Sensex              — e.g. 74,000")
print(SEP)

# ── Auto-open ─────────────────────────────────────────────────────────────────
try:
    sys_name = platform.system()
    import subprocess as sp
    if sys_name == "Darwin":
        sp.Popen(["open", abs_path])
    elif sys_name == "Windows":
        os.startfile(abs_path)
    else:
        sp.run(["xdg-open", abs_path],
               stdout=sp.DEVNULL, stderr=sp.DEVNULL)
except Exception:
    pass
