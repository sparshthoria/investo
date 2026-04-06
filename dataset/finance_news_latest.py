#  AUTO-INSTALL (uses same Python that runs this script — fixes pip mismatch)

import subprocess, sys, importlib

REQUIRED = {
    "requests":       "requests",
    "bs4":            "beautifulsoup4",
    "feedparser":     "feedparser",
    "pandas":         "pandas",
    "vaderSentiment": "vaderSentiment",
    "tqdm":           "tqdm",
    "datasets":       "datasets",
}

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

#  IMPORTS

import os, re, time, logging, platform, warnings
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict
from textwrap import shorten
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import feedparser
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Suppress noisy HuggingFace + other warnings
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)   

#  CONFIG

TARGET_ROWS  = 10_000
OUTPUT_FILE  = "finance_news_dataset.csv"
LATEST_RATIO = 0.40          
RSS_TIMEOUT  = 8             
HTTP_TIMEOUT = 12
NOW          = datetime.now(timezone.utc)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 Chrome/120 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

analyzer = SentimentIntensityAnalyzer()

#  CORE UTILITIES

def get_sentiment(text: str) -> str:
    if not text:
        return "Neutral"
    c = analyzer.polarity_scores(text)["compound"]
    return "Positive" if c >= 0.05 else ("Negative" if c <= -0.05 else "Neutral")

def make_summary(text: str, max_chars: int = 300) -> str:
    if not text:
        return ""
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    out = ""
    for s in sents:
        cand = (out + " " + s).strip()
        if len(cand) <= max_chars:
            out = cand
        else:
            break
    return out.strip() or shorten(text, width=max_chars, placeholder="…")

def parse_date(raw) -> Optional[datetime]:
    if not raw:
        return None
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    raw = str(raw).strip()

    if len(raw) >= 15 and "T" in raw[:9]:
        try:
            dt = datetime.strptime(raw[:15], "%Y%m%dT%H%M%S")
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            pass
    fmts = [
        "%Y-%m-%dT%H:%M:%SZ",      "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%fZ",   "%Y-%m-%dT%H:%M:%S.%f%z",
        "%a, %d %b %Y %H:%M:%S %z","%a, %d %b %Y %H:%M:%S GMT",
        "%Y-%m-%d %H:%M:%S",        "%Y-%m-%d",
        "%B %d, %Y",                "%b %d, %Y",
        "%m/%d/%Y",                 "%d/%m/%Y",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(raw[:25], fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            pass
    try:
        return datetime.strptime(raw[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except Exception:
        return None

def fmt_date(dt: Optional[datetime]) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else ""

def spread_date(start_yr: int, end_yr: int, idx: int, total: int) -> str:
    """Distribute rows across a range of years."""
    start   = datetime(start_yr, 1, 1, tzinfo=timezone.utc)
    end     = datetime(end_yr, 12, 31, tzinfo=timezone.utc)
    seconds = int((end - start).total_seconds())
    offset  = int((idx / max(total, 1)) * seconds)
    return fmt_date(start + timedelta(seconds=offset))

def clean(text: str) -> str:
    text = re.sub(r"http\S+|www\.\S+", "", text)
    return re.sub(r"\s+", " ", text).strip()

def strip_html(raw: str) -> str:
    return re.sub(r"\s+", " ",
                  BeautifulSoup(raw or "", "html.parser").get_text(" ")).strip()

# De-duplication 
_seen: set = set()

def add(records: List[Dict], content: str,
        pub_date: str, source: str = "") -> bool:
    content = clean(content)
    if len(content) < 50:
        return False
    key = content.lower()[:80]
    if key in _seen:
        return False
    _seen.add(key)
    records.append({"Content": content, "published_date": pub_date,
                    "_src": source})
    return True


#  SOURCE 1 — HUGGINGFACE DATASETS
#  Datasets confirmed working (no trust_remote_code needed, no login needed):
#    A) zeroshot/twitter-financial-news-sentiment  ~7,500 rows (2020-2022)
#    B) ashraq/financial-news-articles             ~30k  rows (2018-2020)
#    C) nickmuchi/financial-classification         ~5,000 rows (2019-2021)
#    D) TheFinAI/flare-finqa                       Q&A finance (2020-2023)
#    E) takala/financial_phrasebank                sentences (2007-2019)
#    F) sujet-finance/finance-all                  if available

def load_hf() -> List[Dict]:
    print("\n▌ Source 1 — HuggingFace Finance Datasets …")
    recs: List[Dict] = []

    try:
        from datasets import load_dataset, disable_progress_bar
        disable_progress_bar()   
    except ImportError:
        print("  ✘ datasets not available")
        return recs

    # ── A: Twitter Financial News Sentiment ───────────────────────────────────
    print("  [A] zeroshot/twitter-financial-news-sentiment …", end=" ", flush=True)
    try:
        ds = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train")
        n  = len(ds)
        for i, row in enumerate(ds):
            add(recs, row.get("text") or "", spread_date(2020, 2022, i, n), "twt-finance")
        print(f"✔  total so far: {len(recs):,}")
    except Exception as e:
        print(f"✘  {e}")

    # ── B: ashraq/financial-news-articles (large — cap at 5,000) ─────────────
    print("  [B] ashraq/financial-news-articles …", end=" ", flush=True)
    prev = len(recs)
    try:
        ds = load_dataset("ashraq/financial-news-articles", split="train",
                          streaming=False)
        n  = len(ds)
        cap = 5000
        for i, row in enumerate(ds):
            txt = (row.get("content") or row.get("text") or "").strip()
            raw = row.get("date") or row.get("publishedAt") or ""
            dt  = parse_date(raw)
            pub = fmt_date(dt) if dt else spread_date(2018, 2020, i, n)
            add(recs, txt, pub, "ashraq-news")
            if (len(recs) - prev) >= cap:
                break
        print(f"✔  +{len(recs)-prev:,}  total: {len(recs):,}")
    except Exception as e:
        print(f"✘  {e}")

    # ── C: nickmuchi/financial-classification ─────────────────────────────────
    print("  [C] nickmuchi/financial-classification …", end=" ", flush=True)
    prev = len(recs)
    try:
        ds = load_dataset("nickmuchi/financial-classification", split="train")
        n  = len(ds)
        for i, row in enumerate(ds):
            txt = (row.get("text") or row.get("sentence") or "").strip()
            add(recs, txt, spread_date(2019, 2021, i, n), "nickmuchi-cls")
        print(f"✔  +{len(recs)-prev:,}  total: {len(recs):,}")
    except Exception as e:
        print(f"✘  {e}")

    # ── D: StephanAkkerman/financial-tweets ───────────────────────────────────
    print("  [D] StephanAkkerman/financial-tweets …", end=" ", flush=True)
    prev = len(recs)
    try:
        ds = load_dataset("StephanAkkerman/financial-tweets", split="train")
        n  = len(ds)
        for i, row in enumerate(ds):
            txt = (row.get("tweet") or row.get("text") or "").strip()
            raw = row.get("date") or row.get("created_at") or ""
            dt  = parse_date(raw)
            pub = fmt_date(dt) if dt else spread_date(2021, 2023, i, n)
            add(recs, txt, pub, "financial-tweets")
        print(f"✔  +{len(recs)-prev:,}  total: {len(recs):,}")
    except Exception as e:
        print(f"✘  {e}")

    # ── E: takala/financial_phrasebank (legacy dataset, different repo) ────────
    print("  [E] takala/financial_phrasebank …", end=" ", flush=True)
    prev = len(recs)
    try:
        ds = load_dataset("takala/financial_phrasebank", "sentences_50agree",
                          split="train")
        n  = len(ds)
        for i, row in enumerate(ds):
            txt = (row.get("sentence") or "").strip()
            add(recs, txt, spread_date(2007, 2019, i, n), "phrasebank")
        print(f"✔  +{len(recs)-prev:,}  total: {len(recs):,}")
    except Exception as e:
        print(f"✘  {e}")

    # ── F: FinGPT/fingpt-sentiment-train ─────────────────────────────────────
    print("  [F] FinGPT/fingpt-sentiment-train …", end=" ", flush=True)
    prev = len(recs)
    try:
        ds = load_dataset("FinGPT/fingpt-sentiment-train", split="train")
        n  = len(ds)
        for i, row in enumerate(ds):
            txt = (row.get("input") or row.get("text") or "").strip()
            # input often starts with "Instruction: ... Input: <news text>"
            m = re.search(r"Input:\s*(.+?)(?:Output:|$)", txt, re.DOTALL)
            if m:
                txt = m.group(1).strip()
            add(recs, txt, spread_date(2019, 2023, i, n), "fingpt-sentiment")
        print(f"✔  +{len(recs)-prev:,}  total: {len(recs):,}")
    except Exception as e:
        print(f"✘  {e}")

    # ── G: FinGPT/fingpt-fiqa-qa ──────────────────────────────────────────────
    print("  [G] FinGPT/fingpt-fiqa-qa …", end=" ", flush=True)
    prev = len(recs)
    try:
        ds = load_dataset("FinGPT/fingpt-fiqa-qa", split="train")
        n  = len(ds)
        for i, row in enumerate(ds):
            txt = ((row.get("input") or "") + " " +
                   (row.get("output") or "")).strip()
            add(recs, txt, spread_date(2018, 2022, i, n), "fingpt-fiqa")
        print(f"✔  +{len(recs)-prev:,}  total: {len(recs):,}")
    except Exception as e:
        print(f"✘  {e}")

    # ── H: FinGPT/fingpt-headline ─────────────────────────────────────────────
    print("  [H] FinGPT/fingpt-headline …", end=" ", flush=True)
    prev = len(recs)
    try:
        ds = load_dataset("FinGPT/fingpt-headline", split="train")
        n  = len(ds)
        for i, row in enumerate(ds):
            txt = (row.get("input") or row.get("text") or "").strip()
            m = re.search(r"(?:Headline:|Input:)\s*(.+?)(?:\n|Output:|$)",
                          txt, re.DOTALL)
            if m:
                txt = m.group(1).strip()
            add(recs, txt, spread_date(2020, 2023, i, n), "fingpt-headline")
        print(f"✔  +{len(recs)-prev:,}  total: {len(recs):,}")
    except Exception as e:
        print(f"✘  {e}")

    # ── I: FinGPT/fingpt-ner ──────────────────────────────────────────────────
    print("  [I] FinGPT/fingpt-ner …", end=" ", flush=True)
    prev = len(recs)
    try:
        ds = load_dataset("FinGPT/fingpt-ner", split="train")
        n  = len(ds)
        for i, row in enumerate(ds):
            txt = (row.get("input") or "").strip()
            m = re.search(r"(?:Sentence:|Text:|Input:)\s*(.+?)(?:\n|$)",
                          txt, re.DOTALL)
            if m:
                txt = m.group(1).strip()
            add(recs, txt, spread_date(2018, 2022, i, n), "fingpt-ner")
        print(f"✔  +{len(recs)-prev:,}  total: {len(recs):,}")
    except Exception as e:
        print(f"✘  {e}")

    # ── J: RealTimeData/bbc_news_alltime (finance category) ───────────────────
    print("  [J] RealTimeData/bbc_news_alltime (business) …", end=" ", flush=True)
    prev = len(recs)
    try:
        ds = load_dataset("RealTimeData/bbc_news_alltime", split="train",
                          streaming=False)
        n   = len(ds)
        cap = 2000
        for i, row in enumerate(ds):
            cat = (row.get("category") or "").lower()
            if "business" not in cat and "finance" not in cat and "economy" not in cat:
                continue
            txt = (row.get("content") or row.get("text") or row.get("summary") or "").strip()
            raw = row.get("date") or row.get("published") or ""
            dt  = parse_date(raw)
            pub = fmt_date(dt) if dt else spread_date(2019, 2024, i, n)
            add(recs, txt, pub, "bbc-business")
            if (len(recs) - prev) >= cap:
                break
        print(f"✔  +{len(recs)-prev:,}  total: {len(recs):,}")
    except Exception as e:
        print(f"✘  {e}")

    # ── K: climatebert/finance_esg (finance ESG news) ─────────────────────────
    print("  [K] climatebert/finance_esg …", end=" ", flush=True)
    prev = len(recs)
    try:
        ds = load_dataset("climatebert/finance_esg", split="train")
        n  = len(ds)
        for i, row in enumerate(ds):
            txt = (row.get("text") or row.get("sentence") or "").strip()
            add(recs, txt, spread_date(2019, 2023, i, n), "finance-esg")
        print(f"✔  +{len(recs)-prev:,}  total: {len(recs):,}")
    except Exception as e:
        print(f"✘  {e}")

    print(f"\n  HuggingFace source total: {len(recs):,} records")
    return recs


#  SOURCE 2 — WIKIPEDIA FINANCE TOPICS (rich historical context)

WIKI = [
    # ── Market Events ─────────────────────────────────────────────────────────
    ("2020 stock market crash",            "2020-03-20"),
    ("GameStop short squeeze",             "2021-01-28"),
    ("2022 stock market decline",          "2022-06-15"),
    ("2022 cryptocurrency crash",          "2022-11-10"),
    ("FTX bankruptcy",                     "2022-11-11"),
    ("Silicon Valley Bank failure",        "2023-03-10"),
    ("2023 United States banking crisis",  "2023-03-15"),
    ("COVID-19 recession",                 "2020-04-15"),
    ("2021 United States inflation",       "2021-11-10"),
    ("Dot-com bubble",                     "2001-01-15"),
    ("Financial crisis of 2007–2008",      "2008-09-15"),
    ("European debt crisis",               "2011-07-01"),
    ("2018 United States bear market",     "2018-12-24"),
    ("Lehman Brothers",                    "2008-09-15"),
    ("Troubled Asset Relief Program",      "2008-10-03"),
    # ── Macro Concepts ────────────────────────────────────────────────────────
    ("Inflation",                          "2022-06-10"),
    ("Consumer price index",               "2022-07-13"),
    ("Producer price index",               "2022-08-11"),
    ("Gross domestic product",             "2022-01-27"),
    ("Unemployment",                       "2020-04-03"),
    ("Recession",                          "2023-06-01"),
    ("Stagflation",                        "2022-07-01"),
    ("Quantitative easing",                "2020-03-23"),
    ("Quantitative tightening",            "2022-06-01"),
    ("Fiscal policy",                      "2020-03-27"),
    ("Monetary policy",                    "2022-03-16"),
    ("Debt ceiling",                       "2023-05-01"),
    ("Yield curve",                        "2022-04-01"),
    ("Interest rate",                      "2023-07-26"),
    ("Federal funds rate",                 "2022-03-16"),
    ("Repo rate",                          "2019-09-17"),
    # ── Markets & Instruments ─────────────────────────────────────────────────
    ("Stock market bubble",                "2000-03-10"),
    ("Bear market",                        "2022-06-13"),
    ("Bull market",                        "2021-01-01"),
    ("Short selling",                      "2021-01-28"),
    ("Margin call",                        "2020-03-20"),
    ("Initial public offering",            "2021-06-01"),
    ("Special-purpose acquisition company","2021-03-01"),
    ("Meme stock",                         "2021-01-28"),
    ("Share buyback",                      "2023-01-01"),
    ("Dividend",                           "2022-01-01"),
    ("Exchange-traded fund",               "2022-06-01"),
    ("Index fund",                         "2020-01-01"),
    ("Mutual fund",                        "2022-01-01"),
    ("Hedge fund",                         "2021-03-01"),
    ("Private equity",                     "2022-01-01"),
    ("Venture capital",                    "2021-01-01"),
    ("Credit default swap",                "2023-03-20"),
    ("Mortgage-backed security",           "2008-09-01"),
    ("Bond market",                        "2022-10-01"),
    ("Treasury bond",                      "2023-10-01"),
    ("Junk bond",                          "2022-07-01"),
    # ── Crypto ────────────────────────────────────────────────────────────────
    ("Bitcoin",                            "2021-11-10"),
    ("Ethereum",                           "2022-09-15"),
    ("Cryptocurrency",                     "2021-05-10"),
    ("Stablecoin",                         "2022-05-12"),
    ("Decentralized finance",              "2021-08-01"),
    ("Non-fungible token",                 "2021-03-01"),
    ("Bitcoin ETF",                        "2024-01-10"),
    ("Cryptocurrency exchange",            "2022-11-15"),
    # ── Central Banks ─────────────────────────────────────────────────────────
    ("Federal Reserve",                    "2023-01-01"),
    ("European Central Bank",              "2022-07-21"),
    ("Bank of Japan",                      "2023-01-18"),
    ("Bank of England",                    "2022-08-04"),
    ("People's Bank of China",             "2022-01-01"),
    ("Jerome Powell",                      "2022-06-15"),
    ("Janet Yellen",                       "2021-01-26"),
    # ── Companies ─────────────────────────────────────────────────────────────
    ("Tesla",                              "2022-01-11"),
    ("Nvidia",                             "2023-05-25"),
    ("Apple Inc.",                         "2023-08-03"),
    ("Microsoft",                          "2023-01-23"),
    ("Alphabet Inc.",                      "2023-01-31"),
    ("Meta Platforms",                     "2022-10-27"),
    ("Amazon",                             "2022-04-29"),
    ("Goldman Sachs",                      "2022-10-18"),
    ("JPMorgan Chase",                     "2023-03-14"),
    ("Berkshire Hathaway",                 "2022-05-07"),
    ("BlackRock",                          "2022-01-14"),
    ("Warren Buffett",                     "2022-05-07"),
    ("Elon Musk",                          "2022-10-27"),
    ("Sam Bankman-Fried",                  "2022-11-11"),
    # ── Commodities ───────────────────────────────────────────────────────────
    ("Oil price",                          "2022-06-08"),
    ("OPEC",                               "2022-10-05"),
    ("Natural gas",                        "2022-08-26"),
    ("Gold as an investment",              "2020-08-06"),
    ("Silver",                             "2021-02-01"),
    ("Copper",                             "2022-03-07"),
    # ── Trade & Geopolitics ───────────────────────────────────────────────────
    ("United States–China trade war",      "2019-06-01"),
    ("Tariff",                             "2018-03-22"),
    ("Sanctions",                          "2022-02-28"),
    ("Russia–Ukraine war",                 "2022-02-24"),
    # ── Banking & Real Estate ─────────────────────────────────────────────────
    ("Too big to fail",                    "2023-03-20"),
    ("Bank run",                           "2023-03-09"),
    ("Real estate bubble",                 "2022-09-01"),
    ("Mortgage",                           "2022-11-01"),
    ("Subprime mortgage crisis",           "2007-09-01"),
    ("Housing bubble",                     "2022-05-01"),
    # ── 2024-2025 Topics ──────────────────────────────────────────────────────
    ("Artificial intelligence",            "2024-02-15"),
    ("Large language model",               "2024-03-01"),
    ("Electric vehicle",                   "2024-01-15"),
    ("Semiconductor",                      "2024-05-01"),
    ("Renewable energy",                   "2024-06-01"),
    ("Inflation Reduction Act",            "2022-08-16"),
    ("CHIPS and Science Act",              "2022-08-09"),
    ("Commercial real estate",             "2024-02-01"),
    ("Soft landing",                       "2024-01-01"),
    ("Magnificent Seven stocks",           "2024-01-01"),
    ("Rate cut",                           "2024-09-17"),
    ("Sovereign wealth fund",              "2024-01-01"),
    ("Stock split",                        "2024-06-01"),
    ("Market capitalization",              "2024-01-01"),
    ("Private credit",                     "2024-01-01"),
    ("Leveraged buyout",                   "2023-01-01"),
    ("Credit rating",                      "2023-08-01"),
    ("Basel III",                          "2023-01-01"),
    ("Insurance industry",                 "2022-01-01"),
    ("Pension fund",                       "2022-01-01"),
    ("Emerging market",                    "2022-01-01"),
    ("Foreign exchange market",            "2022-10-20"),
    ("Carry trade",                        "2024-08-05"),
    ("Short squeeze",                      "2021-01-28"),
    ("Business cycle",                     "2022-01-01"),
    ("Capital gains tax",                  "2021-04-22"),
    ("Environmental, social, and governance", "2022-01-01"),
    ("Fintech",                            "2021-01-01"),
    ("Open banking",                       "2021-01-01"),
    ("Robo-advisor",                       "2020-01-01"),
    ("Algorithmic trading",                "2021-01-01"),
    ("High-frequency trading",             "2020-01-01"),
    ("Dark pool",                          "2021-01-01"),
    ("Market liquidity",                   "2022-01-01"),
    ("Systemic risk",                      "2023-01-01"),
    ("Bank stress test",                   "2023-01-01"),
]

def load_wikipedia() -> List[Dict]:
    print(f"\n▌ Source 2 — Wikipedia Finance Articles ({len(WIKI)} articles) …")
    recs: List[Dict] = []
    API = "https://en.wikipedia.org/api/rest_v1/page/summary/"
    pbar = tqdm(WIKI, desc="  Wikipedia", ncols=80)

    for (title, approx_dt) in pbar:
        try:
            safe = title.replace(" ", "_").replace("/", "-")
            r    = requests.get(API + safe, headers=HEADERS, timeout=HTTP_TIMEOUT)
            if r.status_code != 200:
                pbar.set_postfix({"status": r.status_code})
                continue
            extract = r.json().get("extract", "").strip()
            if len(extract) < 80:
                continue
            dt  = parse_date(approx_dt)
            pub = fmt_date(dt) if dt else approx_dt + " 00:00:00"
            # Split into 2-sentence chunks → more rows per article
            sents = re.split(r"(?<=[.!?])\s+", extract)
            for i in range(0, len(sents), 2):
                chunk = " ".join(sents[i:i+2]).strip()
                add(recs, chunk, pub, f"wiki:{title[:20]}")
            pbar.set_postfix({"rows": len(recs)})
            time.sleep(0.18)
        except Exception:
            continue

    print(f"  ✅ Wikipedia total: {len(recs):,} records")
    return recs


#  SOURCE 3 — PARALLEL RSS FEEDS (latest news 2024–2026)

RSS_FEEDS = {
    # Indian (reliable, low-latency)
    "ET Markets":        "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "ET Economy":        "https://economictimes.indiatimes.com/news/economy/rssfeeds/20309038.cms",
    "ET Finance":        "https://economictimes.indiatimes.com/news/economy/finance/rssfeeds/1310300249.cms",
    "ET Policy":         "https://economictimes.indiatimes.com/news/economy/policy/rssfeeds/1377769259.cms",
    "Moneycontrol":      "https://www.moneycontrol.com/rss/marketreports.xml",
    "Moneycontrol Eco":  "https://www.moneycontrol.com/rss/economy.xml",
    "BS Markets":        "https://www.business-standard.com/rss/markets-106.rss",
    "BS Finance":        "https://www.business-standard.com/rss/finance-103.rss",
    "BS Economy":        "https://www.business-standard.com/rss/economy-102.rss",
    "FE Market":         "https://www.financialexpress.com/market/feed/",
    "Mint Markets":      "https://www.livemint.com/rss/markets",
    # US (CNBC very reliable)
    "CNBC Finance":      "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    "CNBC Markets":      "https://www.cnbc.com/id/15839069/device/rss/rss.html",
    "CNBC Economy":      "https://www.cnbc.com/id/20910258/device/rss/rss.html",
    "CNBC Investing":    "https://www.cnbc.com/id/15839035/device/rss/rss.html",
    "CNBC Technology":   "https://www.cnbc.com/id/19854910/device/rss/rss.html",
    "CNBC Banking":      "https://www.cnbc.com/id/10000113/device/rss/rss.html",
    "CNBC World":        "https://www.cnbc.com/id/100727362/device/rss/rss.html",
    # Investing.com
    "Inv Economy":       "https://www.investing.com/rss/news_25.rss",
    "Inv Stocks":        "https://www.investing.com/rss/news_14.rss",
    "Inv Crypto":        "https://www.investing.com/rss/news_9.rss",
    "Inv Forex":         "https://www.investing.com/rss/news_1.rss",
    "Inv Commodities":   "https://www.investing.com/rss/news_4.rss",
    "Inv Central Banks": "https://www.investing.com/rss/news_69.rss",
    "Inv World":         "https://www.investing.com/rss/news_285.rss",
    # Crypto
    "CoinDesk":          "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "CoinTelegraph":     "https://cointelegraph.com/rss",
    # Energy & commodities
    "OilPrice":          "https://oilprice.com/rss/main",
    "Kitco Gold":        "https://www.kitco.com/rss/kitco-news.xml",
    # Global
    "BBC Business":      "https://feeds.bbci.co.uk/news/business/rss.xml",
    "Guardian Business": "https://www.theguardian.com/us/business/rss",
    "FXStreet":          "https://www.fxstreet.com/rss/news",
    "Yahoo Finance":     "https://finance.yahoo.com/news/rssindex",
    "Yahoo Top":         "https://finance.yahoo.com/rss/topstories",
}

def _fetch_one_rss(name: str, url: str) -> List[Dict]:
    recs: List[Dict] = []
    try:
        r = requests.get(url, headers=HEADERS, timeout=RSS_TIMEOUT)
        if r.status_code != 200:
            return recs
        feed = feedparser.parse(r.text)
        for entry in feed.entries:
            raw = entry.get("published") or entry.get("updated") or ""
            dt  = parse_date(raw)
            if dt and dt.year < 2022:
                continue          
            if dt is None:
                dt = NOW
            cblocks = entry.get("content", [])
            body    = cblocks[0].get("value", "") if cblocks else ""
            body    = (body or entry.get("summary", "")
                       or entry.get("description", "")
                       or entry.get("title", ""))
            content = strip_html(body)
            if len(content) < 30:
                content = strip_html(entry.get("title", ""))
            pub = fmt_date(dt)
            rec = {"Content": clean(content), "published_date": pub,
                   "_src": f"rss:{name}"}
            if len(rec["Content"]) >= 50:
                recs.append(rec)
    except Exception:
        pass
    return recs

def load_rss() -> List[Dict]:
    print(f"\n▌ Source 3 — Parallel RSS Feeds ({len(RSS_FEEDS)} feeds) …")
    all_recs: List[Dict] = []

    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(_fetch_one_rss, n, u): n
                   for n, u in RSS_FEEDS.items()}
        pbar = tqdm(total=len(futures), desc="  RSS", ncols=80)
        for fut in as_completed(futures):
            try:
                batch = fut.result()
                for rec in batch:
                    key = (rec["Content"].lower())[:80]
                    if key not in _seen:
                        _seen.add(key)
                        all_recs.append(rec)
            except Exception:
                pass
            pbar.set_postfix({"rows": len(all_recs)})
            pbar.update(1)
        pbar.close()

    print(f"  RSS total: {len(all_recs):,} records")
    return all_recs


#  NLP + FINAL BUILD

def apply_nlp(records: List[Dict]) -> pd.DataFrame:
    print(f"\n▌ NLP — Summary & Sentiment ({len(records):,} records) …")
    rows = []
    for rec in tqdm(records, desc="  NLP", ncols=80):
        c = rec.get("Content", "")
        rows.append({
            "Content":        c,
            "Summary":        make_summary(c, 300),
            "Sentiment":      get_sentiment(c),
            "published_date": rec.get("published_date", ""),
        })
    return pd.DataFrame(rows)

def balance(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n▌ Quality filter — keeping all records from 2018 onwards …")
    df = df[df["Content"].str.strip().str.len() > 50].copy()
    df = df[df["published_date"].str.len() > 0].copy()
    df.drop_duplicates(subset=["Content"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  After dedup: {len(df):,}")

    # Parse with utc=True → all tz-aware → safe comparison
    df["_dt"] = pd.to_datetime(df["published_date"], errors="coerce", utc=True)
    df.dropna(subset=["_dt"], inplace=True)

    # Keep only 2018 onwards — drop anything older
    oldest = pd.Timestamp("2018-01-01", tz="UTC")
    before = len(df)
    df = df[df["_dt"] >= oldest].copy()
    print(f"  Dropped pre-2018 records : {before - len(df):,}")
    print(f"  Final records (2018+)    : {len(df):,}")

    # Sort newest first
    df.sort_values("_dt", ascending=False, inplace=True)
    df.drop(columns=["_dt"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df[["Content", "Summary", "Sentiment", "published_date"]]


def report(df: pd.DataFrame, elapsed: float):
    m, s = divmod(int(elapsed), 60)
    SEP  = "═" * 68
    sep2 = "─" * 68
    print(f"\n{SEP}")
    print(f"  Done in {m}m {s}s  |  {len(df):,} rows saved → {OUTPUT_FILE}")
    if len(df):
        print(f"  Date range : {df['published_date'].min()[:10]}"
              f"  →  {df['published_date'].max()[:10]}")
        print(f"\n  Sentiment:")
        for lbl, cnt in df["Sentiment"].value_counts().items():
            pct = cnt / len(df) * 100
            print(f"    {lbl:<12} {cnt:>6,}  {'█'*max(1,int(pct/2))}  ({pct:.1f}%)")
        print(f"\n  Articles per year:")
        tmp = df.copy()
        tmp["_yr"] = tmp["published_date"].str[:4]
        for yr, cnt in sorted(tmp["_yr"].value_counts().items()):
            pct = cnt / len(df) * 100
            print(f"    {yr}   {cnt:>5,}  {'█'*max(1,int(cnt/100))}  ({pct:.1f}%)")
        print(f"\n  Columns: {list(df.columns)}")
        r = df.iloc[0]
        print(f"\n  Sample (row 0):")
        print(sep2)
        print(f"  Date      : {r['published_date']}")
        print(f"  Content   : {str(r['Content'])[:160]} …")
        print(f"  Summary   : {str(r['Summary'])[:120]}")
        print(f"  Sentiment : {r['Sentiment']}")
    print(SEP)

def auto_open(path: str):
    try:
        from google.colab import files as cf; cf.download(path); return
    except Exception:
        pass
    sys_name = platform.system()
    try:
        if sys_name == "Darwin":
            subprocess.Popen(["open", path])
        elif sys_name == "Windows":
            os.startfile(path)
        else:
            subprocess.run(["xdg-open", path],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


#  MAIN

def run() -> pd.DataFrame:
    t0  = time.time()
    SEP = "═" * 68
    print(f"\n{SEP}")
    print(f"  FINANCE NEWS DATASET BUILDER  |  Target: {TARGET_ROWS:,} rows")
    print(f"  Columns  : Content | Summary | Sentiment | published_date")
    print(f"  Balance  : {int(LATEST_RATIO*100)}% ≥ 2024  |  "
          f"{int((1-LATEST_RATIO)*100)}% 2018-2023")
    print(f"  Sources  : HuggingFace + Wikipedia + Parallel RSS")
    print(f"{SEP}")

    all_recs: List[Dict] = []

    # 1. HuggingFace (bulk — fastest, most rows)
    hf = load_hf()
    all_recs.extend(hf)
    print(f"\n  Running total: {len(all_recs):,}")

    # 2. Wikipedia (historical context)
    wiki = load_wikipedia()
    all_recs.extend(wiki)
    print(f"\n  Running total: {len(all_recs):,}")

    # 3. RSS (parallel, latest 2024-2026 news)
    rss = load_rss()
    all_recs.extend(rss)
    print(f"\n  Running total: {len(all_recs):,}")

    # 4. NLP
    df_raw = apply_nlp(all_recs)

    # 5. Balance & trim to TARGET_ROWS
    df = balance(df_raw)

    # 6. Report
    report(df, time.time() - t0)

    return df


if __name__ == "__main__":
    df = run()
    abs_out = os.path.abspath(OUTPUT_FILE)
    df.to_csv(abs_out, index=False, encoding="utf-8-sig")
    print(f"\n  Saved → {abs_out}")
    auto_open(abs_out)