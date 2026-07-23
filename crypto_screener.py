"""
Daily crypto screener + recommendation-consistency tracker.

Runs once a day via GitHub Actions:
  1. Pulls Upbit KRW-market coins, classifies them by market-cap tier (CoinGecko)
     and by real 24h trading volume (Upbit) to filter out low-liquidity coins.
  2. For each tier group (대형/중형/소형), runs a Monte Carlo simulation +
     trend-extrapolation on a sample of coins, combines that with three weak
     "tilt" factors (volume confirmation, BTC-relative strength, multi-timeframe
     alignment), and picks the top 5 "recommended" coins per group.
  3. Sends a summary message via a private Telegram bot.
  4. Loads predictions.json, checks any past recommendations whose 7-day window
     has elapsed against the real price, records hit/miss, appends today's new
     recommendations, and saves the file back (committed by the workflow).

This is a statistical/backtesting tool, not investment advice. See the README
for how the "recommended" label is computed and its limitations.
"""

import html
import io
import json
import math
import os
import random
import time
from datetime import datetime, timedelta, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import glob

# Try to register a Korean font (installed via `apt-get install fonts-nanum` in the
# workflow) so chart images can show Korean text. If it's not found (e.g. running
# locally on a machine without that package), charts fall back to symbol-only /
# English text rather than crashing or showing tofu boxes.
KOREAN_FONT_NAME = None
for path in glob.glob("/usr/share/fonts/truetype/nanum/Nanum*.ttf") + glob.glob(
    "/usr/share/fonts/**/Nanum*.ttf", recursive=True
):
    try:
        fm.fontManager.addfont(path)
        KOREAN_FONT_NAME = fm.FontProperties(fname=path).get_name()
        matplotlib.rcParams["font.family"] = KOREAN_FONT_NAME
        matplotlib.rcParams["axes.unicode_minus"] = False
        break
    except Exception:  # noqa: BLE001
        continue

import requests

UPBIT_BASE = "https://api.upbit.com/v1"
COINGECKO_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"
STATE_FILE = os.path.join(os.path.dirname(__file__), "predictions.json")
WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), "factor_weights.json")
HORIZON_DAYS = 7
NUM_PATHS = 300
PER_GROUP_CAP = int(os.environ.get("PER_GROUP_CAP", "15"))

# Starting-point weights for each factor tag. These are the honest "best guess"
# values used until enough real settled results exist to replace them with
# empirically measured adjustments (see recompute_factor_weights).
DEFAULT_FACTOR_WEIGHTS = {
    "거래량↑": 5,
    "기간정합": 5,
    "BTC대비강세": 5,
    "BTC대비약세": -5,
    "RSI과매도": 5,
    "RSI과매수(주의)": -8,
    "지지선근접(진입양호)": 8,
    "고점권(진입주의)": -12,
    "이평선이격큼(과열)": -8,
    "BTC약세국면(신뢰도↓)": -10,  # note: applied as part of the trend bonus reduction, not a tag lookup
}
MIN_SAMPLE_PER_FACTOR = 15  # don't trust a factor's measured effect until it has this many settled cases
MIN_TOTAL_SETTLED_FOR_TUNING = 20  # don't touch anything until the whole system has this many settled cases
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

TIER_RANK_BOUNDS = [
    (5, "초대형"),
    (20, "대형"),
    (60, "중형"),
    (150, "소형"),
]

# Raised again: precision over quantity. Fewer "recommended" picks, but each
# one has to clear a higher probability bar AND pass the entry-timing checks
# added below (not already extended near resistance / far above its average).
RECOMMEND_THRESHOLD = 70

SECTOR_MAP = {
    "BTC": "결제/메이저", "LTC": "결제/메이저", "BCH": "결제/메이저", "BSV": "결제/메이저", "DASH": "결제/메이저",
    "DOGE": "밈코인", "SHIB": "밈코인", "PEPE": "밈코인", "BONK": "밈코인", "WIF": "밈코인", "FLOKI": "밈코인",
    "ETH": "레이어1", "SOL": "레이어1", "ADA": "레이어1", "AVAX": "레이어1", "DOT": "레이어1", "ATOM": "레이어1",
    "NEAR": "레이어1", "SUI": "레이어1", "TON": "레이어1", "KLAY": "레이어1", "APT": "레이어1", "SEI": "레이어1",
    "EOS": "레이어1", "XTZ": "레이어1", "QTUM": "레이어1", "ONT": "레이어1", "FLOW": "레이어1", "TIA": "레이어1",
    "TRX": "결제/송금", "XRP": "결제/송금", "XLM": "결제/송금", "XDC": "결제/송금", "ALGO": "결제/송금", "HBAR": "결제/송금",
    "ICP": "인프라", "FTM": "인프라", "WAVES": "인프라", "ZIL": "인프라",
    "ARB": "레이어2", "OP": "레이어2", "STRK": "레이어2", "ZK": "레이어2", "MATIC": "레이어2", "POL": "레이어2",
    "UNI": "디파이", "SUSHI": "디파이", "CAKE": "디파이", "AAVE": "디파이", "MKR": "디파이", "CRV": "디파이",
    "COMP": "디파이", "SNX": "디파이", "YFI": "디파이", "LDO": "디파이", "JUP": "디파이", "ENA": "디파이", "JTO": "디파이",
    "INJ": "디파이파생", "HYPE": "디파이파생", "DYDX": "디파이파생", "GMX": "디파이파생",
    "LINK": "오라클", "GRT": "오라클", "PYTH": "오라클", "BAND": "오라클",
    "TAO": "AI", "FET": "AI", "RENDER": "AI", "AGIX": "AI", "OCEAN": "AI", "ARKM": "AI", "WLD": "AI",
    "SAND": "메타버스/게임", "MANA": "메타버스/게임", "AXS": "메타버스/게임", "GALA": "메타버스/게임",
    "ENJ": "메타버스/게임", "IMX": "메타버스/게임", "CHZ": "메타버스/게임", "WAX": "메타버스/게임",
    "FIL": "스토리지", "STORJ": "스토리지", "AR": "스토리지", "ICX": "스토리지", "SC": "스토리지",
    "ONDO": "RWA", "POLYX": "RWA",
}


def get_sector(sym):
    return SECTOR_MAP.get(sym, "기타")


def tier_for_rank(rank):
    if rank is None:
        return "초소형"
    for bound, label in TIER_RANK_BOUNDS:
        if rank <= bound:
            return label
    return "초소형"


def fetch_upbit_markets():
    r = requests.get(f"{UPBIT_BASE}/market/all", params={"isDetails": "false"}, timeout=15)
    r.raise_for_status()
    return [m for m in r.json() if m["market"].startswith("KRW-")]


def fetch_tickers(markets):
    """markets: list of 'KRW-XXX' strings. Returns dict market -> ticker payload."""
    out = {}
    for i in range(0, len(markets), 100):
        chunk = markets[i:i + 100]
        r = requests.get(f"{UPBIT_BASE}/ticker", params={"markets": ",".join(chunk)}, timeout=15)
        r.raise_for_status()
        for d in r.json():
            out[d["market"]] = d
    return out


def fetch_candles(market, count=100):
    r = requests.get(
        f"{UPBIT_BASE}/candles/days", params={"market": market, "count": count}, timeout=15
    )
    r.raise_for_status()
    return list(reversed(r.json()))  # oldest -> newest


def fetch_market_cap_tiers():
    tiers = {}
    r = requests.get(
        COINGECKO_MARKETS_URL,
        params={"vs_currency": "krw", "order": "market_cap_desc", "per_page": 250, "page": 1},
        timeout=20,
    )
    r.raise_for_status()
    for c in r.json():
        sym = (c.get("symbol") or "").upper()
        rank = c.get("market_cap_rank")
        if not sym:
            continue
        if sym not in tiers or (rank is not None and rank < tiers[sym]["rank"]):
            tiers[sym] = {"rank": rank, "tier": tier_for_rank(rank)}
    return tiers


def classify_popularity(tickers):
    """Rank coins by real 24h Upbit trading value. Returns market -> label."""
    ranked = sorted(
        tickers.items(), key=lambda kv: kv[1].get("acc_trade_price_24h", 0), reverse=True
    )
    pop = {}
    for idx, (market, _) in enumerate(ranked):
        if idx < 30:
            pop[market] = "인기"
        elif idx < 100:
            pop[market] = "보통"
        else:
            pop[market] = "거래저조"
    return pop


def sma(arr, period):
    if len(arr) < period:
        return None
    return sum(arr[-period:]) / period


def rsi(arr, period=14):
    if len(arr) < period + 1:
        return None
    gains = losses = 0.0
    for i in range(len(arr) - period, len(arr)):
        diff = arr[i] - arr[i - 1]
        if diff >= 0:
            gains += diff
        else:
            losses += abs(diff)
    avg_gain, avg_loss = gains / period, losses / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


_FEAR_GREED_CACHE = {}


def fetch_fear_greed():
    """Crypto Fear & Greed Index (alternative.me) - a market-wide sentiment gauge,
    not a per-coin signal. Cached for the run since it's the same for every coin."""
    if "value" in _FEAR_GREED_CACHE:
        return _FEAR_GREED_CACHE
    try:
        r = requests.get("https://api.alternative.me/fng/", params={"limit": 1}, timeout=15)
        r.raise_for_status()
        d = r.json()["data"][0]
        _FEAR_GREED_CACHE["value"] = int(d["value"])
        _FEAR_GREED_CACHE["label"] = d["value_classification"]
    except Exception as e:  # noqa: BLE001
        print(f"fear/greed fetch failed: {e}")
        _FEAR_GREED_CACHE["value"] = None
        _FEAR_GREED_CACHE["label"] = None
    return _FEAR_GREED_CACHE


def fetch_kimchi_premium():
    """Compares Upbit's own BTC/KRW price against Binance's global BTC/USDT price
    (converted to KRW using Upbit's own USDT/KRW market, so both legs come from
    the same exchange's live order book and stay internally consistent). A high
    premium means the Korean market is trading BTC well above the global price -
    historically a sign of local overheating that can unwind sharply; a very low
    or negative premium can mean local panic/capitulation. This is a market-wide
    signal, not specific to any one altcoin."""
    try:
        upbit_btc = requests.get(
            f"{UPBIT_BASE}/ticker", params={"markets": "KRW-BTC"}, timeout=15
        ).json()[0]["trade_price"]
        upbit_usdt = requests.get(
            f"{UPBIT_BASE}/ticker", params={"markets": "KRW-USDT"}, timeout=15
        ).json()[0]["trade_price"]
        binance_btc = float(
            requests.get(
                "https://api.binance.com/api/v3/ticker/price",
                params={"symbol": "BTCUSDT"},
                timeout=15,
            ).json()["price"]
        )
        implied_krw = binance_btc * upbit_usdt
        premium_pct = (upbit_btc - implied_krw) / implied_krw * 100
        return {"premium_pct": premium_pct, "upbit_btc": upbit_btc, "binance_btc_krw": implied_krw}
    except Exception as e:  # noqa: BLE001
        print(f"kimchi premium fetch failed: {e}")
        return {"premium_pct": None, "upbit_btc": None, "binance_btc_krw": None}


def linear_slope(arr):
    n = len(arr)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2
    y_mean = sum(arr) / n
    num = sum((i - x_mean) * (arr[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den else 0.0


def compute_trend_dir(closes):
    """Same SMA20/SMA50 slope logic used per-coin, factored out so it can be
    reused for Binance's BTC closes (which come from a different API shape
    than Upbit candles)."""
    s20_series = [sma(closes[:i], 20) for i in range(20, len(closes) + 1)]
    s50_series = [sma(closes[:i], 50) for i in range(50, len(closes) + 1)]
    slope20 = linear_slope(s20_series[-15:]) if len(s20_series) >= 2 else 0.0
    slope50 = linear_slope(s50_series[-15:]) if len(s50_series) >= 10 else 0.0
    if slope20 > 0 and slope50 >= 0:
        return "상승 추세 연장 가능성"
    elif slope20 < 0 and slope50 <= 0:
        return "하락 추세 연장 가능성"
    return "횡보 / 방향성 약함"


def fetch_binance_btc_trend():
    """Binance is the global BTC market (far higher volume than Upbit's KRW
    market), so its trend is used as the primary 'BTC regime' signal for the
    regime filter, rather than Upbit's own thinner BTC/KRW candles."""
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "1d", "limit": 100},
            timeout=15,
        )
        r.raise_for_status()
        closes = [float(k[4]) for k in r.json()]
        return compute_trend_dir(closes)
    except Exception as e:  # noqa: BLE001
        print(f"binance BTC trend fetch failed: {e}")
        return None


def simulate(market, horizon=HORIZON_DAYS, num_paths=NUM_PATHS):
    candles = fetch_candles(market, 100)
    closes = [c["trade_price"] for c in candles]
    highs = [c.get("high_price", c["trade_price"]) for c in candles]
    lows = [c.get("low_price", c["trade_price"]) for c in candles]
    volumes = [c.get("candle_acc_trade_volume", 0) or 0 for c in candles]
    current_price = closes[-1]

    log_returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    mean_r = sum(log_returns) / len(log_returns)
    var_r = sum((r - mean_r) ** 2 for r in log_returns) / len(log_returns)
    sigma = math.sqrt(max(var_r, 0.0))

    finals = []
    for _ in range(num_paths):
        price = current_price
        for _ in range(horizon):
            z = random.gauss(0, 1)
            price *= math.exp(-0.5 * sigma * sigma + sigma * z)
        finals.append(price)
    finals.sort()
    up_pct = sum(1 for f in finals if f > current_price) / num_paths * 100
    p50 = finals[len(finals) // 2]

    s20_series = [sma(closes[:i], 20) for i in range(20, len(closes) + 1)]
    s50_series = [sma(closes[:i], 50) for i in range(50, len(closes) + 1)]
    slope20 = linear_slope(s20_series[-15:]) if len(s20_series) >= 2 else 0.0
    slope50 = linear_slope(s50_series[-15:]) if len(s50_series) >= 10 else 0.0
    trend_projection = current_price + slope20 * horizon

    if slope20 > 0 and slope50 >= 0:
        trend_dir = "상승 추세 연장 가능성"
    elif slope20 < 0 and slope50 <= 0:
        trend_dir = "하락 추세 연장 가능성"
    else:
        trend_dir = "횡보 / 방향성 약함"

    recent_vol = volumes[-5:]
    prior_vol = volumes[-20:-5]
    avg_recent = sum(recent_vol) / len(recent_vol) if recent_vol else 0
    avg_prior = sum(prior_vol) / len(prior_vol) if prior_vol else avg_recent
    volume_rising = avg_prior > 0 and avg_recent > avg_prior * 1.1

    idx30 = max(0, len(closes) - 31)
    coin_return_30 = (current_price - closes[idx30]) / closes[idx30] if closes[idx30] else 0.0
    long_term_up = current_price > closes[idx30]
    multi_aligned = (trend_dir == "상승 추세 연장 가능성" and long_term_up) or (
        trend_dir == "하락 추세 연장 가능성" and not long_term_up
    )

    rsi14 = rsi(closes, 14)
    resistance = max(highs[-20:])
    support = min(lows[-20:])

    # Pump/spike detection: a huge single-day move on abnormal volume looks
    # good on paper (high recent "return") but is a classic setup for a sharp
    # reversal, not a stable uptrend - so it's flagged separately rather than
    # folded into the normal recommendation logic.
    day_return = (closes[-1] - closes[-2]) / closes[-2] if len(closes) >= 2 and closes[-2] else 0.0
    prior20_vol = volumes[-21:-1] if len(volumes) >= 21 else volumes[:-1]
    avg_prior20_vol = sum(prior20_vol) / len(prior20_vol) if prior20_vol else 0
    vol_spike_ratio = (volumes[-1] / avg_prior20_vol) if avg_prior20_vol > 0 else 1.0
    pump_warning = day_return >= 0.15 and vol_spike_ratio >= 3.0

    # Entry timing: where does the current price sit within its recent
    # support-resistance range, and how far is it stretched above its own
    # 20-day average? Buying near resistance / far above the average is a
    # worse entry even when the probability model and trend both look fine.
    if resistance > support:
        position_ratio = (current_price - support) / (resistance - support)
    else:
        position_ratio = 0.5
    sma20_now = s20_series[-1] if s20_series else None
    pct_above_sma20 = ((current_price - sma20_now) / sma20_now) if sma20_now else 0.0

    return {
        "current_price": current_price,
        "up_pct": up_pct,
        "p50": p50,
        "trend_projection": trend_projection,
        "trend_dir": trend_dir,
        "volume_rising": volume_rising,
        "coin_return_30": coin_return_30,
        "multi_aligned": multi_aligned,
        "rsi14": rsi14,
        "resistance": resistance,
        "support": support,
        "day_return": day_return,
        "pump_warning": pump_warning,
        "position_ratio": position_ratio,
        "pct_above_sma20": pct_above_sma20,
    }


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_state(records):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def load_factor_weights():
    if os.path.exists(WEIGHTS_FILE):
        try:
            with open(WEIGHTS_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            # fill in anything missing (e.g. a new factor added later) with the default
            weights = dict(DEFAULT_FACTOR_WEIGHTS)
            weights.update(loaded)
            return weights
        except Exception as e:  # noqa: BLE001
            print(f"weights load failed, using defaults: {e}")
    return dict(DEFAULT_FACTOR_WEIGHTS)


def save_factor_weights(weights):
    with open(WEIGHTS_FILE, "w", encoding="utf-8") as f:
        json.dump(weights, f, ensure_ascii=False, indent=2)


def recompute_factor_weights(records):
    """Once enough settled results exist, measure whether each factor tag's
    presence actually correlated with higher/lower hit rates than baseline,
    and adjust its weight accordingly. Factors without enough individual
    samples keep their default weight rather than being guessed at."""
    settled = [r for r in records if r.get("settled")]
    if len(settled) < MIN_TOTAL_SETTLED_FOR_TUNING:
        return dict(DEFAULT_FACTOR_WEIGHTS), None

    baseline_rate = sum(1 for r in settled if r.get("hit")) / len(settled)
    weights = {}
    report_lines = []
    for tag, default_w in DEFAULT_FACTOR_WEIGHTS.items():
        with_tag = [r for r in settled if tag in (r.get("factorTags") or [])]
        if len(with_tag) < MIN_SAMPLE_PER_FACTOR:
            weights[tag] = default_w
            continue
        hits = sum(1 for r in with_tag if r.get("hit"))
        rate = hits / len(with_tag)
        diff_pp = (rate - baseline_rate) * 100  # percentage points vs baseline
        adjustment = max(-15.0, min(20.0, diff_pp * 0.5))
        weights[tag] = round(adjustment, 1)
        report_lines.append(
            f"　{tag}: {len(with_tag)}건, 적중률 {rate*100:.1f}% (기준선 {baseline_rate*100:.1f}%) → 가중치 {default_w}→{weights[tag]}"
        )

    report = "\n".join(report_lines) if report_lines else None
    return weights, report


def send_telegram(title, body):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set - skipping message.\n--- message ---\n" + body)
        return
    text = f"{title}\n\n{body}"
    # Telegram messages are capped at 4096 characters; trim defensively.
    if len(text) > 3900:
        text = text[:3900] + "\n...(생략, 스캔 범위를 줄이면 전체가 다 옵니다)"
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=15,
        )
    except Exception as e:  # noqa: BLE001
        print(f"telegram send failed: {e}")


def send_telegram_photo(image_bytes, caption=""):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set - skipping photo.")
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
            data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption},
            files={"photo": ("chart.png", image_bytes, "image/png")},
            timeout=30,
        )
    except Exception as e:  # noqa: BLE001
        print(f"telegram photo send failed: {e}")


def generate_summary_card(top5_by_group, fg_value, fg_label, btc_trend_dir, kp_pct, today_str):
    """One polished, single-canvas summary image with all three tier groups
    side by side. Uses one shared axes (not per-column subplots) so spacing
    and proportions are controlled precisely, and uses scatter markers for the
    entry-position dot (a plt.Circle patch gets stretched into an ellipse
    whenever the axes' x/y data scales differ - scatter markers are defined in
    points, not data units, so they stay circular regardless)."""
    from matplotlib.patches import FancyBboxPatch

    TIER_META = {
        "대형": {"color": "#2563EB", "light": "#EFF4FF"},
        "중형": {"color": "#059669", "light": "#ECFDF5"},
        "소형": {"color": "#D97706", "light": "#FFFBEB"},
    }
    tiers = ["대형", "중형", "소형"]
    max_rows = max((len(top5_by_group[t]) for t in tiers), default=1) or 1

    col_w, gap = 10.0, 0.6
    fig_w = col_w * 3 + gap * 2
    row_h = 2.0
    header_h = 1.6
    top_banner_h = 1.3
    legend_h = 0.9
    fig_h = top_banner_h + header_h + max_rows * row_h + legend_h + 0.6

    fig = plt.figure(figsize=(fig_w / 2.2, fig_h / 2.2), dpi=160)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    y_top = fig_h - 0.15

    # ---- top banner: date + market-wide indicators as pill badges ----
    ax.text(0.2, y_top - 0.45, today_str, fontsize=13, fontweight="bold", color="#111827", va="center")

    def pill(x, y, text, bg, fg):
        w = 0.17 * len(text) + 0.35
        box = FancyBboxPatch((x, y - 0.28), w, 0.56, boxstyle="round,pad=0,rounding_size=0.28",
                              linewidth=0, facecolor=bg, zorder=2)
        ax.add_patch(box)
        ax.text(x + w / 2, y, text, ha="center", va="center", fontsize=9.5, color=fg,
                fontweight="bold", zorder=3)
        return x + w + 0.25

    px = 0.2
    py = y_top - top_banner_h + 0.35
    if fg_value is not None:
        fg_bg, fg_fg = ("#FEF2F2", "#DC2626") if fg_value <= 25 else ("#F0FDF4", "#16A34A") if fg_value >= 75 else ("#F3F4F6", "#4B5563")
        px = pill(px, py, f"공포탐욕 {fg_value} ({fg_label})", fg_bg, fg_fg)
    btc_bg, btc_fg = ("#F0FDF4", "#16A34A") if btc_trend_dir == "상승 추세 연장 가능성" else \
                     ("#FEF2F2", "#DC2626") if btc_trend_dir == "하락 추세 연장 가능성" else ("#F3F4F6", "#4B5563")
    px = pill(px, py, f"BTC {btc_trend_dir.split()[0]}", btc_bg, btc_fg)
    if kp_pct is not None:
        kp_bg, kp_fg = ("#FFF7ED", "#D97706") if kp_pct >= 5 else ("#EFF6FF", "#2563EB") if kp_pct <= -1 else ("#F3F4F6", "#4B5563")
        pill(px, py, f"김프 {kp_pct:+.1f}%", kp_bg, kp_fg)

    ax.plot([0.2, fig_w - 0.2], [y_top - top_banner_h, y_top - top_banner_h], color="#E5E7EB", lw=1, zorder=1)

    # ---- three columns ----
    body_top = y_top - top_banner_h
    for col, tier_name in enumerate(tiers):
        meta = TIER_META[tier_name]
        x0 = 0.2 + col * (col_w + gap)
        picks = top5_by_group[tier_name]

        # rounded header
        header = FancyBboxPatch((x0, body_top - header_h), col_w, header_h,
                                 boxstyle="round,pad=0,rounding_size=0.35",
                                 linewidth=0, facecolor=meta["color"], zorder=2)
        ax.add_patch(header)
        ax.text(x0 + col_w / 2, body_top - header_h / 2 + 0.12, tier_name, ha="center", va="center",
                fontsize=16, color="white", fontweight="bold", zorder=3)
        ax.text(x0 + col_w / 2, body_top - header_h / 2 - 0.35, f"TOP {len(picks)}", ha="center", va="center",
                fontsize=9.5, color="white", alpha=0.85, zorder=3)

        rows_top = body_top - header_h - 0.15

        if not picks:
            ax.text(x0 + col_w / 2, rows_top - max_rows * row_h / 2, "분석 결과 없음",
                    ha="center", va="center", fontsize=10, color="#9CA3AF")
            continue

        for i, r in enumerate(picks):
            row_y1 = rows_top - i * row_h
            row_y0 = row_y1 - (row_h - 0.15)
            row_cy = (row_y0 + row_y1) / 2
            bg = meta["light"] if r["recommended"] else ("#FAFAFA" if i % 2 else "white")
            row_box = FancyBboxPatch((x0, row_y0), col_w, row_y1 - row_y0,
                                      boxstyle="round,pad=0,rounding_size=0.12",
                                      linewidth=0.8, edgecolor="#E5E7EB", facecolor=bg, zorder=1)
            ax.add_patch(row_box)

            name_x = x0 + 0.3
            ax.text(name_x, row_y1 - 0.42, f"{i+1}.", fontsize=10, color="#9CA3AF",
                    va="center", ha="left", zorder=3)
            ax.text(name_x + 0.42, row_y1 - 0.42,
                    f"{r['koName']} ({sym_of_market(r['market'])})",
                    fontsize=11.5, fontweight="bold", color="#111827", va="center", ha="left", zorder=3)

            if r["recommended"]:
                badge_w = 1.0
                badge_x = x0 + col_w - badge_w - 0.25
                ax.add_patch(FancyBboxPatch((badge_x, row_y1 - 0.6), badge_w, 0.42,
                                             boxstyle="round,pad=0,rounding_size=0.21",
                                             linewidth=0, facecolor=meta["color"], zorder=3))
                ax.text(badge_x + badge_w / 2, row_y1 - 0.39, "추천", ha="center", va="center",
                        fontsize=8.5, color="white", fontweight="bold", zorder=4)

            ax.text(name_x, row_cy - 0.15, won_short(r["currentPrice"]),
                    fontsize=10, color="#374151", va="center", ha="left", zorder=3)
            up_color = "#16A34A" if r["upPct"] >= 55 else "#6B7280"
            ax.text(name_x + 2.6, row_cy - 0.15, f"상승확률 {r['upPct']:.0f}%",
                    fontsize=10, color=up_color, fontweight="bold", va="center", ha="left", zorder=3)

            # entry-position bar (support -> resistance), fixed-size circular marker via scatter
            bar_x0, bar_w = x0 + 0.3, col_w - 0.6
            bar_y = row_y0 + 0.4
            ax.add_patch(plt.Rectangle((bar_x0, bar_y - 0.05), bar_w, 0.1,
                                        facecolor="#E5E7EB", edgecolor="none", zorder=2))
            pos = max(0.0, min(1.0, r["positionRatio"]))
            marker_color = "#16A34A" if pos <= 0.4 else ("#DC2626" if pos >= 0.8 else "#D97706")
            ax.scatter([bar_x0 + bar_w * pos], [bar_y], s=90, color=marker_color,
                       edgecolors="white", linewidths=1.2, zorder=4)
            ax.text(bar_x0, bar_y - 0.35, "지지", fontsize=7.5, color="#9CA3AF", ha="left", va="center", zorder=3)
            ax.text(bar_x0 + bar_w, bar_y - 0.35, "저항", fontsize=7.5, color="#9CA3AF", ha="right", va="center", zorder=3)

    # ---- single shared legend at the bottom ----
    legend_y = 0.55
    ax.scatter([0.5], [legend_y], s=90, color="#16A34A", edgecolors="white", linewidths=1.2, zorder=4)
    ax.text(0.75, legend_y, "지지선 근접(진입양호)", fontsize=8.5, color="#4B5563", va="center", ha="left")
    ax.scatter([4.0], [legend_y], s=90, color="#D97706", edgecolors="white", linewidths=1.2, zorder=4)
    ax.text(4.25, legend_y, "중간 구간", fontsize=8.5, color="#4B5563", va="center", ha="left")
    ax.scatter([6.4], [legend_y], s=90, color="#DC2626", edgecolors="white", linewidths=1.2, zorder=4)
    ax.text(6.65, legend_y, "고점권(진입주의)", fontsize=8.5, color="#4B5563", va="center", ha="left")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def sym_of_market(market):
    return market.replace("KRW-", "")


def won_short(n):
    """Compact price format for chart labels (e.g. 8,200만원 / 1.2억원) - the
    full won() formatter is too wide to fit in the card's narrow columns."""
    if n is None:
        return "-"
    if n >= 1e8:
        return f"{n/1e8:.2f}억"
    if n >= 1e4:
        return f"{n/1e4:.0f}만원"
    return f"{round(n):,}원"


def generate_chart_image(top_picks):
    """top_picks: list of (tier_name, record_dict) - one #1 pick per tier group.
    Draws a simple 30-day price line per coin with support/resistance/current
    price marked, as a quick visual companion to the text summary."""
    n = len(top_picks)
    if n == 0:
        return None

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    tier_label_en = {"대형": "Large", "중형": "Mid", "소형": "Small"}
    for ax, (tier_name, r) in zip(axes, top_picks):
        try:
            candles = fetch_candles(r["market"], 30)
            closes = [c["trade_price"] for c in candles]
            sym = r["market"].replace("KRW-", "")
            ax.plot(range(len(closes)), closes, color="#2563EB", linewidth=1.8)
            ax.axhline(r["support"], color="#E11D48", linestyle="--", linewidth=1, label="Support")
            ax.axhline(r["resistance"], color="#16A34A", linestyle="--", linewidth=1, label="Resistance")
            ax.scatter([len(closes) - 1], [closes[-1]], color="#111827", zorder=5)
            # Korean text isn't guaranteed to render on the default GitHub Actions
            # runner font, so chart titles use English tier labels + ticker symbol.
            ax.set_title(f"#1 {tier_label_en.get(tier_name, tier_name)} - {sym}", fontsize=11)
            ax.legend(fontsize=8, loc="upper left")
            ax.tick_params(labelsize=8)
        except Exception as e:  # noqa: BLE001
            print(f"chart generation failed for {r['market']}: {e}")
            ax.text(0.5, 0.5, "chart failed", ha="center", va="center")

    fig.suptitle("Top-1 pick per tier - last 30 days", fontsize=13)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


WEEKLY_ACC_TARGET = 70


def send_weekly_summary(records, now_ms):
    """A once-a-week (Sunday, KST) roll-up: overall + per-tier accuracy over the
    trailing 7 days (checked against a 70% target), plus which coins got
    recommended most often that week."""
    week_ago = now_ms - 7 * 86400000
    recent_settled = [r for r in records if r.get("settled") and r["targetTimestamp"] >= week_ago]
    recent_logged = [r for r in records if r.get("logTimestamp", 0) >= week_ago]

    if not recent_settled and not recent_logged:
        send_telegram(
            "<b>📅 주간 요약</b>",
            "지난 7일간 기록이 아직 없습니다. 다음 주에 다시 확인해드릴게요.",
        )
        return

    lines = ["📅 <b>주간 요약 (최근 7일)</b>", ""]

    def verdict(hits, total, min_n):
        if total < min_n:
            return f"{hits}/{total}건 ({hits/total*100:.1f}%) · 표본부족(판단보류, {min_n}건 이상 필요)" if total else "만기 기록 없음"
        pct = hits / total * 100
        mark = "✅ 목표달성" if pct >= WEEKLY_ACC_TARGET else "⚠️ 목표미달"
        return f"{hits}/{total}건 ({pct:.1f}%) · {mark} (목표 {WEEKLY_ACC_TARGET}%)"

    if recent_settled:
        overall_hits = sum(1 for r in recent_settled if r.get("hit"))
        lines.append(f"전체 적중률: <b>{verdict(overall_hits, len(recent_settled), 5)}</b>")
        for tier_name in ["대형", "중형", "소형"]:
            tier_recs = [r for r in recent_settled if r["tier"] == tier_name]
            hits = sum(1 for r in tier_recs if r.get("hit"))
            lines.append(f"　{tier_name}: {verdict(hits, len(tier_recs), 3)}")
    else:
        lines.append("전체 적중률: 이번 주에 만기된 기록 없음 (7일 뒤부터 채점됩니다)")
    lines.append("")

    # Most frequently recommended coins this week - repeated appearances can mean
    # a persistent trend, or just the model repeatedly liking the same setup.
    if recent_logged:
        counts = {}
        for r in recent_logged:
            key = (r["market"], r["koName"], r["tier"])
            counts[key] = counts.get(key, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        repeats = [(k, v) for k, v in ranked if v >= 2][:10]
        if repeats:
            lines.append("🔁 <b>이번 주 반복 추천 코인</b>")
            for (market, ko_name, tier_name), cnt in repeats:
                name = html.escape(ko_name)
                lines.append(f"　{name}({market.replace('KRW-', '')}, {tier_name}): {cnt}회")
        else:
            lines.append("🔁 이번 주는 2회 이상 반복 추천된 코인이 없습니다.")
        lines.append("")

    lines.append("⚠️ 표본이 적을 땐 수치가 크게 흔들릴 수 있습니다. 참고용으로만 봐주세요.")
    send_telegram("<b>📅 주간 요약</b>", "\n".join(lines))


def won(n):
    if n is None:
        return "-"
    return f"{round(n):,}원"


def main():
    markets = fetch_upbit_markets()
    all_market_codes = [m["market"] for m in markets]
    tickers = fetch_tickers(all_market_codes)
    tiers = fetch_market_cap_tiers()
    popularity = classify_popularity(tickers)

    def sym_of(market):
        return market.replace("KRW-", "")

    def tier_of(market):
        info = tiers.get(sym_of(market))
        return info["tier"] if info else "초소형"

    def rank_of(market):
        info = tiers.get(sym_of(market))
        return info["rank"] if info and info["rank"] is not None else 9999

    liquid = [m for m in markets if popularity.get(m["market"]) != "거래저조"]

    def group_filter(labels):
        g = [m for m in liquid if tier_of(m["market"]) in labels]
        g.sort(key=lambda m: rank_of(m["market"]))
        return g[:PER_GROUP_CAP]

    groups = {
        "대형": group_filter(["초대형", "대형"]),
        "중형": group_filter(["중형"]),
        "소형": group_filter(["소형", "초소형"]),
    }

    btc_result = simulate("KRW-BTC")
    btc_return_30 = btc_result["coin_return_30"]  # relative-strength baseline still uses Upbit's own BTC/KRW move
    binance_trend = fetch_binance_btc_trend()
    # Binance is the global BTC market (far deeper liquidity than Upbit's KRW
    # market), so its trend takes priority for the regime filter. Falls back
    # to Upbit's own BTC trend only if the Binance fetch fails.
    btc_trend_dir = binance_trend if binance_trend is not None else btc_result["trend_dir"]
    btc_trend_source = "바이낸스" if binance_trend is not None else "업비트(바이낸스 조회 실패)"

    results_by_group = {"대형": [], "중형": [], "소형": []}
    for tier_name, coins in groups.items():
        for m in coins:
            try:
                r = simulate(m["market"])
            except Exception as e:  # noqa: BLE001
                print(f"simulate failed for {m['market']}: {e}")
                continue
            rel_strength = r["coin_return_30"] - btc_return_30
            results_by_group[tier_name].append(
                {
                    "market": m["market"],
                    "koName": m["korean_name"],
                    "sector": get_sector(sym_of(m["market"])),
                    "currentPrice": r["current_price"],
                    "upPct": r["up_pct"],
                    "p50": r["p50"],
                    "trendProjection": r["trend_projection"],
                    "trendDir": r["trend_dir"],
                    "volumeRising": r["volume_rising"],
                    "multiAligned": r["multi_aligned"],
                    "relStrength": rel_strength,
                    "rsi14": r["rsi14"],
                    "resistance": r["resistance"],
                    "support": r["support"],
                    "dayReturn": r["day_return"],
                    "pumpWarning": r["pump_warning"],
                    "positionRatio": r["position_ratio"],
                    "pctAboveSma20": r["pct_above_sma20"],
                }
            )
            time.sleep(0.05)  # be polite to the public API

    fear_greed = fetch_fear_greed()
    fg_value = fear_greed.get("value")
    fg_label = fear_greed.get("label")
    # Market-wide contrarian tilt: buying into broad fear / trimming into broad greed
    # is a well-known (weak, unreliable) contrarian heuristic - applied equally to
    # every coin since it reflects overall market mood, not any single coin's chart.
    fg_bonus = 0
    if fg_value is not None:
        if fg_value <= 25:
            fg_bonus = 5   # Extreme Fear - mild contrarian tilt toward "up"
        elif fg_value >= 75:
            fg_bonus = -5  # Extreme Greed - mild caution tilt

    kimchi = fetch_kimchi_premium()
    kp_pct = kimchi.get("premium_pct")
    # A sharply elevated Korea-vs-global premium has historically tended to
    # unwind (premium collapses back toward 0 even if the global price is
    # fine), so it's treated as a mild caution flag across the board - not a
    # per-coin signal.
    kp_bonus = -5 if (kp_pct is not None and kp_pct >= 5) else 0

    WEIGHTS = load_factor_weights()

    def score_and_sort(arr):
        for r in arr:
            rsi_val = r.get("rsi14")
            btc_bearish = btc_trend_dir == "하락 추세 연장 가능성"
            pos_ratio = r["positionRatio"]

            tags = []
            if r["volumeRising"]:
                tags.append("거래량↑")
            if r["multiAligned"]:
                tags.append("기간정합")
            if r["relStrength"] > 0.05:
                tags.append("BTC대비강세")
            elif r["relStrength"] < -0.05:
                tags.append("BTC대비약세")
            if rsi_val is not None and rsi_val <= 30:
                tags.append("RSI과매도")
            elif rsi_val is not None and rsi_val >= 70:
                tags.append("RSI과매수(주의)")
            if pos_ratio <= 0.4:
                tags.append("지지선근접(진입양호)")
            elif pos_ratio >= 0.8:
                tags.append("고점권(진입주의)")
            if r["pctAboveSma20"] > 0.08:
                tags.append("이평선이격큼(과열)")
            if r["trendDir"] == "상승 추세 연장 가능성" and btc_bearish:
                tags.append("BTC약세국면(신뢰도↓)")
            if r["pumpWarning"]:
                tags.append("⚠️급등주의")

            # Trend direction and pump warnings gate the whole recommendation, so they're
            # scored directly rather than through the adaptive per-tag weight table.
            bonus = fg_bonus + kp_bonus
            if r["trendDir"] == "상승 추세 연장 가능성":
                bonus += 15
            elif r["trendDir"] == "하락 추세 연장 가능성":
                bonus -= 15
            if r["pumpWarning"]:
                bonus -= 20

            # Everything else uses the (possibly self-tuned) weight table - falls back to
            # the static defaults until a factor has enough settled history to measure.
            for tag in tags:
                if tag == "⚠️급등주의":
                    continue  # already penalized above via pumpWarning, avoid double-counting
                bonus += WEIGHTS.get(tag, DEFAULT_FACTOR_WEIGHTS.get(tag, 0))

            r["score"] = r["upPct"] + bonus
            r["recommended"] = (
                r["trendDir"] == "상승 추세 연장 가능성"
                and r["upPct"] >= RECOMMEND_THRESHOLD
                and (rsi_val is None or 25 < rsi_val < 70)
                and not r["pumpWarning"]
                and pos_ratio < 0.75                      # not already sitting near resistance
                and r["pctAboveSma20"] <= 0.08             # not badly overextended vs its own average
                and not (btc_bearish and r["relStrength"] <= 0)  # if BTC's weak, demand relative strength
            )
            r["factorTags"] = tags
        arr.sort(key=lambda r: r["score"], reverse=True)
        return arr[:5]

    top5_by_group = {name: score_and_sort(arr) for name, arr in results_by_group.items()}

    # ---- update prediction tracking state ----
    records = load_state()
    now_ms = int(time.time() * 1000)

    # settle any records whose 7-day window has passed
    settled_lines = []
    for rec in records:
        if not rec.get("settled") and now_ms >= rec["targetTimestamp"]:
            market = rec["market"]
            ticker = tickers.get(market)
            if ticker:
                actual = ticker["trade_price"]
                rec["settled"] = True
                rec["actualPrice"] = actual
                rec["hit"] = actual > rec["priceAtLog"]
                mark = "✅ 적중" if rec["hit"] else "❌ 불일치"
                settled_lines.append(
                    f"• {html.escape(rec['koName'])}({rec['tier']}): {mark} "
                    f"({won(rec['priceAtLog'])} → {won(actual)})"
                )

    # append today's new recommendations
    today_str = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")
    for tier_name, picks in top5_by_group.items():
        for r in picks:
            if not r["recommended"]:
                continue
            records.append(
                {
                    "market": r["market"],
                    "koName": r["koName"],
                    "tier": tier_name,
                    "logDateStr": today_str,
                    "logTimestamp": now_ms,
                    "targetTimestamp": now_ms + HORIZON_DAYS * 86400000,
                    "priceAtLog": r["currentPrice"],
                    "actualPrice": None,
                    "settled": False,
                    "hit": None,
                    "factorTags": r.get("factorTags", []),
                }
            )

    records = records[-500:]  # cap file size
    save_state(records)

    settled_all = [r for r in records if r.get("settled")]
    hits = [r for r in settled_all if r.get("hit")]
    overall_acc = (len(hits) / len(settled_all) * 100) if settled_all else None

    # ---- compose notification: short header text + image card (long per-coin
    #      text blocks moved into the image so this doesn't run on forever) ----
    lines = [f"📊 <b>크립토 추천 스크리너</b>  {today_str}", ""]
    if fg_value is not None:
        fg_emoji = "😱" if fg_value <= 25 else ("🤑" if fg_value >= 75 else "😐")
        lines.append(f"{fg_emoji} 공포탐욕지수: <b>{fg_value}</b> ({fg_label})")
    btc_emoji = "🟢" if btc_trend_dir == "상승 추세 연장 가능성" else ("🔴" if btc_trend_dir == "하락 추세 연장 가능성" else "🟡")
    lines.append(f"{btc_emoji} BTC 국면({btc_trend_source} 기준): {btc_trend_dir}")
    if kp_pct is not None:
        kp_emoji = "🔥" if kp_pct >= 5 else ("🧊" if kp_pct <= -1 else "⚖️")
        lines.append(f"{kp_emoji} 김치프리미엄: {kp_pct:+.2f}% (업비트 vs 바이낸스)")
    lines.append("")

    pick_counts = {t: sum(1 for r in top5_by_group[t] if r["recommended"]) for t in ["대형", "중형", "소형"]}
    total_picks = sum(pick_counts.values())
    lines.append(
        f"🎯 오늘 추천 픽: 대형 {pick_counts['대형']} · 중형 {pick_counts['중형']} · "
        f"소형 {pick_counts['소형']} (총 <b>{total_picks}개</b>)"
    )
    lines.append("👇 그룹별 TOP5 요약 카드는 아래 이미지를 참고하세요.")

    any_warn = False
    for tier_name in ["대형", "중형", "소형"]:
        warn_picks = [r for r in top5_by_group[tier_name] if r["pumpWarning"]]
        if warn_picks:
            if not any_warn:
                lines.append("")
                lines.append("⚠️ <b>단기 급등 주의 (추천 제외)</b>")
                any_warn = True
            for r in warn_picks:
                name = html.escape(r["koName"])
                lines.append(
                    f"　{tier_name} {name}({sym_of(r['market'])}): 전일대비 +{r['dayReturn']*100:.1f}%, 되돌림 위험"
                )

    if settled_lines:
        lines.append("")
        lines.append("🎯 <b>오늘 만기된 과거 추천 결과</b>")
        lines.extend(settled_lines)

    lines.append("")
    if overall_acc is not None:
        lines.append(f"📈 전체 누적 적중률: <b>{overall_acc:.1f}%</b> ({len(hits)}/{len(settled_all)}건)")
    else:
        lines.append("📈 전체 누적 적중률: 아직 없음 (오늘 추천이 7일 뒤부터 순차적으로 채점됩니다)")
    lines.append("")
    lines.append("⚠️ 과거 데이터 기반 통계 모델이며 투자 조언이 아닙니다.")

    message = "\n".join(lines)
    print(message)
    send_telegram("<b>오늘의 크립토 추천 코인</b>", message)

    # ---- summary card image: all three tiers, one glance ----
    try:
        card_img = generate_summary_card(top5_by_group, fg_value, fg_label, btc_trend_dir, kp_pct, today_str)
        if card_img:
            send_telegram_photo(card_img, caption="그룹별 TOP5 요약 카드 (⭐=추천, 점 위치=지지~저항 구간 내 진입 위치)")
    except Exception as e:  # noqa: BLE001
        print(f"summary card step failed: {e}")

    # ---- weekly summary + adaptive weight retuning (Sundays, KST) ----
    try:
        kst_now = datetime.now(timezone.utc) + timedelta(hours=9)
        if kst_now.weekday() == 6:  # Sunday
            send_weekly_summary(records, now_ms)
            new_weights, tuning_report = recompute_factor_weights(records)
            save_factor_weights(new_weights)
            if tuning_report:
                send_telegram(
                    "<b>⚙️ 이번 주 가중치 재조정</b>",
                    "표본이 충분한 요인들의 가중치를 실제 적중률 기준으로 다시 계산했습니다:\n\n" + tuning_report,
                )
    except Exception as e:  # noqa: BLE001
        print(f"weekly summary/tuning failed: {e}")

    # ---- chart image: #1 pick per tier (normal picks only, skip pump-flagged) ----
    try:
        top_picks = []
        for tier_name in ["대형", "중형", "소형"]:
            normal = [r for r in top5_by_group[tier_name] if not r["pumpWarning"]]
            if normal:
                top_picks.append((tier_name, normal[0]))
        img = generate_chart_image(top_picks)
        if img:
            send_telegram_photo(img, caption="그룹별 1위 코인 최근 30일 차트 (지지=빨강, 저항=초록)")
    except Exception as e:  # noqa: BLE001
        print(f"chart step failed: {e}")


if __name__ == "__main__":
    main()
