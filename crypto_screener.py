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

import requests

UPBIT_BASE = "https://api.upbit.com/v1"
COINGECKO_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"
STATE_FILE = os.path.join(os.path.dirname(__file__), "predictions.json")
HORIZON_DAYS = 7
NUM_PATHS = 300
PER_GROUP_CAP = int(os.environ.get("PER_GROUP_CAP", "15"))
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


def linear_slope(arr):
    n = len(arr)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2
    y_mean = sum(arr) / n
    num = sum((i - x_mean) * (arr[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den else 0.0


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


def send_weekly_summary(records, now_ms):
    """A once-a-week (Sunday, KST) roll-up: overall + per-tier accuracy over the
    trailing 7 days, so trends are easier to see than digging through daily messages."""
    week_ago = now_ms - 7 * 86400000
    recent = [r for r in records if r.get("settled") and r["targetTimestamp"] >= week_ago]
    if not recent:
        send_telegram(
            "<b>📅 주간 요약</b>",
            "지난 7일간 만기된 추천 기록이 아직 없습니다. 다음 주에 다시 확인해드릴게요.",
        )
        return

    lines = ["📅 <b>주간 요약 (최근 7일)</b>", ""]
    overall_hits = sum(1 for r in recent if r.get("hit"))
    lines.append(f"전체: <b>{overall_hits}/{len(recent)}건 적중 ({overall_hits/len(recent)*100:.1f}%)</b>")
    lines.append("")
    for tier_name in ["대형", "중형", "소형"]:
        tier_recs = [r for r in recent if r["tier"] == tier_name]
        if not tier_recs:
            lines.append(f"{tier_name}: 만기 기록 없음")
            continue
        hits = sum(1 for r in tier_recs if r.get("hit"))
        lines.append(f"{tier_name}: {hits}/{len(tier_recs)}건 ({hits/len(tier_recs)*100:.1f}%)")
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
    btc_return_30 = btc_result["coin_return_30"]
    btc_trend_dir = btc_result["trend_dir"]

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

    def score_and_sort(arr):
        for r in arr:
            bonus = fg_bonus
            # BTC regime filter: an altcoin "uptrend" signal is less trustworthy
            # when BTC itself is in a downtrend, since alts overwhelmingly
            # correlate with BTC's direction. Full bonus only when BTC agrees
            # (or is at least not clearly bearish).
            btc_bearish = btc_trend_dir == "하락 추세 연장 가능성"
            if r["trendDir"] == "상승 추세 연장 가능성":
                bonus += 5 if btc_bearish else 15
            elif r["trendDir"] == "하락 추세 연장 가능성":
                bonus -= 15
            if r["volumeRising"]:
                bonus += 5
            if r["multiAligned"]:
                bonus += 5
            if r["relStrength"] > 0.05:
                bonus += 5
            elif r["relStrength"] < -0.05:
                bonus -= 5
            # RSI: oversold coins get a small mean-reversion tilt (unless the trend is
            # clearly bearish, in which case "oversold" may just mean "still falling").
            # Overbought coins get a caution penalty - even a strong uptrend is a worse
            # entry when already stretched.
            rsi_val = r.get("rsi14")
            if rsi_val is not None:
                if rsi_val <= 30 and r["trendDir"] != "하락 추세 연장 가능성":
                    bonus += 5
                elif rsi_val >= 70:
                    bonus -= 8
            if r["pumpWarning"]:
                bonus -= 20  # sharp single-day spike on abnormal volume - treat as caution, not signal

            # Entry timing: reward being near support (room to run before resistance),
            # penalize being already stretched near resistance or far above the 20-day
            # average - a "good" trend is still a bad trade at a bad price.
            pos_ratio = r["positionRatio"]
            if pos_ratio <= 0.4:
                bonus += 8
            elif pos_ratio >= 0.8:
                bonus -= 12
            if r["pctAboveSma20"] > 0.08:
                bonus -= 8

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
            if r["trendDir"] == "상승 추세 연장 가능성" and btc_bearish:
                tags.append("BTC약세국면(신뢰도↓)")
            if r["pumpWarning"]:
                tags.append("⚠️급등주의")
            if pos_ratio <= 0.4:
                tags.append("지지선근접(진입양호)")
            elif pos_ratio >= 0.8:
                tags.append("고점권(진입주의)")
            if r["pctAboveSma20"] > 0.08:
                tags.append("이평선이격큼(과열)")
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
                }
            )

    records = records[-500:]  # cap file size
    save_state(records)

    settled_all = [r for r in records if r.get("settled")]
    hits = [r for r in settled_all if r.get("hit")]
    overall_acc = (len(hits) / len(settled_all) * 100) if settled_all else None

    # ---- compose notification ----
    TIER_EMOJI = {"대형": "🔵", "중형": "🟢", "소형": "🟠"}
    lines = [f"📊 <b>크립토 추천 스크리너</b>  {today_str}", ""]
    if fg_value is not None:
        fg_emoji = "😱" if fg_value <= 25 else ("🤑" if fg_value >= 75 else "😐")
        lines.append(f"{fg_emoji} 공포탐욕지수: <b>{fg_value}</b> ({fg_label})")
    btc_emoji = "🟢" if btc_trend_dir == "상승 추세 연장 가능성" else ("🔴" if btc_trend_dir == "하락 추세 연장 가능성" else "🟡")
    lines.append(f"{btc_emoji} BTC 국면: {btc_trend_dir}")
    lines.append("")

    for tier_name in ["대형", "중형", "소형"]:
        emoji = TIER_EMOJI.get(tier_name, "")
        recent_settled = [
            rec for rec in records
            if rec.get("settled") and rec["tier"] == tier_name
            and rec["targetTimestamp"] >= now_ms - 7 * 86400000
        ]
        recent_hits = sum(1 for rec in recent_settled if rec.get("hit"))
        recent_txt = (
            f" (최근7일 적중률 {recent_hits/len(recent_settled)*100:.0f}%, {len(recent_settled)}건)"
            if len(recent_settled) >= 3 else ""
        )
        lines.append(f"{emoji} <b>{tier_name} TOP5</b>{recent_txt}")

        picks = top5_by_group[tier_name]
        normal_picks = [r for r in picks if not r["pumpWarning"]]
        warn_picks = [r for r in picks if r["pumpWarning"]]

        if not normal_picks:
            lines.append("  (분석 결과 없음)")
        for i, r in enumerate(normal_picks, 1):
            tag = " ⭐추천" if r["recommended"] else ""
            rsi_txt = f"{r['rsi14']:.0f}" if r.get("rsi14") is not None else "N/A"
            factor_txt = " · ".join(r["factorTags"]) if r["factorTags"] else "없음"
            name = html.escape(r["koName"])
            lines.append(f"<b>{i}. {name}({sym_of(r['market'])})</b> [{r['sector']}]{tag}")
            lines.append(
                f"　현재가 {won(r['currentPrice'])} | 상승확률 <b>{r['upPct']:.1f}%</b> | {r['trendDir']}"
            )
            lines.append(
                f"　목표가(중앙값/추세) {won(r['p50'])} / {won(r['trendProjection'])}"
            )
            lines.append(f"　지지 {won(r['support'])} ~ 저항 {won(r['resistance'])} | RSI {rsi_txt}")
            lines.append(f"　진입 위치: 지지~저항 구간 내 {r['positionRatio']*100:.0f}% 지점")
            lines.append(f"　보정요인: {factor_txt}")
            lines.append("")

        if warn_picks:
            lines.append("　⚠️ <b>단기 급등 주의 (추천 제외)</b>")
            for r in warn_picks:
                name = html.escape(r["koName"])
                lines.append(
                    f"　- {name}({sym_of(r['market'])}): 전일대비 +{r['dayReturn']*100:.1f}%, "
                    f"거래량 급증 → 되돌림 위험"
                )
            lines.append("")
        lines.append("")

    if settled_lines:
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

    # ---- weekly summary (Sundays, KST) ----
    try:
        kst_now = datetime.now(timezone.utc) + timedelta(hours=9)
        if kst_now.weekday() == 6:  # Sunday
            send_weekly_summary(records, now_ms)
    except Exception as e:  # noqa: BLE001
        print(f"weekly summary failed: {e}")

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
