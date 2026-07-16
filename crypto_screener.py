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

import json
import math
import os
import random
import time
from datetime import datetime, timezone

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

    return {
        "current_price": current_price,
        "up_pct": up_pct,
        "p50": p50,
        "trend_projection": trend_projection,
        "trend_dir": trend_dir,
        "volume_rising": volume_rising,
        "coin_return_30": coin_return_30,
        "multi_aligned": multi_aligned,
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
        text = text[:3900] + "\n...(생략)"
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": text},
            timeout=15,
        )
    except Exception as e:  # noqa: BLE001
        print(f"telegram send failed: {e}")


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
                    "currentPrice": r["current_price"],
                    "upPct": r["up_pct"],
                    "trendProjection": r["trend_projection"],
                    "trendDir": r["trend_dir"],
                    "volumeRising": r["volume_rising"],
                    "multiAligned": r["multi_aligned"],
                    "relStrength": rel_strength,
                }
            )
            time.sleep(0.05)  # be polite to the public API

    def score_and_sort(arr):
        for r in arr:
            bonus = 0
            if r["trendDir"] == "상승 추세 연장 가능성":
                bonus += 15
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
            r["score"] = r["upPct"] + bonus
            r["recommended"] = r["trendDir"] == "상승 추세 연장 가능성" and r["upPct"] >= 55
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
                mark = "적중" if rec["hit"] else "불일치"
                settled_lines.append(
                    f"- {rec['koName']}({rec['tier']}): {mark} "
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
    lines = [f"[크립토 추천 스크리너] {today_str}", ""]
    for tier_name in ["대형", "중형", "소형"]:
        lines.append(f"■ {tier_name} TOP5")
        picks = top5_by_group[tier_name]
        if not picks:
            lines.append("  (분석 결과 없음)")
        for i, r in enumerate(picks, 1):
            tag = " ✓추천" if r["recommended"] else ""
            lines.append(
                f"  {i}. {r['koName']}({sym_of(r['market'])}){tag} "
                f"{won(r['currentPrice'])} 상승확률 {r['upPct']:.1f}%"
            )
        lines.append("")

    if settled_lines:
        lines.append("■ 오늘 만기된 과거 추천 결과")
        lines.extend(settled_lines)
        lines.append("")

    if overall_acc is not None:
        lines.append(f"■ 누적 적중률: {overall_acc:.1f}% ({len(hits)}/{len(settled_all)}건)")
    else:
        lines.append("■ 누적 적중률: 아직 만기된 기록 없음")

    lines.append("")
    lines.append("※ 과거 데이터 기반 통계 모델이며 투자 조언이 아닙니다.")

    message = "\n".join(lines)
    print(message)
    send_telegram("오늘의 크립토 추천 코인", message)


if __name__ == "__main__":
    main()
