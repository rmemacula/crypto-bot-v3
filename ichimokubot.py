from typing import Optional

import os
import json
import time
import logging
from datetime import datetime, timezone, timedelta

import requests
import pandas as pd
import numpy as np

from telegram.ext import Updater, CommandHandler, CallbackContext

# ================== CONFIG ==================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = int(os.getenv("CHAT_ID", "0"))

if not TELEGRAM_TOKEN:
    raise RuntimeError("Missing TELEGRAM_TOKEN env var")
if not CHAT_ID:
    raise RuntimeError("Missing CHAT_ID env var")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ================== COINS ==================
MANUAL_TOP_10 = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "TRXUSDT", "LINKUSDT",
]

TIMEFRAMES = {"1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w"}
DATA_FILE = "last_signals.json"

# ================== BINANCE HELPERS ==================

def fetch_top_volume_pairs(limit: int = 30) -> list[str]:
    """Top N USDT perpetual symbols by 24h quote volume."""
    url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data)
        df = df[df["symbol"].str.endswith("USDT")]
        df["quoteVolume"] = df["quoteVolume"].astype(float)
        top = df.sort_values("quoteVolume", ascending=False).head(limit)
        return top["symbol"].tolist()
    except Exception as e:
        logging.error("Error fetching top volume pairs: %s", e)
        return []

TF_MS = {
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}

def candle_open_time_ms(df: pd.DataFrame, closed: bool = True) -> Optional[int]:
    """
    closed=True -> last CLOSED candle open time (df[-2])
    closed=False -> current FORMING candle open time (df[-1])
    """
    if df is None or len(df) < 2:
        return None
    idx = -2 if closed else -1
    return int(df.iloc[idx]["time"])

def candle_close_time_ms(open_ms: int, tf_label: str) -> int:
    return int(open_ms + TF_MS[tf_label])

def fmt_ms(ms: int, tz=timezone(timedelta(hours=8))) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=tz).strftime("%Y-%m-%d %I:%M %p")

def load_symbols() -> list[str]:
    manual = set(MANUAL_TOP_10)
    top_vol = set(fetch_top_volume_pairs(30))
    combined = sorted(list(manual | top_vol))
    logging.info("✅ Loaded %s symbols (Top 10 + Top 30 by volume).", len(combined))
    return combined


def fetch_ohlcv(symbol: str, interval: str, limit: int = 150) -> Optional[pd.DataFrame]:
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        df = pd.DataFrame(
            data,
            columns=["time", "o", "h", "l", "c", "v", "ct", "qv", "n", "tb", "tq", "ig"],
        )
        for col in ["o", "h", "l", "c", "v"]:
            df[col] = df[col].astype(float)
        return df
    except Exception:
        return None


# ================== TECHNICALS ==================

def detect_crt_on_last_pair(df: pd.DataFrame) -> dict:
    """Simple CRT: sweep prev high/low then close back inside prev range."""
    if df is None or len(df) < 3:
        return {"bullish_crt": False, "bearish_crt": False, "prev_high": None, "prev_low": None,
                "curr_high": None, "curr_low": None, "curr_close": None}

    prev, curr = df.iloc[-3], df.iloc[-2]
    prev_open, prev_close, prev_high, prev_low = map(float, [prev["o"], prev["c"], prev["h"], prev["l"]])
    curr_close, curr_high, curr_low = map(float, [curr["c"], curr["h"], curr["l"]])

    bullish_crt = prev_close < prev_open and (curr_low < prev_low) and (prev_low < curr_close < prev_high)
    bearish_crt = prev_close > prev_open and (curr_high > prev_high) and (prev_low < curr_close < prev_high)

    return {
        "bullish_crt": bullish_crt,
        "bearish_crt": bearish_crt,
        "prev_high": prev_high,
        "prev_low": prev_low,
        "curr_high": curr_high,
        "curr_low": curr_low,
        "curr_close": curr_close,
    }


def analyze_df(df: pd.DataFrame) -> dict:
    # Needs enough candles for Ichimoku (52 + 26 shift) + stable RSI
    if df is None or len(df) < 104:
        return {
            "signal": "Neutral",
            "price": None,
            "rsi": None,
            "bull_count": 0,
            "bear_count": 0,
            "checklist_bull": [],
            "checklist_bear": [],
            "sl": None,
            "tp": None,
            "kijun_live": None,  # ✅ add
        }

    close = df["c"]
    high = df["h"]
    low = df["l"]

    # ----- Ichimoku core -----
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2

    # Unshifted Senkou series (we will index/shift as needed)
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (high.rolling(52).max() + low.rolling(52).min()) / 2

    # ----- RSI (14) -----
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    last_idx = -2  # last CLOSED candle (avoid using forming candle)
    price = float(close.iloc[last_idx])

    rsi_raw = rsi.iloc[last_idx]
    rsi_val = float(rsi_raw) if pd.notna(rsi_raw) else None

    tenkan_raw = tenkan.iloc[last_idx]
    kijun_raw = kijun.iloc[last_idx]
    tenkan_v = float(tenkan_raw) if pd.notna(tenkan_raw) else None
    kijun_v = float(kijun_raw) if pd.notna(kijun_raw) else None
    kijun_live_raw = kijun.iloc[-1]  # ✅ forming candle kijun
    kijun_live = float(kijun_live_raw) if pd.notna(kijun_live_raw) else None


    # ----- "Current" cloud at current candle -----
    # Cloud is plotted 26 periods forward, so to compare price NOW vs cloud,
    # compare price at last_idx against senkou values at (last_idx - 26).
    cloud_idx = last_idx - 26
    cloud_a_raw = senkou_a.iloc[cloud_idx]
    cloud_b_raw = senkou_b.iloc[cloud_idx]
    cloud_a_current = float(cloud_a_raw) if pd.notna(cloud_a_raw) else None
    cloud_b_current = float(cloud_b_raw) if pd.notna(cloud_b_raw) else None

    price_above_cloud = (
        cloud_a_current is not None
        and cloud_b_current is not None
        and price > max(cloud_a_current, cloud_b_current)
    )
    price_below_cloud = (
        cloud_a_current is not None
        and cloud_b_current is not None
        and price < min(cloud_a_current, cloud_b_current)
    )

    # ----- Chikou vs cloud (FIXED) -----
    # Chikou = today's close plotted 26 back
    # Compare current price to the cloud at the position 26 periods back
    # That cloud was projected from 52 periods back (26 + 26)
    chikou_cloud_idx = last_idx - 52  # ✅ FIXED: Go back 52 total

    chikou_a_raw = senkou_a.iloc[chikou_cloud_idx]
    chikou_b_raw = senkou_b.iloc[chikou_cloud_idx]
    chikou_a = float(chikou_a_raw) if pd.notna(chikou_a_raw) else None
    chikou_b = float(chikou_b_raw) if pd.notna(chikou_b_raw) else None

    chikou_above = chikou_a is not None and chikou_b is not None and price > max(chikou_a, chikou_b)
    chikou_below = chikou_a is not None and chikou_b is not None and price < min(chikou_a, chikou_b)

    # ----- Future cloud bias (FIXED) -----
    # The cloud 26 periods ahead is represented by CURRENT Senkou calculations
    future_a_raw = senkou_a.iloc[last_idx]  # ✅ FIXED: Use current, not shifted
    future_b_raw = senkou_b.iloc[last_idx]

    future_cloud_bullish = pd.notna(future_a_raw) and pd.notna(future_b_raw) and float(future_a_raw) > float(future_b_raw)
    future_cloud_bearish = pd.notna(future_a_raw) and pd.notna(future_b_raw) and float(future_a_raw) < float(future_b_raw)

    # ----- Checklist -----
    checklist_bull = [
        ("Price above cloud", price_above_cloud),
        ("Tenkan > Kijun", (tenkan_v is not None and kijun_v is not None and tenkan_v > kijun_v)),
        ("Chikou above cloud", chikou_above),
        ("Future cloud bullish", future_cloud_bullish),
    ]
    checklist_bear = [
        ("Price below cloud", price_below_cloud),
        ("Tenkan < Kijun", (tenkan_v is not None and kijun_v is not None and tenkan_v < kijun_v)),
        ("Chikou below cloud", chikou_below),
        ("Future cloud bearish", future_cloud_bearish),
    ]

    bull_count = sum(bool(v) for _, v in checklist_bull)
    bear_count = sum(bool(v) for _, v in checklist_bear)

    # ----- Signal + simple SL/TP -----
    signal = "Neutral"
    sl = None
    tp = None

    if bull_count >= 3 and cloud_a_current is not None and cloud_b_current is not None:
        signal = "BUY"
        sl = min(cloud_a_current, cloud_b_current) * 0.995
        tp = price + 2 * (price - sl)

    elif bear_count >= 3 and cloud_a_current is not None and cloud_b_current is not None:
        signal = "SELL"
        sl = max(cloud_a_current, cloud_b_current) * 1.005
        tp = price - 2 * (sl - price)

    crt = detect_crt_on_last_pair(df)

    return {
        "price": price,
        "rsi": rsi_val,
        "signal": signal,
        "bull_count": bull_count,
        "bear_count": bear_count,
        "checklist_bull": checklist_bull,
        "checklist_bear": checklist_bear,
        "sl": sl,
        "tp": tp,
        "kijun_live": kijun_live,  # ✅ add
        "crt_bull": crt.get("bullish_crt", False),
        "crt_bear": crt.get("bearish_crt", False),
        "crt_prev_high": crt.get("prev_high"),
        "crt_prev_low": crt.get("prev_low"),
        "crt_curr_high": crt.get("curr_high"),
        "crt_curr_low": crt.get("curr_low"),
        "crt_curr_close": crt.get("curr_close"),
    }
def format_checklist(a: dict) -> str:
    lines = []
    sig = a.get("signal")
    bull = a.get("checklist_bull") or []
    bear = a.get("checklist_bear") or []

    for (bull_label, bull_val), (bear_label, bear_val) in zip(bull, bear):
        if sig == "BUY":
            lines.append(f"{'✅' if bull_val else '❌'} {bull_label}")
        elif sig == "SELL":
            lines.append(f"{'✅' if bear_val else '❌'} {bear_label}")

    if a.get("crt_bull"):
        lines.append("🕯️ CRT: Bullish (swept prior low, closed back inside)")
    elif a.get("crt_bear"):
        lines.append("🕯️ CRT: Bearish (swept prior high, closed back inside)")

    return "\n".join(lines)


def tradingview_link(symbol: str, tf_label: str) -> str:
    tf_map = {"1h": "60", "4h": "240", "1d": "1D", "1w": "1W"}
    tv_symbol = f"{symbol}.P"
    return f"https://www.tradingview.com/chart/?symbol=BINANCE:{tv_symbol}&interval={tf_map.get(tf_label, '60')}"


# ================== STATE ==================

def load_last_signals() -> dict:
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_last_signals(data: dict) -> None:
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f)


def key_for(symbol: str, tf: str) -> str:
    return f"{symbol}_{tf}"

def kijun_touch_key(symbol: str, tf: str, direction: str) -> str:
    # direction = "BUY" or "SELL" to avoid confusion if direction flips later
    return f"{symbol}_{tf}_KIJUN_TOUCH_{direction}"

def current_candle_open_time_ms(df: pd.DataFrame) -> Optional[int]:
    if df is None or len(df) < 2:
        return None
    return int(df.iloc[-1]["time"])  # candle OPEN time in ms

def is_kijun_hit_intrabar(df: pd.DataFrame, kijun_live: Optional[float]) -> bool:
    """
    LIVE touch: current candle (forming) high/low contains live kijun value.
    """
    if df is None or len(df) < 2 or kijun_live is None:
        return False
    curr = df.iloc[-1]
    hi = float(curr["h"])
    lo = float(curr["l"])
    k = float(kijun_live)
    return lo <= k <= hi

def live_price(df: pd.DataFrame) -> Optional[float]:
    if df is None or len(df) < 1:
        return None
    return float(df.iloc[-1]["c"])

def aligned_strong_min2(symbol: str) -> Optional[dict]:
    """
    Returns dict if >=2 of (1h, 4h, 1d) are STRONG 4/4 in the SAME direction.
    """
    need = [("1h", TIMEFRAMES["1h"]), ("4h", TIMEFRAMES["4h"]), ("1d", TIMEFRAMES["1d"])]
    strong = {}  # tf_label -> info

    for tf_label, interval in need:
        df = fetch_ohlcv(symbol, interval)
        if df is None or len(df) < 104:
            continue

        a = analyze_df(df)
        sig = a.get("signal")

        is_strong_buy = sig == "BUY" and a.get("bull_count", 0) == 4
        is_strong_sell = sig == "SELL" and a.get("bear_count", 0) == 4
        if not (is_strong_buy or is_strong_sell):
            continue

        open_ms = candle_open_time_ms(df, closed=True)  # last CLOSED candle open
        if open_ms is None:
            continue
        close_ms = candle_close_time_ms(open_ms, tf_label)

        strong[tf_label] = {
            "signal": sig,          # BUY/SELL
            "analysis": a,          # analysis object for that TF
            "open_ms": open_ms,
            "close_ms": close_ms,
        }

    if len(strong) < 2:
        return None

    # Pick direction that has >=2 strong TFs
    buys = [tf for tf, v in strong.items() if v["signal"] == "BUY"]
    sells = [tf for tf, v in strong.items() if v["signal"] == "SELL"]

    if len(buys) >= 2 and len(buys) > len(sells):
        direction = "BUY"
        aligned_tfs = sorted(buys)
    elif len(sells) >= 2 and len(sells) > len(buys):
        direction = "SELL"
        aligned_tfs = sorted(sells)
    elif len(buys) >= 2 and len(sells) >= 2 and len(buys) == len(sells):
        # ambiguous (2 BUY + 2 SELL can't happen here since max is 3 TFs, but keep it safe)
        return None
    else:
        return None

    return {
        "symbol": symbol,
        "direction": direction,
        "aligned_tfs": aligned_tfs,  # e.g. ["1h","4h"] or ["1h","4h","1d"]
        "tfs": strong,              # contains info for each TF that is strong
    }
# ================== GLOBALS ==================
TOP_VOLUME_SYMBOLS = set(fetch_top_volume_pairs(30))
SYMBOLS = load_symbols()
LAST_SIGNALS = load_last_signals()


def volume_tag(symbol: str) -> str:
    return " 🔥 High Volume" if symbol in TOP_VOLUME_SYMBOLS else ""


def refresh_pairs(context: CallbackContext):
    global SYMBOLS, TOP_VOLUME_SYMBOLS
    logging.info("🔄 Refreshing symbol list (Top 10 + Top 30)...")
    new_symbols = load_symbols()
    TOP_VOLUME_SYMBOLS = set(fetch_top_volume_pairs(30))
    added = set(new_symbols) - set(SYMBOLS)
    removed = set(SYMBOLS) - set(new_symbols)
    SYMBOLS = new_symbols

    msg = f"♻️ Updated symbols: {len(SYMBOLS)} total."
    if added:
        msg += f"\n➕ Added: {', '.join(sorted(list(added))[:10])}..."
    if removed:
        msg += f"\n➖ Removed: {', '.join(sorted(list(removed))[:10])}..."

    try:
        context.bot.send_message(chat_id=CHAT_ID, text=msg)
    except Exception:
        pass


# ================== COMMANDS ==================

def cmd_test(update, context):
    update.message.reply_text("✅ Ichimoku bot is working!")


def cmd_status(update, context):
    if not context.args:
        return update.message.reply_text("Usage: /status BTC")

    sym = context.args[0].upper() + "USDT"
    if sym not in SYMBOLS:
        return update.message.reply_text("Unknown coin")

    messages = []
    for tf_label, interval in TIMEFRAMES.items():
        df = fetch_ohlcv(sym, interval)
        if df is None or len(df) < 104:
            continue
        a = analyze_df(df)
        msg = (
            f"📊 {sym} ({tf_label}){volume_tag(sym)}\n"
            f"Signal: {a['signal']}\n"
            f"💰 Price: {a['price']:.4f} USDT\n"
            f"📊 RSI: {a['rsi']:.2f}\n"
            f"📈 [View on TradingView]({tradingview_link(sym, tf_label)})\n"
        )
        if a.get("sl") is not None and a.get("tp") is not None:
            msg += f"🎯 SL: {a['sl']:.4f} | TP: {a['tp']:.4f}\n"
        msg += "\n" + format_checklist(a)
        messages.append(msg)

    update.message.reply_text("\n\n".join(messages), parse_mode="Markdown", disable_web_page_preview=True)


def _scan_strong(update, tf_label: str):
    interval = TIMEFRAMES[tf_label]
    update.message.reply_text(f"⏳ Scanning {tf_label.upper()} Ichimoku + CRT ...")

    buy_msgs, sell_msgs = [], []
    manila_tz = timezone(timedelta(hours=8))

    for sym in SYMBOLS:
        df = fetch_ohlcv(sym, interval)
        if df is None or len(df) < 104:
            continue
        a = analyze_df(df)

        is_strong_buy = a["signal"] == "BUY" and a.get("bull_count", 0) == 4
        is_strong_sell = a["signal"] == "SELL" and a.get("bear_count", 0) == 4
        if not (is_strong_buy or is_strong_sell):
            continue

        ts = datetime.fromtimestamp(df.iloc[-2]["time"] / 1000, tz=manila_tz).strftime("%Y-%m-%d %I:%M %p")
        msg = (
            f"{'🟩' if is_strong_buy else '🟥'} *{sym}* — STRONG {a['signal']} (4/4){volume_tag(sym)}\n"
            f"🕒 Time: {ts}\n"
            f"💰 Price: {a['price']:.4f}\n"
            f"📊 RSI: {a['rsi']:.2f}\n"
            f"🔗 [TradingView]({tradingview_link(sym, tf_label)})\n\n"
            f"{format_checklist(a)}"
        )
        (buy_msgs if is_strong_buy else sell_msgs).append(msg)

    def send_in_batches(msgs, title):
        if not msgs:
            return
        batch_size = 5
        for i in range(0, len(msgs), batch_size):
            batch = msgs[i : i + batch_size]
            header = f"{title} ({i+1}-{min(i+batch_size, len(msgs))} of {len(msgs)})\n\n"
            update.message.reply_text(
                header + "\n\n".join(batch),
                parse_mode="Markdown",
                disable_web_page_preview=True,
            )

    send_in_batches(buy_msgs, f"🟩 *STRONG BUYs ({tf_label.upper()})*")
    send_in_batches(sell_msgs, f"🟥 *STRONG SELLs ({tf_label.upper()})*")
    if not buy_msgs and not sell_msgs:
        update.message.reply_text(f"⚪ No {tf_label.upper()} 4/4 signals found.")
    update.message.reply_text(f"✅ {tf_label.upper()} scan complete.")


def cmd_status1d(update, context):
    _scan_strong(update, "1d")


def cmd_status1w(update, context):
    _scan_strong(update, "1w")


def cmd_statusaligned(update, context):
    update.message.reply_text("⏳ Scanning aligned STRONG signals (1h, 4h, 1D)...")

    tf_scan = [("1h", TIMEFRAMES["1h"]), ("4h", TIMEFRAMES["4h"]), ("1d", TIMEFRAMES["1d"])]
    aligned_buys = []
    aligned_sells = []

    for sym in SYMBOLS:
        buy_tfs, sell_tfs = [], []
        for tf_label, interval in tf_scan:
            df = fetch_ohlcv(sym, interval)
            if df is None or len(df) < 104:
                continue
            a = analyze_df(df)
            if a["signal"] == "BUY" and a.get("bull_count", 0) == 4:
                buy_tfs.append(tf_label)
            elif a["signal"] == "SELL" and a.get("bear_count", 0) == 4:
                sell_tfs.append(tf_label)

        if len(buy_tfs) >= 2:
            aligned_buys.append((sym, buy_tfs))
        if len(sell_tfs) >= 2:
            aligned_sells.append((sym, sell_tfs))

    aligned_buys.sort(key=lambda x: (-len(x[1]), x[0]))
    aligned_sells.sort(key=lambda x: (-len(x[1]), x[0]))

    buy_lines = [
        f"🟩 *{sym}* — STRONG BUY aligned: *{', '.join(tfs)}*{volume_tag(sym)}\n"
        f"🔗 [TradingView (1H)]({tradingview_link(sym, '1h')})"
        for sym, tfs in aligned_buys
    ]

    sell_lines = [
        f"🟥 *{sym}* — STRONG SELL aligned: *{', '.join(tfs)}*{volume_tag(sym)}\n"
        f"🔗 [TradingView (1H)]({tradingview_link(sym, '1h')})"
        for sym, tfs in aligned_sells
    ]

    def send_batches(lines, title):
        if not lines:
            return
        batch_size = 20
        for i in range(0, len(lines), batch_size):
            batch = lines[i : i + batch_size]
            header = f"{title} ({i+1}-{min(i+batch_size, len(lines))} of {len(lines)})\n\n"
            update.message.reply_text(
                header + "\n".join(batch),
                parse_mode="Markdown",
                disable_web_page_preview=True,
            )

    if not buy_lines and not sell_lines:
        update.message.reply_text("⚪ No coins found with >=2 aligned STRONG signals (1h/4h/1D).")
        return

    send_batches(buy_lines, "🟩 *ALIGNED STRONG BUYs (>=2 TFs)*")
    send_batches(sell_lines, "🟥 *ALIGNED STRONG SELLs (>=2 TFs)*")
    update.message.reply_text("✅ /statusaligned scan complete.")


def cmd_statusvolume(update, context):
    if not TOP_VOLUME_SYMBOLS:
        return update.message.reply_text("⚠️ No top volume data available yet.")

    msg = "🔥 *Top 30 High-Volume Coins Currently Monitored*\n\n"
    for i, sym in enumerate(sorted(list(TOP_VOLUME_SYMBOLS)), 1):
        msg += f"{i:02d}. {sym}\n"
    msg += "\n♻️ This list refreshes automatically every 4 hours."
    update.message.reply_text(msg, parse_mode="Markdown")


# ================== ALERT JOB ==================

def check_and_alert(context: CallbackContext):
    """
    ALERT JOB (FINAL):
    - Runs frequently (e.g., every 30s) but ONLY does work once per NEW 1H candle (i.e., at 1H close).
    - Sends alerts ONLY for /statusaligned-style alignment (>=2 TFs out of 1H/4H/1D are STRONG 4/4 and SAME direction).
    - Alerts when:
        * First time aligned
        * Direction flips (BUY<->SELL)
        * Alignment improves (2 TFs -> 3 TFs)
    - No more per-timeframe alerts (1h/4h/1d/1w).
    """
    global LAST_SIGNALS, SYMBOLS

    bot = context.bot
    manila_tz = timezone(timedelta(hours=8))

    # ----------------------------
    # 1) Gate: run only once per new 1H candle
    # ----------------------------
    df_ref = fetch_ohlcv("BTCUSDT", TIMEFRAMES["1h"], limit=3)
    if df_ref is None or len(df_ref) < 3:
        return

    # New 1H candle open time (forming candle). When this changes, the previous 1H closed.
    current_1h_open_ms = int(df_ref.iloc[-1]["time"])
    scan_key = "__LAST_1H_OPEN_SCANNED__"
    last_scanned = LAST_SIGNALS.get(scan_key)

    if last_scanned == current_1h_open_ms:
        return  # still same 1H candle, do nothing

    # Mark scanned (so we don't scan again until next 1H candle)
    LAST_SIGNALS[scan_key] = current_1h_open_ms
    save_last_signals(LAST_SIGNALS)

    # The 1H close time that triggered this scan = new candle open time
    tick_close_ms = current_1h_open_ms
    tick_close_str = datetime.fromtimestamp(tick_close_ms / 1000, tz=manila_tz).strftime("%Y-%m-%d %I:%M %p")

    logging.info("⏰ 1H candle closed at %s — running AUTO /statusaligned (>=2 TFs)...", tick_close_str)

    # ----------------------------
    # 2) Scan symbols for >=2 TF aligned (1H/4H/1D) STRONG 4/4 same direction
    # ----------------------------
    aligned_cards = []

    for sym in SYMBOLS:
        out = aligned_strong_min2(sym)   # <-- must exist (the >=2 TF function)
        if not out:
            continue

        direction = out["direction"]           # "BUY" or "SELL"
        aligned_tfs = out["aligned_tfs"]       # list like ["1h","4h"] or ["1h","4h","1d"]
        tfs = out["tfs"]                       # dict with per-tf {"analysis","close_ms",...}

        # ----------------------------
        # 3) Spam control: only alert on first / flip / 2->3 improvement
        # ----------------------------
        state_key = f"{sym}_ALIGNED_MIN2"
        prev_state = LAST_SIGNALS.get(state_key)          # e.g. "BUY|2"
        now_state = f"{direction}|{len(aligned_tfs)}"     # direction + aligned count

        send_it = False
        if not prev_state:
            send_it = True
        else:
            parts = prev_state.split("|")
            prev_dir = parts[0] if len(parts) > 0 else None
            prev_n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0

            if prev_dir != direction:
                send_it = True
            elif len(aligned_tfs) > prev_n:
                send_it = True

        if not send_it:
            continue

        # ----------------------------
        # 4) Build message
        #    Use 1H metrics if 1H is aligned; else use first aligned TF
        # ----------------------------
        base_tf = "1h" if "1h" in aligned_tfs else aligned_tfs[0]
        base_a = tfs[base_tf]["analysis"]

        time_lines = []
        for tf in aligned_tfs:
            close_ms = tfs[tf]["close_ms"]
            close_str = datetime.fromtimestamp(close_ms / 1000, tz=manila_tz).strftime("%Y-%m-%d %I:%M %p")
            time_lines.append(f"🕒 *{tf.upper()} strong candle close:* {close_str}")

        msg = (
            f"{'🟩' if direction=='BUY' else '🟥'} *{sym}* — *ALIGNED STRONG {direction} (>=2 TFs)*{volume_tag(sym)}\n"
            f"✅ *Aligned TFs:* {', '.join([tf.upper() for tf in aligned_tfs])}\n\n"
            f"✅ *Alignment ticked at (1H close):* {tick_close_str}\n\n"
            + "\n".join(time_lines)
            + f"\n\n💰 *Price ({base_tf.upper()} closed):* {base_a['price']:.4f}\n"
        )

        rsi = base_a.get("rsi")
        msg += f"📊 *RSI ({base_tf.upper()}):* {rsi:.2f}\n" if isinstance(rsi, (int, float)) else f"📊 *RSI ({base_tf.upper()}):* N/A\n"
        msg += f"🔗 [TradingView (1H)]({tradingview_link(sym, '1h')})\n\n"
        msg += f"*{base_tf.upper()} Checklist*\n" + format_checklist(base_a)

        aligned_cards.append(msg)

        # Save state so it won't spam until flip or 2->3 improvement
        LAST_SIGNALS[state_key] = now_state

    # ----------------------------
    # 5) Send in batches + persist
    # ----------------------------
    if aligned_cards:
    for msg in aligned_cards:
        full_msg = (
            f"📌 *AUTO /statusaligned (>=2 TFs)* — 1H close: {tick_close_str}\n\n"
            + msg
        )
        try:
            bot.send_message(
                chat_id=CHAT_ID,
                text=full_msg,
                parse_mode="Markdown",
                disable_web_page_preview=True,
            )
            time.sleep(0.5)  # small delay to avoid Telegram flood
        except Exception as e:
            logging.error("Failed to send aligned alert: %s", e)

    save_last_signals(LAST_SIGNALS)
    logging.info("✅ Sent %d separate aligned alerts.", len(aligned_cards))


        save_last_signals(LAST_SIGNALS)
        logging.info("✅ Auto aligned scan sent %d alerts.", len(aligned_cards))
    else:
        logging.info("⚪ No aligned (>=2 TFs) strong signals on this 1H close.")

def heartbeat(context: CallbackContext):
    try:
        context.bot.send_message(chat_id=CHAT_ID, text="💓 Bot is alive")
    except Exception:
        pass

# ================== MAIN ==================

def main():
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("test", cmd_test))
    dp.add_handler(CommandHandler("status", cmd_status))
    dp.add_handler(CommandHandler("status1d", cmd_status1d))
    dp.add_handler(CommandHandler("status1w", cmd_status1w))
    dp.add_handler(CommandHandler("statusvolume", cmd_statusvolume))
    dp.add_handler(CommandHandler("statusaligned", cmd_statusaligned))

    jq = updater.job_queue
    jq.run_repeating(check_and_alert, interval=30, first=10)
    jq.run_repeating(heartbeat, interval=14400, first=20)
    jq.run_repeating(refresh_pairs, interval=14400, first=60)

    updater.start_polling()
    updater.bot.send_message(chat_id=CHAT_ID, text="🚀 Ichimoku bot restarted and running!")
    updater.idle()


if __name__ == "__main__":
    main()
