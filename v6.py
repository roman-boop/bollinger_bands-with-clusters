import pandas as pd
import numpy as np
from binance.client import Client
from scipy.stats import zscore
import time
import datetime
import threading
import telebot
import hmac
import hashlib
import requests
import json

api_key = ""
api_secret = ""

# === НАСТРОЙКИ ===
symbol = "ETHUSDT"
interval = "15m"
bb_period = 40
bb_std = 1.5
STOP_LOSS_PCT = 0.004
QTY_ETH = 0.1
FEE_PCT = 0.0006  # 0.11% на всю сделку (от входа до выхода, суммарно)

bot = telebot.TeleBot(":")
TELEGRAM_CHAT_ID = 

client = Client()
config = {
    'min_cluster': 3,
    'bull_quant': 0.78,
    'bear_quant': 0.22,
    'rsi': 60
}

def safe_send_message(chat_id, text, retry=False):
    try:
        bot.send_message(chat_id, text)
    except Exception as e:
        print(f"[TELEGRAM ERROR] {e}")
        if retry:
            time.sleep(5)
            try:
                bot.send_message(chat_id, text)
            except Exception as e2:
                print(f"[TELEGRAM RETRY FAIL] {e2}")

safe_send_message(TELEGRAM_CHAT_ID, f" Бот запущен! Конфиг {config}")

# === Индикаторы ===
def compute_rsi(df, period=96):
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=1).mean()
    avg_loss = loss.rolling(period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].bfill()
    return df

def compute_csc(df, min_cluster, bull_quant, bear_quant, window=10000):
    df = df.copy()
    df = df.iloc[-window:].reset_index(drop=True)
    
    bull_thr = df['CSI'].quantile(bull_quant)
    bear_thr = df['CSI'].quantile(bear_quant)
    
    df['sentiment'] = np.where(df['CSI'] >= bull_thr, 'bull',
                        np.where(df['CSI'] <= bear_thr, 'bear', 'neutral'))
    df['cluster_id'] = pd.Series(dtype='object')
    curr_type, curr_start, length = None, None, 0
    for i, s in df['sentiment'].items():
        if s == curr_type and s in ['bull', 'bear']:
            length += 1
        else:
            if curr_type in ['bull', 'bear'] and length >= min_cluster:
                df.loc[curr_start:i-1, 'cluster_id'] = f"{curr_type}_{curr_start}"
            if s in ['bull', 'bear']:
                curr_type, curr_start, length = s, i, 1
            else:
                curr_type, length = None, 0
    if curr_type in ['bull', 'bear'] and length >= min_cluster:
        df.loc[curr_start:df.index[-1], 'cluster_id'] = f"{curr_type}_{curr_start}"
    return df

def fetch_klines_paged(symbol=symbol, interval=interval, total_bars=20000, client=None):
    if client is None:
        client = Client()
    limit = 1000
    data = []
    end_time = int(time.time() * 1000)
    while len(data) < total_bars:
        bars_to_fetch = min(limit, total_bars - len(data))
        try:
            klines = client.futures_klines(symbol=symbol, interval=interval, limit=bars_to_fetch, endTime=end_time)
        except Exception as e:
            print("Ошибка Binance API:", e)
            break
        if not klines:
            break
        data = klines + data
        end_time = klines[0][0] - 1
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
    df = df.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
    return df

def append_new_klines(df, symbol=symbol, interval=interval, client=None):
    if client is None:
        client = Client()
    if df.empty:
        return df
    last_close_ms = df['close_time'].iloc[-1] + 1
    new_klines = []
    attempts = 0
    while not new_klines and attempts < 5:
        try:
            new_klines = client.futures_klines(symbol=symbol, interval=interval, startTime=last_close_ms, limit=5)
        except Exception as e:
            print("Ошибка Binance API при аппенде:", e)
            time.sleep(1)
            attempts += 1
            continue
        if not new_klines:
            print("No new klines yet, retrying...")
            time.sleep(1)
            attempts += 1
    if new_klines:
        new_df = pd.DataFrame(new_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')
        new_df[['open','high','low','close','volume']] = new_df[['open','high','low','close','volume']].astype(float)
        df = pd.concat([df, new_df]).drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
    return df

def compute_bollinger(df):
    df['ma'] = df['close'].rolling(bb_period).mean()
    df['std'] = df['close'].rolling(bb_period).std()
    df['upper'] = df['ma'] + bb_std * df['std']
    df['lower'] = df['ma'] - bb_std * df['std']
    return df

def get_csi(df, window = 10000):
    df = df.copy()
    df = df.iloc[-window:].reset_index(drop=True)  # берем только последние window свечей
    
    body = (df['close'] - df['open']).abs()
    rng = (df['high'] - df['low']).replace(0, np.nan)
    body_ratio = body / rng
    direction = np.where(df['close'] > df['open'], 1, -1)
    vol_score = df['volume'] / df['volume'].rolling(50).max()
    range_z = zscore(df['high'] - df['low']).clip(-3, 3)
    tr = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hc': (df['high'] - df['close'].shift(1)).abs(),
        'lc': (df['low'] - df['close'].shift(1)).abs()
    }).max(axis=1)
    atr = tr.rolling(14).mean().bfill()
    df['CSI'] = direction * (0.5 * body_ratio + 0.3 * vol_score + 0.2 * range_z) / atr
    return df

def check_signal_row(row, prev_row):
    if np.isnan(row['lower']) or np.isnan(prev_row['CSI']) or np.isnan(row['CSI']):
        return None
    cluster = row['cluster_id']
    if not isinstance(cluster, str):
        return None
    long_cond = (
        row['close'] < row['lower'] and
        row['CSI'] > 0 and row['CSI'] > prev_row['CSI'] and
        cluster.startswith('bull') and row['RSI'] < config['rsi']
    ) 
    short_cond = (
        row['close'] > row['upper'] and
        row['CSI'] < 0 and row['CSI'] < prev_row['CSI'] and
        cluster.startswith('bear') and row['RSI'] > (100 - config['rsi'])
    )
    if long_cond:
        return 'buy'
    elif short_cond:
        return 'sell'
    return None

def get_server_time_offset():
    url = "https://open-api.bingx.com/openApi/swap/v2/server/time"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()
    if data.get("code") == 0:
        server_time = int(data["data"]["serverTime"])
        local_time = int(time.time() * 1000)
        return server_time - local_time
    else:
        raise Exception(f"Ошибка получения времени сервера: {data}")

time_offset = get_server_time_offset()
print("Time offset (ms):", time_offset)

BASE_URL = "https://open-api.bingx.com"
def _timestamp():
    return int(time.time() * 1000) + time_offset

def _to_bingx_symbol(symbol: str) -> str:
    return symbol.replace("USDT", "-USDT")

def _sign(query: str) -> str:
    return hmac.new(api_secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()

def _request(method: str, path: str, params=None):
    if params is None:
        params = {}
    sorted_keys = sorted(params)
    query = "&".join([f"{k}={params[k]}" for k in sorted_keys if params[k] is not None])
    if query:
        query += "&"
    query += "timestamp=" + str(_timestamp())
    signature = _sign(query)
    url = f"{BASE_URL}{path}?{query}&signature={signature}"
    headers = {"X-BX-APIKEY": api_key}
    r = requests.request(method, url, headers=headers, data={})
    r.raise_for_status()
    return r.json()

def _public_request(path: str, params, timeout: int = 10):
    url = f"{BASE_URL}{path}"
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def place_order(symbol, side, qty_eth):
    position_type = side
    side_text = "BUY" if side == "long" else "SELL"
    position_side = "LONG" if side == "long" else "SHORT"

    params = {
        "symbol": _to_bingx_symbol(symbol),
        "side": side_text,
        "positionSide": position_side,
        "type": "MARKET",
        "quantity": qty_eth,
        "recvWindow": 5000,
        "timeInForce": "GTC",
    }

    try:
        resp = _request("POST", "/openApi/swap/v2/trade/order", params)
        if resp.get("code") != 0:
            error_msg = resp.get("msg", "Неизвестная ошибка")
            safe_send_message(TELEGRAM_CHAT_ID, f"Ошибка открытия сделки: {error_msg} | Full resp: {resp}")
            return None, None
        order_data = resp['data']['order']
        if order_data['status'] == 'FILLED':
            avg_price = float(order_data['avgPrice'])
            safe_send_message(TELEGRAM_CHAT_ID, f"✅ Открыта {side.upper()} позиция на {qty_eth} ETH @ {avg_price:.2f}. {resp}")
            # Теперь размещаем отдельный стоп-ордер
            sl = avg_price * (1 - STOP_LOSS_PCT) if side == "long" else avg_price * (1 + STOP_LOSS_PCT)
            sl_side = "SELL" if side == "long" else "BUY"
            sl_pos_side = "LONG" if side == "long" else "SHORT"
            sl_params = {
                "symbol": _to_bingx_symbol(symbol),
                "type": "STOP_MARKET",
                "side": sl_side,
                "positionSide": sl_pos_side,
                "stopPrice": sl,
                "workingType": "MARK_PRICE",
                "closePosition": True,
                "recvWindow": 5000,
            }
            sl_resp = _request("POST", "/openApi/swap/v2/trade/order", sl_params)
            if sl_resp.get("code") == 0:
                safe_send_message(TELEGRAM_CHAT_ID, f"✅ Стоп-лосс установлен @ {sl:.2f}.")
            else:
                safe_send_message(TELEGRAM_CHAT_ID, f"❗ Ошибка установки стоп-лосса: {sl_resp}")
            return avg_price, sl
        else:
            safe_send_message(TELEGRAM_CHAT_ID, f"❗ Ордер не заполнен: {resp}")
            return None, None
    except Exception as e:
        safe_send_message(TELEGRAM_CHAT_ID, f"❗ Ошибка ордера: {e}")
        return None, None

def close_position(position_type, qty_eth):
    try:
        if position_type == "long":
            params_side = "SELL"
            params_pos_side = "LONG"
        else:
            params_side = "BUY"
            params_pos_side = "SHORT"

        params = {
            "symbol": _to_bingx_symbol(symbol),
            "side": params_side,
            "positionSide": params_pos_side,
            "type": "MARKET",
            "closePosition": "true",
            "recvWindow": 5000,
            "timeInForce": "GTC",
            "quantity": qty_eth,
        }

        resp = _request("POST", "/openApi/swap/v2/trade/order", params)
        print("Close response:", resp)
        if resp.get("code") == 0:
            safe_send_message(TELEGRAM_CHAT_ID, f" Закрыта {position_type.upper()} позиция ({qty_eth} ETH)")
        else:
            safe_send_message(TELEGRAM_CHAT_ID, f"❗ Ошибка закрытия: {resp}")
    except Exception as e:
        safe_send_message(TELEGRAM_CHAT_ID, f"❗ Ошибка при закрытии позиции: {e}")

# === Логика циклов ===
def get_bar_minutes(interval):
    if interval.endswith('m'):
        return int(interval[:-1])
    raise ValueError("Unsupported interval format")

bar_minutes = get_bar_minutes(interval)

def wait_until_next_bar():
    now = datetime.datetime.now(datetime.timezone.utc)
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    minutes_from_day = (now - day_start).total_seconds() / 60
    bar_index = int(minutes_from_day // bar_minutes)
    bar_minute_start = bar_index * bar_minutes
    bar_delta = datetime.timedelta(minutes=bar_minute_start)
    bar_start = day_start + bar_delta
    if (now - bar_start).total_seconds() >= 10:
        bar_minute_start += bar_minutes
        bar_delta = datetime.timedelta(minutes=bar_minute_start)
        bar_start = day_start + bar_delta
    sleep_s = (bar_start - now).total_seconds()
    if sleep_s > 0:
        safe_send_message(TELEGRAM_CHAT_ID, f"⏳ Ожидание следующего бара: {sleep_s:.2f} сек")
        time.sleep(sleep_s + 0.2)
    return bar_start

# === Хранение сделок и статистики ===
trades = []
equity = 0.0
equity_curve = [0.0]
peak_equity = 0.0
max_drawdown = 0.0

def update_stats(entry_price, exit_price, position_type, reason, entry_time, exit_time):
    global equity, equity_curve, peak_equity, max_drawdown, trades
    pnl = ((exit_price - entry_price) / entry_price * 100.0) if position_type == "long" \
          else ((entry_price - exit_price) / entry_price * 100.0)
    net_pnl = pnl - (FEE_PCT * 100.0)
    equity += net_pnl
    equity_curve.append(equity)
    peak_equity = max(peak_equity, equity)
    max_drawdown = max(max_drawdown, peak_equity - equity)

    trade = {
        "entry_time": entry_time,
        "exit_time": exit_time,
        "type": position_type,
        "entry": entry_price,
        "exit": exit_price,
        "pnl": pnl,
        "net_pnl": net_pnl,
        "reason": reason
    }
    trades.append(trade)

    wins = [t for t in trades if t["net_pnl"] > 0]
    losses = [t for t in trades if t["net_pnl"] <= 0]
    win_rate = (len(wins) / len(trades) * 100.0) if trades else 0.0
    profit_factor = (sum(t["net_pnl"] for t in wins) / abs(sum(t["net_pnl"] for t in losses))) if losses else float("inf")

    msg = (
        f" Сделка закрыта ({position_type.upper()}): {reason}\n"
        f"Entry: {entry_price:.2f}, Exit: {exit_price:.2f}\n"
        f"PnL: {pnl:.2f}% | Net: {net_pnl:.2f}%\n\n"
        f"Всего сделок: {len(trades)}\n"
        f"Winrate: {win_rate:.2f}% | PF: {profit_factor:.2f}\n"
        f"Equity: {equity:.2f}% | Max DD: {max_drawdown:.2f}%"
    )
    print(msg)
    safe_send_message(TELEGRAM_CHAT_ID, msg)

if __name__ == '__main__':
    open_trades = []
    last_processed_closed_ts = None
    bar_counter = 0

    df = fetch_klines_paged(symbol, interval, 5000, client)
    safe_send_message(TELEGRAM_CHAT_ID, f" Загружены начальные данные: {len(df)} баров")

    while True:
        try:
            bar_start = wait_until_next_bar()
            df = append_new_klines(df, symbol, interval, client)

            # пересчёт индикаторов
            df = compute_rsi(df)
            df = compute_bollinger(df)
            df = get_csi(df)
            df = compute_csc(df, config['min_cluster'], config['bull_quant'], config['bear_quant'])

            # сигналы
            signals = [None]
            for i in range(1, len(df)):
                signals.append(check_signal_row(df.iloc[i], df.iloc[i-1]))
            df['signal'] = signals

            last_closed = df.iloc[-2]  # FIXED: Use the latest closed bar
            last_closed_ts = last_closed['timestamp']

            if last_processed_closed_ts is not None and pd.Timestamp(last_processed_closed_ts) == pd.Timestamp(last_closed_ts):
                print("Skipping repeat processing of bar")
                continue
            last_processed_closed_ts = last_closed_ts

            bar_counter += 1

            signal = last_closed['signal']
            cur_row = df.iloc[-1]
            cur_ts = cur_row['timestamp']
            cur_idx = bar_counter

            print(f"Current bar ts: {cur_ts}, index: {cur_idx}")
            safe_send_message(TELEGRAM_CHAT_ID, f"Debug: Processing bar {bar_counter}, ts={cur_ts}")

            # === ВЫХОДЫ (проверяем перед входом, чтобы избежать проверки на баре входа) ===
            still_open = []
            for t in open_trades:
                hit_stop = (cur_row['low'] <= t["stop"]) if t["type"] == "long" else (cur_row['high'] >= t["stop"])
                exit_by_time = (cur_idx - t["entry_index"]) >= 3

                print(f"Checking trade {t['type']}: bars_held={cur_idx - t['entry_index']}, exit_by_time={exit_by_time}, hit_stop={hit_stop}")
                safe_send_message(TELEGRAM_CHAT_ID, f"Debug check trade {t['type']}: bars_held={cur_idx - t['entry_index']}, exit_by_time={exit_by_time}, hit_stop={hit_stop}")

                if hit_stop or exit_by_time:
                    reason = 'stop_loss' if hit_stop else 'time_exit'
                    exit_price = t["stop"] if hit_stop else cur_row['close']
                    exit_time = cur_ts
                    if exit_by_time and not hit_stop:
                        threading.Thread(target=close_position, args=(t["type"], QTY_ETH)).start()
                    update_stats(t["entry"], exit_price, t["type"], reason, t["entry_time"], exit_time)
                else:
                    still_open.append(t)

            open_trades = still_open

            # === ВХОД ===
            if signal in ['buy', 'sell']:
                position_type = 'long' if signal == 'buy' else 'short'
                est_entry_price = last_closed['close']  # Estimated for initial calc
                est_stop_price = (est_entry_price * (1 - STOP_LOSS_PCT)) if position_type == 'long' else (est_entry_price * (1 + STOP_LOSS_PCT))
                entry_time = cur_ts

                # Place order and get actual prices
                def place_and_update():
                    actual_entry, actual_stop = place_order(symbol, position_type, QTY_ETH)
                    if actual_entry is not None:
                        # Update the trade with actual values (thread-safe)
                        for tr in open_trades:
                            if tr['entry_time'] == entry_time:  # Match by time
                                tr['entry'] = actual_entry
                                tr['stop'] = actual_stop
                                break

                threading.Thread(target=place_and_update).start()

                trade = {
                    "type": position_type,
                    "entry": est_entry_price,  # Will be updated async with actual
                    "stop": est_stop_price,    # Will be updated async with actual
                    "entry_time": entry_time,
                    "entry_index": bar_counter
                }
                open_trades.append(trade)

                safe_send_message(TELEGRAM_CHAT_ID, f" Вход {position_type.upper()} @ ~{est_entry_price:.2f} | стоп ~{est_stop_price:.2f} (актуальные цены обновятся после филла)")

        except Exception as e:
            safe_send_message(TELEGRAM_CHAT_ID, f"[ОШИБКА] {e}")
            time.sleep(5)