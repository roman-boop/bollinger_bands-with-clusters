import pandas as pd
import numpy as np
from binance.client import Client
from scipy.stats import zscore
import time
import math
from itertools import product

# === НАСТРОЙКИ ===
symbol = "ETHUSDT"
interval = "15m"
client = Client()
config = {
    'bull_quant': 0.75,
    'bear_quant': 0.25
}

def fetch_klines_paged(symbol='ETHUSDT', interval='15m', total_bars=5000, client=client):
    limit = 1000
    data = []
    current_end = None
    while len(data) < total_bars:
        bars_to_fetch = min(limit, total_bars - len(data))
        try:
            klines = client.futures_klines(
                symbol=symbol.upper(),
                interval=interval,
                limit=bars_to_fetch,
                endTime=current_end
            )
        except Exception as e:
            print("Ошибка Binance API:", e)
            break
        if not klines:
            break
        data = klines + data  # prepend
        current_end = klines[0][0] - 1
        time.sleep(0.1)

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=[
        'timestamp','open','high','low','close','volume',
        'close_time','quote_asset_volume','number_of_trades',
        'taker_buy_base','taker_buy_quote','ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
    df = df.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
    df = df.rename(columns={'timestamp': 'time'})
    return df[["time", "open", "high", "low", "close", "volume"]]

def compute_rsi(df, period=24):
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=1).mean()
    avg_loss = loss.rolling(period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], np.nan).fillna(1)  # Handle div by zero
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50)  # Neutral value
    return df

def compute_csc(df, min_cluster, bull_quant, bear_quant):
    bull_thr = df['CSI'].quantile(bull_quant)
    bear_thr = df['CSI'].quantile(bear_quant)

    df['sentiment'] = np.where(df['CSI'] >= bull_thr, 'bull',
                               np.where(df['CSI'] <= bear_thr, 'bear', 'neutral'))
    df['cluster_id'] = pd.Series(dtype='object')
    curr_type, curr_start, length = None, None, 0

    for i in df.index:
        s = df.at[i, 'sentiment']
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

def compute_bollinger(df, bb_period, bb_std):
    df['ma'] = df['close'].rolling(bb_period).mean()
    df['std'] = df['close'].rolling(bb_period).std()
    df['upper'] = df['ma'] + bb_std * df['std']
    df['lower'] = df['ma'] - bb_std * df['std']
    # Fill NaNs with ma for early bars
    df['upper'] = df['upper'].fillna(df['ma'])
    df['lower'] = df['lower'].fillna(df['ma'])
    return df

def atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr'] = true_range.rolling(period).mean()
    # Fill initial ATR with average TR
    df['atr'] = df['atr'].fillna(true_range.mean())
    return df

def get_csi(df):
    body = (df['close'] - df['open']).abs()
    rng = df['high'] - df['low']
    rng = rng.replace(0, np.nan).fillna(rng.mean())  # Avoid div by zero
    body_ratio = body / rng
    direction = np.where(df['close'] > df['open'], 1, -1)
    vol_max = df['volume'].rolling(50).max().fillna(df['volume'].max())
    vol_score = df['volume'] / vol_max

    rng_col = df['high'] - df['low']
    exp_mean = rng_col.expanding().mean()
    exp_std = rng_col.expanding().std().fillna(1)  # Avoid div by zero
    range_z = ((rng_col - exp_mean) / exp_std).clip(-3, 3)
    range_z = range_z.fillna(0)

    tr = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hc': (df['high'] - df['close'].shift(1)).abs(),
        'lc': (df['low'] - df['close'].shift(1)).abs()
    }).max(axis=1)

    atr_val = tr.rolling(14).mean()
    atr_val = atr_val.fillna(atr_val.mean())  # Fill NaNs
    df['CSI'] = direction * (0.5 * body_ratio + 0.3 * vol_score + 0.2 * range_z) / atr_val
    df['CSI'] = df['CSI'].fillna(0)
    return df

def check_signal_row(row, prev_row, rsi_thr):
    if np.isnan(row['lower']) or np.isnan(prev_row['CSI']) or np.isnan(row['CSI']):
        return None
    cluster = row['cluster_id']
    if not isinstance(cluster, str):
        return None

    long_cond = (
        row['close'] < row['lower'] and
        row['CSI'] > 0 and row['CSI'] > prev_row['CSI'] and
        cluster.startswith('bull') and row['RSI'] < rsi_thr
    )
    short_cond = (
        row['close'] > row['upper'] and
        row['CSI'] < 0 and row['CSI'] < prev_row['CSI'] and
        cluster.startswith('bear') and row['RSI'] > (100 - rsi_thr)
    )

    if long_cond:
        return 'buy'
    elif short_cond:
        return 'sell'
    return None

def run_backtest(df, bb_period, bb_std, rsi_thr, min_cluster, sl_multiplier, tp_multiplier, trailing_multiplier, max_hold_bars):
    min_history = max(bb_period, 100) + 50  # Buffer for indicators
    if len(df) < min_history:
        return None

    in_position = False
    entry_price = None
    entry_index = None
    position_type = None
    sl_price = None
    tp_price = None
    trailing_sl = None
    completed_trades = []

    for i in range(min_history, len(df)):
        hist_df = df.iloc[:i+1].copy()

        # Compute indicators
        hist_df = compute_rsi(hist_df)
        hist_df = compute_bollinger(hist_df, bb_period, bb_std)
        hist_df = atr(hist_df)
        hist_df = get_csi(hist_df)
        hist_df = compute_csc(hist_df, min_cluster, config['bull_quant'], config['bear_quant'])

        row = hist_df.iloc[-1]
        prev_row = hist_df.iloc[-2] if len(hist_df) > 1 else row
        signal = check_signal_row(row, prev_row, rsi_thr)

        if not in_position:
            if signal in ['buy', 'sell']:
                in_position = True
                entry_index = i
                entry_price = row['close']
                position_type = 'long' if signal == 'buy' else 'short'
                current_atr = row['atr']
                
                # ATR-based SL and RR-based TP
                if position_type == 'long':
                    sl_price = entry_price - current_atr * sl_multiplier
                    risk = entry_price - sl_price
                    tp_price = entry_price + risk * tp_multiplier
                    trailing_sl = sl_price
                else:
                    sl_price = entry_price + current_atr * sl_multiplier
                    risk = sl_price - entry_price
                    tp_price = entry_price - risk * tp_multiplier
                    trailing_sl = sl_price

        elif in_position:
            current_low = row['low']
            current_high = row['high']
            current_close = row['close']
            current_open = row['open']
            current_atr = row['atr']
            
            # Update trailing stop only if in profit
            if position_type == 'long' and current_close > entry_price:
                new_trailing = current_close - current_atr * trailing_multiplier
                trailing_sl = max(trailing_sl, new_trailing)
            elif position_type == 'short' and current_close < entry_price:
                new_trailing = current_close + current_atr * trailing_multiplier
                trailing_sl = min(trailing_sl, new_trailing)
            
            # Check exits with simulated slippage
            hit_sl = False
            hit_tp = False
            time_exit = (i - entry_index >= max_hold_bars)
            
            slippage = current_atr * 0.05  # 5% of ATR as slippage
            
            if position_type == 'long':
                if current_low <= trailing_sl:
                    hit_sl = True
                    exit_price = trailing_sl - slippage
                elif current_high >= tp_price:
                    hit_tp = True
                    exit_price = tp_price - slippage
                elif time_exit:
                    exit_price = current_close
            else:
                if current_high >= trailing_sl:
                    hit_sl = True
                    exit_price = trailing_sl + slippage
                elif current_low <= tp_price:
                    hit_tp = True
                    exit_price = tp_price + slippage
                elif time_exit:
                    exit_price = current_close
            
            if hit_sl or hit_tp or time_exit:
                pnl = (
                    (exit_price - entry_price) / entry_price * 100
                    if position_type == 'long'
                    else (entry_price - exit_price) / entry_price * 100
                )
                reason = 'sl' if hit_sl else ('tp' if hit_tp else 'time')
                completed_trades.append({
                    'entry_time': df.iloc[entry_index]['time'],
                    'exit_time': df.iloc[i]['time'],
                    'position_type': position_type,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_%': pnl,
                    'reason': reason
                })
                in_position = False

    trades_df = pd.DataFrame(completed_trades)
    if trades_df.empty:
        return None

    LEVERAGE = 10
    TAKER_FEE = 0.0006
    trades_df['pnl_%'] = trades_df['pnl_%'] * LEVERAGE
    trades_df['fee_%'] = (TAKER_FEE + TAKER_FEE) * LEVERAGE * 100
    trades_df['net_pnl_%'] = trades_df['pnl_%'] - trades_df['fee_%']

    total_net_pnl = trades_df['net_pnl_%'].sum()
    win_rate = (trades_df['net_pnl_%'] > 0).mean() * 100
    profit_factor = trades_df.loc[trades_df['net_pnl_%'] > 0, 'net_pnl_%'].sum() / abs(
        trades_df.loc[trades_df['net_pnl_%'] < 0, 'net_pnl_%'].sum()
    ) if (trades_df['net_pnl_%'] < 0).any() else np.inf
    avg_win = trades_df.loc[trades_df['net_pnl_%'] > 0, 'net_pnl_%'].mean() if any(trades_df['net_pnl_%'] > 0) else 0
    avg_loss = trades_df.loc[trades_df['net_pnl_%'] < 0, 'net_pnl_%'].mean() if any(trades_df['net_pnl_%'] < 0) else 0

    equity_curve = trades_df['net_pnl_%'].cumsum()
    max_drawdown = (equity_curve.cummax() - equity_curve).max() if not equity_curve.empty else 0

    if not trades_df.empty:
        equity_curve.index = pd.to_datetime(trades_df['entry_time'])
        daily_returns = equity_curve.resample('1D').last().pct_change().dropna()
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() != 0 else 0
    else:
        sharpe_ratio = 0

    tp_hits = (trades_df['reason'] == 'tp').sum()
    sl_hits = (trades_df['reason'] == 'sl').sum()
    time_exits = (trades_df['reason'] == 'time').sum()

    return {
        'bb_period': bb_period,
        'bb_std': bb_std,
        'rsi': rsi_thr,
        'min_cluster': min_cluster,
        'sl_mult': sl_multiplier,
        'tp_mult': tp_multiplier,
        'trail_mult': trailing_multiplier,
        'max_hold': max_hold_bars,
        'trades': len(trades_df),
        'winrate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'sharpe': sharpe_ratio,
        'max_dd': max_drawdown,
        'net_pnl': total_net_pnl,
        'tp_hits': tp_hits,
        'sl_hits': sl_hits,
        'time_exits': time_exits
    }

# === Main ===
if __name__ == "__main__":
    df = fetch_klines_paged(symbol, interval)
    print(f"Загружено {len(df)} баров")

    param_grid = {
        'bb_period': [20, 40, 70],
        'bb_std': [1, 1.5, 2],
        'rsi': [60, 70, 80],
        'min_cluster': [2, 3, 4],
        'sl_multiplier': [1.0, 1.5, 2.0],
        'tp_multiplier': [1.5, 2.0, 3.0],
        'trailing_multiplier': [1.0, 1.5, 2.0],
        'max_hold_bars': [10, 15, 20]
    }

    results = []
    for bb_p in param_grid['bb_period']:
        for bb_s in param_grid['bb_std']:
            for rsi_thr in param_grid['rsi']:
                for cluster in param_grid['min_cluster']:
                    for sl_mult in param_grid['sl_multiplier']:
                        for tp_mult in param_grid['tp_multiplier']:
                            for trail_mult in param_grid['trailing_multiplier']:
                                for max_hold in param_grid['max_hold_bars']:
                                    print(f"Testing params: bb_period={bb_p}, bb_std={bb_s}, rsi={rsi_thr}, min_cluster={cluster}, sl_mult={sl_mult}, tp_mult={tp_mult}, trail_mult={trail_mult}, max_hold={max_hold}")
                                    stats = run_backtest(df, bb_p, bb_s, rsi_thr, cluster, sl_mult, tp_mult, trail_mult, max_hold)
                                    if stats:
                                        results.append(stats)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=['net_pnl', 'sharpe', 'profit_factor'], ascending=[False, False, False])

    print("\n===== ТОП-20 КОМБИНАЦИЙ =====")
    print(results_df.head(20))

    results_df.to_csv("param_optimization_prof.csv", index=False)
    print("Results saved to param_optimization_prof.csv")