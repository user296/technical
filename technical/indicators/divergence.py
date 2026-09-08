# Ported from "Divergence for many indicator v3" by LonesomeTheBlue,
# licensed under the Mozilla Public License 2.0: https://mozilla.org/MPL/2.0/
import numpy as np
import pandas as pd
import talib.abstract as ta

from technical.indicators.overlap_studies import vwma as _sma_vwma


def _pivot(series: pd.Series, left: int, right: int, mode: str) -> pd.Series:
    window = left + right + 1

    def _check(x):
        c = x[left]
        left_part, right_part = x[:left], x[left + 1 :]
        if mode == "high":
            ok = (left_part <= c).all() and (right_part < c).all()
        else:
            ok = (left_part >= c).all() and (right_part > c).all()
        return c if ok else np.nan

    return series.rolling(window).apply(_check, raw=True)


def _rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / length, adjust=False).mean()


def _cci(src: pd.Series, length: int) -> pd.Series:
    sma = src.rolling(length).mean()
    dev = src.rolling(length).apply(lambda w: np.abs(w - w.mean()).mean(), raw=True)
    return (src - sma) / (0.015 * dev)


def _mfi(close: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    # talib.MFI hardcodes typical price (H+L+C)/3 as the source; the source
    # script calls mfi(close, 14), so this reimplements it with close instead.
    rmf = close * volume
    delta = close.diff()
    pmf = rmf.where(delta > 0, 0.0).rolling(length).sum()
    nmf = rmf.where(delta < 0, 0.0).rolling(length).sum()
    return 100 * pmf / (pmf + nmf)


def _vwma(df: pd.DataFrame, length: int) -> pd.Series:
    return _sma_vwma(df, length)


def _no_crossing(
    check_arr: np.ndarray, ref: float, cur: float, i: int, span: int, direction: int
) -> bool:
    if span < 2:
        return True
    diff = (cur - ref) / span
    line = cur - diff
    for x in range(1, span):
        v = check_arr[i - x]
        if not np.isnan(v):
            if direction > 0 and v > line:
                return False
            if direction < 0 and v < line:
                return False
        line -= diff
    return True


DEFAULT_INDICATORS = (
    "rsi",
    "macd",
    "macdhist",
    "mom",
    "cci",
    "obv",
    "stk",
    "diosc",
    "vwmacd",
    "cmf",
    "mfi",
)


def compute_indicator_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()
    df["rsi"] = ta.RSI(df, timeperiod=14)
    macd = ta.MACD(df, fastperiod=12, slowperiod=26, signalperiod=9)
    df["macd"] = macd["macd"]
    df["macdhist"] = macd["macdhist"]
    df["mom"] = ta.MOM(df, timeperiod=10)
    df["cci"] = _cci(df["close"], 10)
    df["obv"] = ta.OBV(df)
    df["stk"] = ta.STOCHF(df, fastk_period=14, fastd_period=3, fastd_matype=0)["fastd"]
    trur = ta.ATR(df, timeperiod=14)
    di = (df["high"] + df["low"]).diff()
    df["diosc"] = (100 * _rma(di, 14) / trur).ffill()
    vwma12 = _vwma(df, 12)
    vwma26 = _vwma(df, 26)
    df["vwmacd"] = vwma12 - vwma26
    cmfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"])
    df["cmf"] = (cmfm * df["volume"]).rolling(21).sum() / df["volume"].rolling(21).sum()
    df["mfi"] = _mfi(df["close"], df["volume"], 14)
    return df


def compute_pivot_columns(dataframe: pd.DataFrame, left_bars: int, right_bars: int) -> pd.DataFrame:
    df = dataframe.copy()
    df["ph"] = _pivot(df["high"], left_bars, right_bars, "high")
    df["pl"] = _pivot(df["low"], left_bars, right_bars, "low")
    df["newtop"] = _pivot(df["high"], left_bars, 0, "high")
    df["newbot"] = _pivot(df["low"], left_bars, 0, "low")
    return df


def _detect_side(
    i: int,
    pivot_counter: int,
    new_pivot: float,
    ref_price_arr: np.ndarray,
    close: np.ndarray,
    ind: dict,
    indicators: tuple,
    check_cut: bool,
    direction: int,
):
    if np.isnan(new_pivot) or i - pivot_counter < 0:
        return 0, None
    ref_price = ref_price_arr[i - pivot_counter]
    if np.isnan(ref_price) or new_pivot == ref_price:
        return 0, None
    hidden = (new_pivot < ref_price) if direction > 0 else (new_pivot > ref_price)
    if not _no_crossing(close, ref_price, new_pivot, i, pivot_counter, direction):
        return 0, None

    count = 0
    for col in indicators:
        ref, cur = ind[col][i - pivot_counter], ind[col][i]
        if np.isnan(ref) or np.isnan(cur):
            continue
        if direction > 0:
            triggered = (ref < cur) if hidden else (ref > cur)
        else:
            triggered = (ref > cur) if hidden else (ref < cur)
        if triggered and (
            not check_cut or _no_crossing(ind[col], ref, cur, i, pivot_counter, direction)
        ):
            count += 1
    return count, hidden


def run_divergence_loop(
    ph: np.ndarray,
    pl: np.ndarray,
    ph0: np.ndarray,
    pl0: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    ind: dict,
    indicators: tuple,
    left_bars: int,
    check_cut: bool,
):
    n = len(ph)
    bear = np.zeros(n, dtype=int)
    bear_hidden = np.zeros(n, dtype=int)
    bull = np.zeros(n, dtype=int)
    bull_hidden = np.zeros(n, dtype=int)

    topc, botc = left_bars, left_bars
    for i in range(n):
        topc = left_bars if not np.isnan(ph[i]) else topc + 1
        botc = left_bars if not np.isnan(pl[i]) else botc + 1

        count, hidden = _detect_side(i, topc, ph0[i], high, close, ind, indicators, check_cut, 1)
        if hidden is not None:
            if hidden:
                bear_hidden[i] = count
            else:
                bear[i] = count

        count, hidden = _detect_side(i, botc, pl0[i], low, close, ind, indicators, check_cut, -1)
        if hidden is not None:
            if hidden:
                bull_hidden[i] = count
            else:
                bull[i] = count

    return bear, bear_hidden, bull, bull_hidden


def populate_divergences(
    dataframe: pd.DataFrame,
    left_bars: int = 5,
    right_bars: int = 5,
    check_cut: bool = True,
    indicators: tuple = DEFAULT_INDICATORS,
) -> pd.DataFrame:
    df = compute_indicator_columns(dataframe)
    df = compute_pivot_columns(df, left_bars, right_bars)

    high, low, close = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    ph, pl = df["ph"].to_numpy(), df["pl"].to_numpy()
    ph0, pl0 = df["newtop"].to_numpy(), df["newbot"].to_numpy()
    ind = {c: df[c].to_numpy() for c in indicators}

    bear, bear_hidden, bull, bull_hidden = run_divergence_loop(
        ph, pl, ph0, pl0, high, low, close, ind, indicators, left_bars, check_cut
    )

    df["bearish_divergence"] = bear
    df["bearish_hidden_divergence"] = bear_hidden
    df["bullish_divergence"] = bull
    df["bullish_hidden_divergence"] = bull_hidden
    return df
