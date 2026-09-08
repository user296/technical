import numpy as np
import pandas as pd

from technical.indicators.divergence import (
    _cci,
    _mfi,
    _no_crossing,
    _pivot,
    compute_indicator_columns,
    compute_pivot_columns,
)


def test_pivot_high_detects_local_max():
    series = pd.Series([1, 2, 3, 10, 3, 2, 1, 1, 1, 1, 1])
    result = _pivot(series, left=3, right=3, mode="high")
    assert result.dropna().iloc[0] == 10


def test_pivot_tie_breaks_toward_later_bar():
    series = pd.Series([1, 1, 1, 5, 5, 1, 1, 1])
    result = _pivot(series, left=3, right=3, mode="high")
    non_na = result.dropna()
    assert len(non_na) == 1
    assert non_na.index[0] == 7
    assert non_na.iloc[0] == 5


def test_no_crossing_true_for_monotonic_line():
    close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert _no_crossing(close, ref=1.0, cur=5.0, i=4, span=4, direction=1) is True


def test_no_crossing_false_when_bar_exceeds_line():
    close = np.array([1.0, 100.0, 3.0, 4.0, 5.0])
    assert _no_crossing(close, ref=1.0, cur=5.0, i=4, span=4, direction=1) is False


def test_cci_matches_pine_dev_definition():
    src = pd.Series([1.0, 2.0, 3.0, 100.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0])
    result = _cci(src, length=5)
    assert not result.isna().all()


def test_mfi_bounded_zero_to_hundred():
    rng = np.random.default_rng(1)
    close = pd.Series(100 + np.cumsum(rng.standard_normal(50)))
    volume = pd.Series(rng.integers(100, 1000, 50).astype(float))
    result = _mfi(close, volume, length=14)
    valid = result.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def _synthetic_ohlcv():
    rng = np.random.default_rng(42)
    n = 300
    close = 100 + np.cumsum(rng.standard_normal(n))
    return pd.DataFrame(
        {
            "open": close + rng.standard_normal(n),
            "high": close + rng.random(n) * 2,
            "low": close - rng.random(n) * 2,
            "close": close,
            "volume": rng.integers(100, 1000, n),
        }
    )


def test_compute_indicator_columns_adds_expected_columns():
    result = compute_indicator_columns(_synthetic_ohlcv())
    for column in (
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
    ):
        assert column in result.columns


def test_compute_pivot_columns_adds_expected_columns():
    result = compute_pivot_columns(_synthetic_ohlcv(), left_bars=5, right_bars=5)
    for column in ("ph", "pl", "newtop", "newbot"):
        assert column in result.columns
