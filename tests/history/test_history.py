import pytest
from river.stream import iter_pandas
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from batchstream.history.base.history_manager import HistoryManager



# arrange
@pytest.fixture
def n_flush_clock():
    return 30

@pytest.fixture
def n_to_stay():
    return 20

@pytest.fixture
def XY():
    return load_iris(return_X_y=True)

@pytest.fixture
def expected_X_y(XY, n_to_stay):
    X, Y = XY
    return pd.DataFrame(X).iloc[-n_to_stay:, :], pd.Series(Y, dtype=np.uint8).iloc[-n_to_stay:]

@pytest.fixture
def history(n_flush_clock, n_to_stay):
    return HistoryManager(n_flush_clock=n_flush_clock, n_to_stay=n_to_stay) 

# act
@pytest.fixture
def actual_X_y(XY, history: HistoryManager):
    X, Y = XY
    for x, y in iter_pandas(pd.DataFrame(X), pd.Series(Y)):
        history.update_history_x(x)
        history.update_history_y(y)
    return history.x_history, history.y_history

# assert
def test_history(actual_X_y, expected_X_y):
    X_actual, y_actual = actual_X_y
    X_expected, y_expected = expected_X_y
    pd.testing.assert_frame_equal(X_actual, X_expected)
    pd.testing.assert_series_equal(y_actual, y_expected)
