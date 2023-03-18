import pytest
from batchstream.monitoring.pipeline.steps.river_monitoring_step import RiverMonitoringStep
from river.drift import ADWIN
import random
import numpy as np



# arrange
@pytest.fixture
def data_stream():
    rng = random.Random(12345)
    data_stream = rng.choices([0, 1], k=1000) + rng.choices(range(4, 8), k=1000)
    return data_stream

@pytest.fixture
def river_monitoring_step():
    a = ADWIN()
    monitoring_step = RiverMonitoringStep(step_name='ADWIN', river_detector=a)
    return monitoring_step

@pytest.fixture
def expected():
    return 1023

# act
@pytest.fixture
def actual(data_stream, river_monitoring_step):
    results = []
    for i, val in enumerate(data_stream):
        res = river_monitoring_step.monitor(data_stream[:i+1])
        results.append(list(res[0].values())[0])
    drift_idx = np.where(np.array(results))[0][0]
    return  drift_idx

# assert
def test_river_monitoring_step_adwin(actual, expected):
    assert actual == expected
    