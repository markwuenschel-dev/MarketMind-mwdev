# tests/python/unit/test_loader_factory_matrix.py
import pytest

from pysrc.pipeline.stages.market_data.sources.data_loader import (
    CSVLoader,
    FREDLoader,
    InfluxDBLoader,
    TwitterLoader,
    build_loader,
)


@pytest.mark.parametrize(
    ("name", "cls"),
    [
        ("csv", CSVLoader),
        ("fred", FREDLoader),
        ("twitter", TwitterLoader),
        ("influxdb", InfluxDBLoader),
        # ...
    ],
)
def test_build_loader_matrix(name, cls):
    inst = build_loader(name)
    assert isinstance(inst, cls)


def test_build_loader_unknown():
    with pytest.raises(ValueError):
        build_loader("not-a-thing")
