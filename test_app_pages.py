"""Smoke tests for every Streamlit page (skipped automatically if streamlit is not installed)."""
import pathlib, pytest
pytest.importorskip("streamlit"); pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest
from src.data_generation.simulator import generate
from src.engines.pandas_engine import aggregate

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAGES = ["app.py"] + [f"pages/{n}" for n in ["1_Data_Simulation.py", "2_KPI_Dashboard.py", "3_Big_Data_Benchmark.py",
                                              "4_Budget_Recommendation.py", "5_Reports.py"]]


def _state():
    df = generate(60_000, seed=5)
    return dict(df=df, cube=aggregate(df), simulated=True, data_label="test", rec=None)


@pytest.mark.parametrize("page", PAGES)
def test_page_runs_with_data(page):
    at = AppTest.from_file(str(ROOT / page), default_timeout=120)
    for k, v in _state().items():
        at.session_state[k] = v
    at.run()
    assert not at.exception, at.exception


@pytest.mark.parametrize("page", PAGES)
def test_page_empty_state(page):
    at = AppTest.from_file(str(ROOT / page), default_timeout=60).run()
    assert not at.exception, at.exception
