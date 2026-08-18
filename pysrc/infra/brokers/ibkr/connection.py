from collections.abc import Iterator
from contextlib import contextmanager

from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from pysrc.core.errors import IBKRConnectionError
from pysrc.ops.mm_logkit import configure_logger, get_logger
from pysrc.pipeline.pipeline_config import get_config

# replace the strict import with a tolerant alias
try:
    from ib_insync import IB as IBKR  # many envs expose IB, not IBKR
except Exception:

    class IBKR:  # fallback stub for import-only smoke test
        pass


configure_logger(
    "marketmind",
    level="DEBUG",
    handlers=[{"type": "stream", "target": "stderr", "level": "DEBUG", "kind": "kv"}],
)
log = get_logger(__name__)


class IBKRConfig(BaseModel):
    host: str
    port: int
    client_id: int


def get_ibkr_config():
    config = get_config()
    return IBKRConfig(
        host=config.real_time_market_data.interactive_brokers.host,
        port=config.real_time_market_data.interactive_brokers.port,
        client_id=config.real_time_market_data.interactive_brokers.client_id,
    )


@contextmanager
def ibkr_connection() -> Iterator[IBKR]:
    ibkr_cfg = get_ibkr_config()
    ibkr = IBKR()
    try:

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(min=1, max=10),
            retry=retry_if_exception_type(ConnectionError),
            before_sleep=lambda retry_state: log.info(
                "Retrying connection", attempt=retry_state.attempt_number
            ),
        )
        def _connect():
            ibkr.connect(ibkr_cfg.host, ibkr_cfg.port, ibkr_cfg.client_id)

        _connect()
        yield ibkr
    except Exception as e:
        log.error("Failed to connect to IBKR after retries", error=str(e))
        raise IBKRConnectionError("Failed to connect to IBKR") from e
    finally:
        ibkr.disconnect()
