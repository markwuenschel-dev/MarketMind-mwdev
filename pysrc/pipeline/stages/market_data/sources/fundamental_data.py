# Relocated from pysrc.data.fundamental_data

from pysrc.domain.interfaces import AbstractAPIDataManager
from pysrc.pipeline.pipeline_config import get_runtime_config
from pysrc.pipeline.stages.market_data.sources.runtime import APIDataSource


class FundamentalDataManager(AbstractAPIDataManager):
    @property
    def _config_key(self) -> str:
        return "fundamental_data_sources"

    def _build_params(self, query: str) -> dict:
        return {
            "identifier": query,
            "api_key": get_runtime_config().security.credentials.bloomberg_api_key,
        }


@FundamentalDataManager.register("Bloomberg")
class BloombergSource(APIDataSource):
    def __init__(self, config):
        super().__init__(config)
        self.api_key = get_runtime_config().security.credentials.bloomberg_api_key
