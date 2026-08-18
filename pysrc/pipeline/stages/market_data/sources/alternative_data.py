# Relocated from pysrc.data.alternative_data

from pysrc.domain.interfaces import AbstractAPIDataManager
from pysrc.pipeline.pipeline_config import get_runtime_config
from pysrc.pipeline.stages.market_data.sources.runtime import APIDataSource


class AlternativeDataManager(AbstractAPIDataManager):
    @property
    def _config_key(self) -> str:
        return "alternative_data_sources"

    def _build_params(self, query: str) -> dict:
        return {"q": query, "format": "json"}


@AlternativeDataManager.register("Twitter")
class TwitterSource(APIDataSource):
    def __init__(self, config):
        super().__init__(config)
        self.bearer_token = get_runtime_config().security.credentials.twitter_bearer_token


@AlternativeDataManager.register("ESG")
class ESGSource(APIDataSource):
    def __init__(self, config):
        super().__init__(config)


@AlternativeDataManager.register("Weather")
class WeatherSource(APIDataSource):
    def __init__(self, config):
        super().__init__(config)
        self.api_key = get_runtime_config().security.credentials.weather_api_key
