"""Modality-agnostic input channel contracts for the local meta-router.

Technical indicators are the first channel. Future channels (fundamentals,
macro, news, crypto, microstructure) register through the same contract
without meta-router code changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal

Modality = Literal[
    "technical",
    "market_state",
    "fundamental",
    "macro",
    "news",
    "sentiment",
    "options",
    "microstructure",
    "on_chain",
    "cross_asset",
]

TECHNICAL_INDICATOR_CHANNEL_ID: Final[str] = "technical_indicators_v1"
MACRO_STATE_CHANNEL_ID: Final[str] = "macro_state_v1"


@dataclass(frozen=True, slots=True)
class ChannelAvailability:
    """Point-in-time availability mask metadata for a channel."""

    channel_id: str
    valid_time_column: str = "date"
    knowledge_time_column: str = "date"
    interval_column: str = "interval"
    instrument_column: str = "instrument"


@dataclass(frozen=True, slots=True)
class FeatureChannel:
    """Declarative feature channel consumed by model and meta-router pipelines."""

    channel_id: str
    modality: Modality
    product_path_key: str
    feature_policy: str
    interval: str = "1d"
    normalization_policy: str = "zscore_train_fold"
    availability: ChannelAvailability | None = None
    schema_version: str = "feature_channel.v1"


@dataclass(frozen=True, slots=True)
class ChannelManifest:
    """Resolved channel set for an experiment run."""

    channels: tuple[FeatureChannel, ...] = field(default_factory=tuple)
    primary_channel_id: str = TECHNICAL_INDICATOR_CHANNEL_ID

    def channel_ids(self) -> tuple[str, ...]:
        return tuple(c.channel_id for c in self.channels)


def default_technical_channel(*, processed_data_root: str = "data/processed") -> FeatureChannel:
    return FeatureChannel(
        channel_id=TECHNICAL_INDICATOR_CHANNEL_ID,
        modality="technical",
        product_path_key="full_indicator_feature_panel",
        feature_policy="full_indicator_universe_v1",
        interval="1d",
        availability=ChannelAvailability(channel_id=TECHNICAL_INDICATOR_CHANNEL_ID),
    )


def technical_channel_manifest() -> ChannelManifest:
    return ChannelManifest(
        channels=(default_technical_channel(),),
        primary_channel_id=TECHNICAL_INDICATOR_CHANNEL_ID,
    )


def default_macro_state_channel(*, processed_data_root: str = "data/processed") -> FeatureChannel:
    return FeatureChannel(
        channel_id=MACRO_STATE_CHANNEL_ID,
        modality="macro",
        product_path_key="macro_state_panel",
        feature_policy="macro_ensemble_state_v1",
        interval="1d",
        availability=ChannelAvailability(channel_id=MACRO_STATE_CHANNEL_ID),
    )


def channel_manifest_with_optional_macro(*, include_macro: bool = False) -> ChannelManifest:
    channels: list[FeatureChannel] = [default_technical_channel()]
    if include_macro:
        channels.append(default_macro_state_channel())
    return ChannelManifest(
        channels=tuple(channels),
        primary_channel_id=TECHNICAL_INDICATOR_CHANNEL_ID,
    )


__all__ = [
    "ChannelAvailability",
    "ChannelManifest",
    "FeatureChannel",
    "MACRO_STATE_CHANNEL_ID",
    "TECHNICAL_INDICATOR_CHANNEL_ID",
    "channel_manifest_with_optional_macro",
    "default_macro_state_channel",
    "default_technical_channel",
    "technical_channel_manifest",
]
