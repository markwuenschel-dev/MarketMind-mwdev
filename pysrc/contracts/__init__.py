"""Typed contracts shared across MarketMind product domains.

This package contains data-only boundaries.  It must not import pipeline,
model, strategy, portfolio, meta, backtesting, or artifact implementations.
"""

from pysrc.contracts.candidate_spec import CandidateSpec
from pysrc.contracts.feature_channel import ChannelAvailability, ChannelManifest, FeatureChannel
from pysrc.contracts.meta_router import MetaRouterConfig
from pysrc.contracts.product_artifacts import (
    PredictionValue,
    StandardizedPredictionArtifact,
    StandardizedTradeIntentArtifact,
)
from pysrc.contracts.trade_intent import TradeDirection, TradeIntent

__all__ = [
    "ChannelAvailability",
    "CandidateSpec",
    "ChannelManifest",
    "FeatureChannel",
    "MetaRouterConfig",
    "TradeDirection",
    "TradeIntent",
    "PredictionValue",
    "StandardizedPredictionArtifact",
    "StandardizedTradeIntentArtifact",
]
