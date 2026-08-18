# tests/python/property/ml_strategies.py
# Hypothesis strategies for ensemble weight vectors (extensible for future DF/matrix strategies).

from __future__ import annotations

from hypothesis import strategies as st
from hypothesis.strategies import composite


@composite
def weight_vectors_for_ensemble(draw, n_models: int | None = None):
    if n_models is None:
        n_models = draw(st.integers(min_value=2, max_value=10))

    edge_case = draw(
        st.sampled_from(["uniform", "zero_sum", "one_hot", "negative", "tiny", "huge", "random"])
    )

    if edge_case == "uniform":
        return [1.0 / n_models] * n_models

    if edge_case == "zero_sum":
        weights = draw(
            st.lists(
                st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
                min_size=n_models,
                max_size=n_models,
            )
        )
        weights[-1] = -sum(weights[:-1])
        return weights

    if edge_case == "one_hot":
        idx = draw(st.integers(min_value=0, max_value=n_models - 1))
        w = [0.0] * n_models
        w[idx] = 1.0
        return w

    if edge_case == "negative":
        return draw(
            st.lists(
                st.floats(min_value=-1, max_value=1, allow_nan=False, allow_infinity=False),
                min_size=n_models,
                max_size=n_models,
            )
        )

    if edge_case == "tiny":
        return draw(
            st.lists(
                st.floats(min_value=1e-15, max_value=1e-10, allow_nan=False, allow_infinity=False),
                min_size=n_models,
                max_size=n_models,
            )
        )

    if edge_case == "huge":
        return draw(
            st.lists(
                st.floats(min_value=1e10, max_value=1e15, allow_nan=False, allow_infinity=False),
                min_size=n_models,
                max_size=n_models,
            )
        )

    return draw(
        st.lists(
            st.floats(min_value=0, max_value=10, allow_nan=False, allow_infinity=False),
            min_size=n_models,
            max_size=n_models,
        )
    )
