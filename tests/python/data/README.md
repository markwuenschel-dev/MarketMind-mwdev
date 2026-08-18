
# Robust Fixtures: Torture Pack

Each file is crafted to expose brittle assumptions across parsing, schema handling, time logic,
hashing/caching behavior, streaming, and configuration.

## Files

- `zero_byte.csv` — Empty file: assert graceful "empty frame" behavior.
- `header_only_prices.csv` — Headers without rows.
- `bom_weird.csv` — UTF-8 BOM + odd headers/spaces; numeric strings in volume.
- `unsorted_dupe_timestamps.csv` — Unsorted, duplicate (ts,symbol), mixed tz vs. naive, negative volume.
- `irregular_freq.csv` — Missing dates and intra-minute sample amid daily cadence.
- `stale_prices.csv` — Flat OHLC and mostly zero volumes (stale-feed detection).
- `corp_actions.csv` — Split 2.0, reverse 0.5, special dividend on weekend ex-date.
- `symbol_rename.csv` + `symbol_map.csv` — Identity continuity test FB→META.
- `schema_drift_v3.csv` — Renamed columns, extra fields, mixed-format volumes.
- `types_mixed.csv` — Mixed numeric representations, NaN/inf/underscores/commas.
- `csv_with_blank_lines.csv` — Blank rows and trailing separators.
- `semicolon_delimited.csv` — Non-comma delimiter and delimiter inside symbol values.
- `jsonl_prices.jsonl` — Line-delimited JSON for alternative ingestion path.
- `gzip_prices.csv.gz` — Gzipped CSV.
- `encoding_latin1.csv` — Latin-1 encoded text with accents.
- `unsupported.avro` — Intentional unsupported format (should raise).
- `malformed_csv.csv` — Broken quotes and embedded newline (robust CSV reader?).
- `overflow_values.csv` — Absurd outliers and invalid negative lows.
- `micro_batches_stream/` — Chunked stream: burst, single-row, big burst, empty, and a poison chunk.
- `crossfile_prices.csv` + `crossfile_metadata.csv` — Mismatched symbol universes.
- `timezones_mixed.csv` — EST/EDT/naive/UTC mixing across DST boundary.
- `config_unknown_keys.yaml` — Unknown key (`leveel`) to test schema permissiveness.
- `config_env_overrides.yaml` — Env interpolation and type coercion.
- `config_bad_types.yaml` — Wrong types to test validation/coercion.

## Suggested Assertions (examples)
- Dedup policy: on (ts,symbol) duplicates, keep latest by `source` or deterministic rule.
- DST & tz normalization: output index tz-aware UTC or canonical trading tz; sorted uniqueness.
- Corporate actions: either adjust prices or emit explicit flags; never silently double-adjust.
- Mixed numerics: parse `'1_000'`, `'1,234'`, `'2e5'`; decide policy for `'NaN'`, `'inf'`, empty.
- Delimiters/encodings: auto-detect or require explicit, but fail with precise error class/message.
- Streaming: partial batches ok; poison chunk triggers recovery and metric/manifest update.
- Cross-file consistency: symbols in metadata must be a superset of prices (or vice versa), else raise.
- Config: unknown keys rejected (if strict) or logged; env vars coerced to correct types; bad types fail-fast.

