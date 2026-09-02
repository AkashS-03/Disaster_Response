# Stage 01 - Data Engineer Report

## 1. Data Sources
The pipeline ingests four primary raw data streams:
- `river_rainfall_data.csv`: Hourly river and rain gauge telemetry.
- `emergency_calls.csv`: 911 dispatch volumes.
- `infrastructure_records.csv`: Road and bridge closure logs.
- `historical_flood_logs.csv`: Historical incidents tracking river/rain thresholds against flood occurrences.

*(Note: Since real data was unavailable, a stochastic simulation engine in `data_ingestion.py` synthesizes these datasets with realistic causal relationships—e.g., heavy rain leads to rising rivers, which leads to closures and emergency calls).*

## 2. Dataset Structure
The initial raw datasets are disjointed CSV files. The pipeline cleans, normalizes, and merges them on a unified `(zone_id, timestamp)` index to produce a single `master_dataset.csv`.

## 3. Cleaning Process
- **Zone IDs**: Standardized to title case (e.g., `Zone_A`) to prevent join mismatches.
- **Timestamps**: Parsed into standard ISO 8601 UTC datetimes across all datasets.
- **Value Bounds**: Handled impossible values (e.g., negative rainfall or negative closures are capped at 0).

## 4. Missing-Value Strategy
- **River Gauge**: Forward-filled (`ffill`) since physical gauge levels generally persist unless actively drained.
- **Rainfall**: Missing hours assume `0.0` mm of rain.
- **Emergency Calls / Closures**: Missing data points are replaced with `0`, assuming no active logging indicates no active events.

## 5. Duplicate Handling
Duplicate row enforcement is applied via `drop_duplicates(subset=['zone_id', 'timestamp'])` to ensure temporal uniqueness before joining.

## 6. Timestamp Handling
All timestamps are loaded into `pandas.Datetime` objects, forced to UTC, and used as the primary merge key. 

## 7. Data Merging Strategy
A left-join strategy is utilized on `zone_id` and `timestamp`, originating from the most continuous time-series (river/rainfall gauges), appending emergency calls and infrastructure records. Historical probabilities are joined via a rounded bucket of `river_level`.

## 8. Feature Engineering Performed
- `total_infrastructure_closures`: Sum of roads and bridges.
- `river_level_trend`: 1-hour ($\Delta$) change in river depth.
- `historical_flood_probability`: Computed probability of flooding given the current river level bucket.

## 9. 72-Hour Rolling Window Approach
As explicitly requested by the project requirements, critical 72-hour rolling windows are generated utilizing Pandas time-based rolling aggregations (`.rolling("72h")`) on the `timestamp` index.
- `river_level_rolling_72h_avg`: Average river height over the last 3 days.
- `rainfall_rolling_72h_sum`: Cumulative rain accumulation over the last 3 days.

## 10. Validation Checks
Automated schema validation (`data_validation.py`) enforces:
1. Presence of all mandatory columns.
2. 0% missing values in critical columns post-cleaning.
3. 0 duplicate index collisions.
4. Non-negative constraints on absolute metrics.
5. Strict `LOW`, `MODERATE`, `SEVERE` categorization.

## 11. Final Dataset Statistics
- **Total Zones**: 4 (`Zone_A`, `Zone_B`, `Zone_C`, `Zone_D`)
- **Total Observations**: 1,344 rows (14 days of hourly data per zone)
- **Target Risk Distribution**: (See terminal output during build process)

## 12. Known Limitations
- Synthetically generated target labels follow static physical thresholds. The ML Engineer should be aware that the decision boundary is rule-based in this mock dataset.
- Rolling window features contain trailing zeros/defaults at the absolute beginning of the time series (first 72 hours).

## 13. Handoff Notes for the EDA Engineer
The clean master dataset is available at `data/processed/master_dataset.csv`. 
- **EDA Engineer**: You can immediately load this CSV and begin analyzing the distributions, checking how the `72h` rolling features correlate with the `risk_label`, and producing visualizations.
- No further timezone parsing or missing value imputation is required on your end.
