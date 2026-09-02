# Stage 01 - Data Dictionary

This document details the columns available in `data/processed/master_dataset.csv`.

| Column Name | Description | Type | Source | Allowed/Expected Range | Null Handling |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `zone_id` | City zone identifier | string | All sources | `Zone_A` to `Zone_D` | None |
| `timestamp` | Observation timestamp (UTC) | datetime | All sources | ISO 8601 string | None |
| `river_level` | River gauge level | float | River gauge | `1.0 - 10.0` (meters) | Forward fill / linear interpolation |
| `rainfall` | Hourly rainfall measurement | float | Rain gauge | `0.0 - 100.0` (mm) | Missing replaced with `0.0` |
| `emergency_call_volume` | Number of emergency 911 calls in the hour | integer | 911 logs | `0 - 100+` | Missing replaced with `0` |
| `road_closures` | Number of closed roads | integer | Infrastructure | `0 - 20` | Missing replaced with `0` |
| `bridge_closures` | Number of closed bridges | integer | Infrastructure | `0 - 5` | Missing replaced with `0` |
| `historical_flood_probability` | Probability of historical flood given current river bucket | float | Derived (Historical) | `0.0 - 1.0` | Default `0.0` |
| `river_level_rolling_72h_avg` | 72-hour rolling average of river level | float | Derived (Gauges) | `1.0 - 10.0` | Default `0.0` for early window |
| `rainfall_rolling_72h_sum` | 72-hour rolling cumulative sum of rainfall | float | Derived (Gauges) | `0.0 - 500.0` | Default `0.0` |
| `emergency_calls_24h_sum` | 24-hour rolling sum of emergency calls | float | Derived (911 logs) | `0.0 - 1000.0` | Default `0.0` |
| `total_infrastructure_closures` | `road_closures` + `bridge_closures` | integer | Derived (Infra) | `0 - 25` | N/A |
| `river_level_trend` | 1-hour change in river level | float | Derived (Gauges) | `-5.0 to 5.0` | Default `0.0` for first hour |
| `risk_label` | Target triage classification for ML system | string | Synthetic Logic | `LOW`, `MODERATE`, `SEVERE` | None |

### Synthetic Label Methodology
The `risk_label` is synthetically generated using physical business rules based on the 72-hour data windows:
- **SEVERE**: Extreme river levels (> 4.5m) OR extreme 72-hour accumulated rainfall (> 150mm) paired with elevated river levels (> 3.5m).
- **MODERATE**: Elevated river levels (> 3.0m) OR heavy 72-hour accumulated rainfall (> 80mm) OR high emergency call volumes (> 100 calls in 24h).
- **LOW**: Normal baseline conditions.
