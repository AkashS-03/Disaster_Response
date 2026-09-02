# Disaster Response Coordination - Stage 01

## Role: Data Engineer

This repository strictly contains the Data Engineering pipeline for Stage 01 of the Disaster Response ML system. It prepares a reliable `master_dataset.csv` for downstream Exploratory Data Analysis (EDA) and Machine Learning (ML).

### Pipeline Flow
`Raw Telemetry` -> `data_ingestion.py` -> `data_cleaning.py` -> `build_master_dataset.py` -> `data_validation.py` -> `master_dataset.csv`

### Project Structure
```
stage_01_ml/
├── data/
│   ├── raw/                      # Raw synthetic datasets
│   ├── processed/                # Final master dataset output
│   └── data_dictionary.md        # Feature schema documentation
├── data_engineer/
│   ├── data_ingestion.py         # Generates/loads raw datasets
│   ├── data_cleaning.py          # Data sanitation and imputation
│   ├── data_validation.py        # Constraint validation
│   └── build_master_dataset.py   # Core feature engineering pipeline
├── tests/
│   └── test_data_engineer.py     # Pytest unit tests
├── docs/
│   └── data_engineer_report.md   # Detailed methodology report
├── requirements.txt
└── README.md
```
*(Other role folders are intentionally kept empty for future development).*

### Features
- Handles multi-stream ingestion (river gauges, rainfall, 911 calls, infrastructure closures, historical logs).
- Handles missing values, deduplication, timestamp formatting (UTC ISO-8601), and anomaly capping.
- Computes critical **72-hour rolling windows** as requested by the project spec.
- Defines synthetic `risk_label` categorization (`LOW`, `MODERATE`, `SEVERE`).

### How to Run

**1. Install Dependencies**
```bash
pip install -r requirements.txt
```

**2. Run the Data Engineer Pipeline**
This single command triggers ingestion, cleaning, temporal merging, 72-hour rolling feature engineering, validation, and saves the final master dataset.
```bash
python data_engineer/build_master_dataset.py
```

**3. Run Unit Tests**
```bash
pytest tests/
```

### Next Steps / Handoff
The Data Engineer phase is complete. The clean `data/processed/master_dataset.csv` is validated and ready for the **EDA Engineer** to begin exploratory analysis.
