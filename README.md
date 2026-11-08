# Road Accident Intelligence — Premium Interactive Dashboard

A responsive dark-themed Streamlit dashboard built from the supplied road-accident project outputs. The data is bundled with the app and loads automatically; visitors do not need to upload CSV or ZIP files.

## Included experience

- Premium dark interface with responsive layout, custom typography, refined KPI cards, and styled navigation.
- Global state, year, severity, and city filters for dashboard analysis.
- Overview charts for severity, state, road type, weather, time period, season, and day of week.
- State and city hotspot rankings based on the supplied records.
- Embedded interactive Folium geographical map (`data/geographical_hotspot_map.html`).
- Decision Tree severity prediction form trained from the bundled dataset.
- DBSCAN and hierarchical cluster-label summaries and clustering comparison.
- Apriori vs FP-Growth comparison and searchable-size association-rule display.
- Dataset explorer with text search and CSV export.

## Project structure

```text
road_accident_website/
├── app.py
├── requirements.txt
├── README.md
└── data/
    ├── cleaned_data.csv
    ├── geographical_hotspot_map.html
    ├── dbscan_labels.csv
    ├── hier_labels.csv
    ├── clustering_comparison.csv
    ├── algorithm_comparison.csv
    └── association_rules.csv
```

## Run locally (Windows PowerShell)

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

If `py` is unavailable, use `python -m pip` and `python -m streamlit`.

## Deploy publicly

1. Extract this folder and create a GitHub repository.
2. Upload the complete contents of this folder, including the `data/` directory.
3. Open [Streamlit Community Cloud](https://share.streamlit.io/) and connect the repository.
4. Set the app entry point to `app.py` and deploy.
5. Share the generated public URL.

Keep the `data/` directory in the repository: it contains the project data and precomputed analysis outputs required by the app. Anyone who can access the public app may be able to access information represented by these bundled files, so publish only data you are allowed to share.

## Important interpretation notes

- Counts are counts of rows in the supplied dataset; they are not population-adjusted accident-risk estimates.
- The Folium map is the supplied pre-generated HTML map. Its map tiles and external JavaScript/CSS libraries need an internet connection. The map itself is not recalculated by the dashboard filters.
- DBSCAN and hierarchical labels are displayed from their saved output files. The hierarchical labels file contains a subset of records.
- The Decision Tree is retrained in the app using the bundled data and a held-out test split. Its displayed score is a model evaluation result, not a guarantee of prediction quality. Predictions are for academic demonstration only and must not be used for emergency or safety-critical decisions.
- Association-rule support, confidence, and lift describe co-occurrence, not causation.
