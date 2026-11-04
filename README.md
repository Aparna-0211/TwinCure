# TwinCure — Optimizing Treatments from Similar Patient Journeys

**TwinCure** is an educational AI/ML software prototype that explores how similar patient histories can be analyzed to generate traceable hybrid journey examples.

Rather than focusing only on disease prediction, TwinCure demonstrates a modular workflow involving text similarity, treatment-path segment analysis, compatibility validation, hybrid journey generation, and transparent evidence-support scoring.

> **Important:** TwinCure uses entirely synthetic educational data. It is not a medical device, clinically validated system, or tool for diagnosis or treatment decisions.

## Project Overview

Patient journeys contain sequential information, including symptoms, ordered steps, and recorded outcomes. TwinCure explores how these journeys can be compared and how selected segments from different synthetic journeys can be combined under explicitly defined compatibility and sequence rules.

The project emphasizes transparency and traceability: every generated hybrid retains information about its original source journeys, segments, steps, and ordering decisions.

## Key Features

- **Patient Similarity:** Uses TF-IDF and cosine similarity to identify textually similar synthetic patient histories.
- **Treatment Path Analysis:** Generates contiguous journey segments and classifies them using constructed outcome values.
- **Compatibility Validation:** Checks candidate segments against explicit synthetic compatibility and sequence rules.
- **Hybrid Journey Generation:** Creates separate hybrid examples from validated combinations while preserving source provenance.
- **Evidence-Support Scoring:** Calculates a transparent score using similarity, outcome support, and journey-level agreement.
- **Interactive Dashboard:** Provides six Streamlit sections to explore the complete workflow.

## Workflow

1. **Data Storage:** Synthetic patient histories and ordered journey steps are stored in SQLite.
2. **Preprocessing:** Symptom descriptions are standardized for text analysis.
3. **Similarity Analysis:** TF-IDF and cosine similarity identify matching journeys.
4. **Treatment Analysis:** Contiguous segments are generated and classified from synthetic outcomes.
5. **Compatibility Analysis:** Explicit rules determine whether candidate segments can be combined and in which order.
6. **Hybrid Generation:** Separate hybrid examples are generated only from validated combinations.
7. **Evidence Scoring:** Each hybrid receives an evidence-support summary based on its contributing synthetic journeys.
8. **Dashboard Visualization:** Results are presented through an interactive Streamlit application.

## Dataset

The initial demonstration uses a small, constructed dataset in the **Heart Disease** domain.

| Dataset detail | Count |
|---|---:|
| Synthetic patient journeys | 8 |
| Ordered journey steps | 24 |
| Sequence rules | 2 |
| Compatibility rules | 3 |

The dataset includes synthetic patient IDs, age bands, symptom descriptions, journey outcomes, step categories, descriptions, and step-level outcomes.

All records carry the label `SYNTHETIC_EDUCATIONAL_ONLY`.

The dataset is fabricated for software demonstration and does not contain real patient records or clinical evidence.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application and analysis |
| Pandas | Data handling and processing |
| NumPy | Numerical operations |
| Scikit-learn | TF-IDF and cosine similarity |
| SQLite | Local database |
| Streamlit | Interactive dashboard |
| Matplotlib | Included in requirements; not currently used for dashboard charts |
| pytest | Automated testing |

## Project Structure

```text
TwinCure/
├── config/
│   └── settings.toml
├── data/
│   ├── synthetic_patients.csv
│   ├── synthetic_steps.csv
│   └── synthetic_rules.json
├── database/
│   ├── schema.sql
│   └── .gitkeep
├── src/
│   └── twincure/
│       ├── app.py
│       ├── db.py
│       ├── preprocessing.py
│       ├── similarity.py
│       ├── treatment_analysis.py
│       ├── segment_selection.py
│       ├── hybrid_generation.py
│       ├── evidence_scoring.py
│       └── dashboard/
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```

The local SQLite database is generated during setup and is intentionally excluded from Git.

## Installation and Setup

### Prerequisites

- Python 3.10 or newer
- Git
- pip

### 1. Clone the repository

```bash
git clone https://github.com/Aparna-0211/TwinCure.git
cd TwinCure
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Initialize the database

Run this command from the project root:

```bash
python -c "from twincure.db import initialize_database; initialize_database('database/twincure.db', 'database/schema.sql', 'data/synthetic_patients.csv', 'data/synthetic_steps.csv', 'data/synthetic_rules.json')"
```

If your environment does not automatically include the `src` directory in Python's import path, use:

**Windows PowerShell:**

```powershell
$env:PYTHONPATH = "src"
```

**Linux / macOS:**

```bash
export PYTHONPATH=src
```

Then run the initialization command.

### 5. Launch the dashboard

```bash
python -m streamlit run src/twincure/app.py
```

The Streamlit application will open in your browser.

## Evidence-Support Scoring

TwinCure calculates a synthetic evidence-support score using three components:

- **S — Similarity Support:** Mean cosine similarity across unique contributing source journeys.
- **O — Outcome Support:** Outcome support calculated from unique source steps in selected positive segments.
- **A — Agreement Support:** Positive journey votes divided by the total positive and negative journey votes for the hybrid's contributing journeys.

The scoring formula is:

```text
Score = 100 × (0.4118S + 0.3529O + 0.2353A)
```

Each component ranges from 0 to 1 when defined.

The calculation uses unique source journeys and step identifiers to avoid duplicate contributions. If a required component cannot be calculated because evidence is insufficient, the score is withheld rather than replacing missing values with zero.

**The score represents support within the constructed synthetic examples only. It is not a probability of treatment success or a measure of clinical effectiveness.**

## Testing

TwinCure includes automated tests for its core modules and Streamlit dashboard.

Run the complete test suite from the project root:

```bash
pytest -q
```

The current implementation has **99 passing tests**, covering:

- Synthetic database initialization and data integrity.
- Text preprocessing and similarity matching.
- Treatment segment generation and classification.
- Compatibility and sequence validation.
- Hybrid generation and source provenance.
- Evidence-support calculations and missing-evidence handling.
- Dashboard pages, navigation, and error states.

## Current Limitations

- The starter dataset is small and entirely synthetic.
- The similarity threshold (`0.20`) is a configurable demonstration value, not an empirically validated cutoff.
- Compatibility and sequence rules are handcrafted educational rules, not clinical protocols.
- TF-IDF measures textual overlap and does not establish medical equivalence.
- Synthetic outcomes do not establish causality or real-world treatment effectiveness.
- Evidence-support scores are not clinically validated predictions or confidence measures.
- The dashboard currently displays tables and metrics rather than Matplotlib charts.
- SQLite synchronization uses `INSERT OR IGNORE`; changes to existing source records require a controlled database rebuild.

## Future Enhancements

Potential academic improvements include:

- Larger synthetic datasets and additional demonstration scenarios.
- Richer visualizations of journey sequences and source provenance.
- More extensive evaluation of similarity thresholds.
- Improved CSV-to-SQLite synchronization.
- Exploration of Graph Neural Networks (GNNs), Explainable AI (XAI), and healthcare-focused language models as future research directions.

These are proposed enhancements, not features currently implemented.

## Disclaimer

TwinCure is a **synthetic educational software prototype**, not a medical device or clinically validated system.

It is not intended for diagnosis, prescribing, treatment planning, or clinical decision-making. Similarity scores, segment classifications, compatibility decisions, generated hybrid journeys, and evidence-support scores are experimental outputs based solely on constructed examples and rules.

They do not represent clinical evidence, medical advice, or predictions of patient outcomes.

**Do not enter real patient information or personally identifiable information.**
