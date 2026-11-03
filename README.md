# TwinCure — Optimizing Treatments from Similar Patient Journeys

TwinCure is a college-level AI/ML software project that demonstrates a modular approach to learning from similar patient journeys. Instead of treating the task as a basic disease-prediction problem, it explores how text similarity, ordered journey segments, explicit synthetic compatibility rules, provenance-preserving composition, and transparent evidence summaries can be combined in one educational workflow.

## Problem statement and motivation

Patient-journey information is sequential: a history contains symptoms, ordered steps, and recorded outcomes. A single prediction label can hide that sequence and the evidence used to justify a proposed path. TwinCure is motivated by the software-engineering challenge of making comparisons and combinations traceable: users should be able to see which synthetic journeys were similar, which source steps form a segment, why two segments were or were not compatible, how order was established, and which unique journeys contributed to a score.

This project does not establish that similar histories are medically equivalent, that a sequence caused an outcome, or that a constructed hybrid would work in practice. It is a software prototype, not a clinical method.

## Project objectives

- Store patient histories and ordered journey steps while preserving their source identifiers and outcomes.
- Find textually similar histories with TF-IDF and cosine similarity using a configurable demonstration threshold.
- Analyze contiguous historical journey segments and label them from the dataset's constructed outcome values.
- Select candidate segments only through explicit synthetic compatibility and sequence rules; report unknown, invalid, unresolved, and ambiguous cases rather than guessing.
- Build separate, traceable hybrid examples from eligible segments, retaining source patient, segment, step, order, and outcome provenance.
- Summarize available synthetic evidence with a transparent, unique-journey-based formula and withhold scores when required evidence is insufficient.
- Present the workflow and comparisons in a six-section Streamlit dashboard.

## Current implementation status

Stages 1–3 (project setup/data, preprocessing/similarity, and treatment-path segment analysis), Stage 4A (positive candidate selection with compatibility and sequence validation), Stage 4B (traceable hybrid journey generation), Stage 5 (synthetic evidence-support scoring), and dashboard Stages 6A–6F (Home, Patient Similarity, Treatment Analysis, Compatibility Analysis, Hybrid Journey, and Evidence Scoring) are implemented and covered by the project test suite.

## Synthetic educational dataset

The starter dataset contains **8 constructed patient journeys** and **24 ordered journey steps** in the Heart Disease demonstration domain. Patient rows use synthetic IDs such as `SYN001`; step rows use synthetic IDs such as `ST001A`. The CSV contains age bands, broad symptom text, journey-level constructed outcomes, step categories, step descriptions, step outcomes, and notes. The JSON file provides explicitly synthetic compatibility and sequencing rules. Records carry the label `SYNTHETIC_EDUCATIONAL_ONLY`. These are fabricated educational examples, not sourced clinical cases or real patient records; do not add real patient data or personally identifying information.

## Full TwinCure workflow

1. Store synthetic patient histories and ordered journey steps in SQLite.
2. Standardize symptom text, then Stage 2 uses TF-IDF and cosine similarity to find journeys above the configurable demonstration threshold.
3. Stage 3 builds bounded contiguous segments from each matching journey and labels them using only the synthetic outcome fields.
4. Stage 4A checks positive segments against explicit compatibility and sequencing rules. Unknown, incompatible, or ambiguous results do not become eligible combinations.
5. Stage 4B creates separate hybrid examples only from Stage 4A-approved combinations and retains step-level source provenance.
6. Stage 5 calculates S (unique-journey similarity), O (unique-step outcome support aggregated by journey), A (hybrid-specific unique-journey agreement), and the weighted synthetic evidence-support score. Stage 6 presents each stage through the Streamlit dashboard.

## Project directory structure

```text
.
├── config/settings.toml
├── data/
│   ├── synthetic_patients.csv
│   ├── synthetic_steps.csv
│   └── synthetic_rules.json
├── database/
│   ├── .gitkeep
│   └── schema.sql
├── src/twincure/
│   ├── app.py
│   ├── db.py, preprocessing.py, similarity.py
│   ├── treatment_analysis.py, segment_selection.py
│   ├── hybrid_generation.py, evidence_scoring.py
│   └── dashboard/ (six Streamlit page modules)
├── tests/ (core and dashboard tests)
├── .gitignore
├── README.md
└── requirements.txt
```

The local runtime database `database/twincure.db` is created during setup and is intentionally ignored by Git; the tracked `.gitkeep` keeps the directory visible in a fresh clone.

## Project architecture

- `data/`: synthetic patient and ordered-step CSV files plus explicit educational sequence/compatibility rules.
- `config/settings.toml`: demonstration threshold and future segment settings.
- `database/schema.sql`: relational schema and integrity constraints.
- `src/twincure/db.py`: SQLite initialization and read helpers.
- `src/twincure/preprocessing.py`: Pandas loading and symptom-text standardization.
- `src/twincure/similarity.py`: TF-IDF/cosine matching and threshold filtering.
- `src/twincure/treatment_analysis.py`: ordered within-journey segment generation and outcome labels.
- `src/twincure/segment_selection.py`: positive segment selection plus explicit compatibility/sequence validation.
- `src/twincure/hybrid_generation.py`: hybrid candidates from Stage 4A-eligible combinations, with step-level provenance.
- `src/twincure/evidence_scoring.py`: explainable evidence-support components and weighted score, with insufficient-evidence guards.
- `src/twincure/app.py`: Streamlit entry point and six-section navigation.
- `src/twincure/dashboard/home.py`: Home page, live SQLite statistics, and database health display.
- `src/twincure/dashboard/compatibility_analysis.py`: Stage 2→3→4A compatibility UI with provenance and rule-decision tables.
- `src/twincure/dashboard/hybrid_journey.py`: Stage 2→3→4A→4B hybrid display with validated order and step-level provenance.
- `src/twincure/dashboard/evidence_scoring.py`: Stage 2→3→4A→4B→5 evidence-support view with component details and unique-journey votes.
- `tests/`: core module tests plus Streamlit page integration, rule-status, provenance, and error-state tests.

The database separates patient journeys (`patients`) from ordered steps (`journey_steps`) so source patient IDs, step IDs, order, and outcomes are retained. Stage 2 loads complete patient rows from SQLite and adds a normalized text column rather than overwriting the stored symptoms.

## Technology stack

- **Python 3.10+** for the application and modular analysis pipeline.
- **Pandas** for patient histories, ordered steps, and result tables; **NumPy** for numerical support.
- **Scikit-learn** for TF-IDF text vectors and cosine similarity.
- **SQLite** through Python's built-in `sqlite3` module for local storage; no separate database server is required.
- **Streamlit** for the six-section interactive dashboard.
- **Matplotlib** is included in the declared requirements for plotting; the current dashboard presents tables and metrics and does not yet render Matplotlib charts.
- **pytest** for automated core-module and dashboard integration tests.
- **Graphviz** is not used by the current implementation and is not a dependency.

## Setup

Python 3.10 or newer is recommended. SQLite and `sqlite3` are included with Python. From the project root:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Initialize or rebuild the database

Initialize the local database from the bundled synthetic sources:

```bash
PYTHONPATH=src python -c "from twincure.db import initialize_database; initialize_database('database/twincure.db', 'database/schema.sql', 'data/synthetic_patients.csv', 'data/synthetic_steps.csv', 'data/synthetic_rules.json')"
```

This creates or imports into `database/twincure.db`. The database is local runtime output and should not be committed.

## CSV-to-SQLite synchronization policy

The current importer uses `INSERT OR IGNORE`. This is intentionally repeatable, but **it does not update an existing database row when a CSV row with the same patient or step ID has changed**. Re-running initialization therefore adds previously unseen records but leaves existing IDs' stored values unchanged. It also does not delete rows removed from a CSV.

For an additive-only change that introduces new synthetic patient IDs and new step IDs, rerunning the initialization command imports those new rows. Ensure every step refers to an existing synthetic patient and that step IDs and `(patient_id, step_order)` values are unique.

For edits, removals, or reordered existing rows, use a controlled full rebuild because the local SQLite database is a derived copy of the synthetic CSV sources. From the project root, back up the current database, import the updated CSV files into a fresh temporary database, validate it, then replace the old database only after the checks succeed:

```bash
cp database/twincure.db database/twincure.backup.db
rm -f database/twincure.rebuilt.db
PYTHONPATH=src python -c "from twincure.db import initialize_database; initialize_database('database/twincure.rebuilt.db', 'database/schema.sql', 'data/synthetic_patients.csv', 'data/synthetic_steps.csv', 'data/synthetic_rules.json')"
PYTHONPATH=src python - <<'PY'
from twincure.db import list_patients, list_steps
p = 'database/twincure.rebuilt.db'
patients, steps = list_patients(p), list_steps(p)
assert patients and steps
assert all(row['dataset_label'] == 'SYNTHETIC_EDUCATIONAL_ONLY' for row in patients)
print(f'Validated rebuilt database: {len(patients)} journeys, {len(steps)} steps')
PY
mv database/twincure.rebuilt.db database/twincure.db
```

Keep the backup until the rebuilt database has been reviewed. Do not run a rebuild while the application is using the database. This workflow changes no schema; it rebuilds the local database from the CSV and JSON sources. A future incremental upsert/synchronization feature would need explicit rules for changed rows, deleted rows, and changed step order, and is not implemented in this stage.

## Stage 2 functions and algorithms

### `preprocessing.py`

- `load_patient_histories(db_path)` reads all patient rows from SQLite into a Pandas DataFrame in stable patient-ID order.
- `standardize_symptom_text(value)` handles null/blank values, lowercases text, converts punctuation to spaces, and collapses repeated whitespace. For example, `" CHEST,  Discomfort! "` becomes `"chest discomfort"`.
- `preprocess_patient_histories(histories)` returns a copy with `normalized_symptoms`; it leaves the original `symptoms_text` and all other patient fields intact.
- `load_preprocessed_histories(db_path)` combines loading and preprocessing.

### `similarity.py`

- `load_similarity_threshold(settings_path)` reads `similarity_threshold` from TOML. The initial `0.20` value in `config/settings.toml` is a demonstration setting, not a validated or clinical cutoff.
- `rank_similar_patients(query_symptoms, histories, settings_path)` standardizes the new synthetic case, constructs TF-IDF word vectors for the query and stored descriptions, then calculates cosine similarity. TF-IDF gives more weight to words that are informative within this small collection; cosine similarity compares vector direction, producing a score from 0 to 1.
- The function returns only rows meeting or exceeding the configured threshold, ranked from highest score downward. The original patient fields remain in every result; `normalized_symptoms` and `similarity_score` are added.
- `find_similar_patients(db_path, query_symptoms, settings_path)` loads histories from SQLite and runs the ranking workflow.
- Empty query text returns status `empty_query`. No stored candidates or no candidates meeting the cutoff return explicit no-match statuses and an empty result table, rather than a fabricated match.

Similarity scores indicate textual overlap only. They are not diagnostic scores or evidence that a treatment will work.

## Stage 3: Treatment path analysis

Treatment path analysis examines the stored steps for the patient journeys returned by Stage 2. It describes historical segments from the synthetic dataset only; it does not merge journeys, recommend care, or infer that a step caused an outcome.

### How segments are generated

`treatment_analysis.py` receives the Stage 2 result, uses its patient IDs and similarity scores to select source journeys, and loads those journeys' ordered steps from SQLite. Each individual step is generated as a segment. It also generates every contiguous window up to `max_segment_length` from `config/settings.toml` (initially 2). Windows are generated within one patient journey only, preserve original step order, and do not cross a gap in step order. Segment IDs encode source patient and start/end step positions. Each output retains source patient ID, step IDs, ordered positions, descriptions, outcomes, mean, classification, and the source similarity score.

### Segment outcome categories

- **Positive:** every step in the segment has a recorded synthetic outcome, the segment mean is at least +0.5, and at least one step has outcome +1.
- **Negative:** every step has a recorded outcome, the segment mean is at most −0.5, and at least one step has outcome −1.
- **Mixed/Unclear:** all other cases, including neutral/mixed results or any missing/invalid outcome.

Missing outcomes are not imputed. When any outcome is missing, the segment is retained, its mean is left missing, and it is labelled Mixed/Unclear. Missing descriptions remain missing rather than being replaced with invented text. Journey steps with no usable patient ID or step order cannot safely be sequenced, so they are not used to generate segments.

### Beginner-friendly module overview

- `load_max_segment_length(settings_path)` reads and validates the configured maximum from TOML.
- `analyze_treatment_paths(similarity_result, db_path, settings_path)` loads steps for the matched source journeys, preserving each Stage 2 similarity score.
- `generate_treatment_segments(steps, similarity_by_patient, max_segment_length)` sorts and groups steps by patient, splits at order gaps, and emits single-step and contiguous multi-step windows without crossing journey boundaries.
- The segment builder calculates a mean only when all outcomes are present and classifies using the thresholds above. It preserves the original step IDs, orders, descriptions, and outcomes in each row.

This is an educational analysis of constructed outcomes, not real-world clinical evidence or a treatment decision tool.

## Stage 4A: Segment selection and compatibility validation

Stage 4A takes Stage 3 segments and identifies positive segments that can be considered together as candidate pairs for a later stage. It does **not** merge steps or build a hybrid journey.

`select_candidate_segments(segments, db_path, rules_path)` in `segment_selection.py`:

- Retains only `Positive` segments as positive candidates. Negative and Mixed/Unclear records remain in the rejected/reporting output with an explanation.
- Deduplicates repeated segment IDs so the same source segment cannot be counted twice.
- Compares candidates only across distinct source patient journeys. Same-journey pairs are explicitly not compared; a patient therefore contributes no more than once to a candidate pair/component.
- Looks up the source step categories in SQLite, using original step IDs, and applies only the JSON compatibility rules. Explicit incompatibility rejects the pair; every pair must be explicitly compatible; unknown or missing rules remain unconfirmed.
- Checks existing step order inside each segment against explicit sequence rules, then checks whether exactly one direction between the two journeys is allowed. A unique direction is reported as ordered; two valid directions are ambiguous; absent support is unresolved; explicitly prohibited direction(s) are invalid. Ambiguous and unresolved orders are never chosen arbitrarily.
- Returns positive candidates, rejected segments and reasons, pairwise compatibility checks, sequence results, eligible candidate pairs, eligible segment rows, and decision explanations. Each result preserves source patient ID, segment ID, source step IDs, step positions, outcomes, and similarity score.

The existing rules file explicitly marks OBSERVATION–REVIEW and REVIEW–SUPPORT as compatible, and OBSERVATION–SUPPORT as unknown. The directed order rules allow OBSERVATION → REVIEW and REVIEW → SUPPORT. For example, an OBSERVATION segment from one journey plus a REVIEW segment from another can qualify when those exact categories are present: compatibility is explicit and the order has one supported direction. OBSERVATION plus SUPPORT remains unconfirmed because compatibility is unknown, even if both outcomes are positive. A pair with an explicit incompatible rule is rejected. The starter rules contain no incompatible pair, so incompatible behavior is tested with an isolated test-only rules fixture; the project’s synthetic rules file is not changed.

All rule labels are synthetic educational metadata, not clinical guidance. No new compatibility rule is inferred from positive outcomes.

## Stage 4B: Hybrid journey generation

`generate_hybrid_journeys(selection_result)` in `hybrid_generation.py` consumes the structured result from Stage 4A and uses only its `eligible_combinations`. It does not rerun compatibility checks, infer rules, or accept segments from the positive-candidate list alone. Each Stage 4A-approved combination becomes its own separate proposed hybrid journey; alternative combinations are not concatenated into one larger journey.

For each candidate, the module follows Stage 4A's `ordered_segment_ids` exactly. It preserves the original step order inside each source segment, then appends those source steps in the validated between-segment order. It never chooses an order for an unresolved or ambiguous pair because Stage 4A does not mark those pairs eligible.

Every output step includes a deterministic hybrid journey ID, hybrid position, original source patient and segment IDs, source step ID, original step order, description, outcome, segment classification, and source similarity score. Missing description/outcome values remain missing. Segment-level inclusion explanations and Stage 4A compatibility/sequence rule explanations are included in the structured output.

The module performs defensive structural checks against duplicate segment IDs, duplicate source step IDs, repeated source journeys, non-positive classifications, and incomplete provenance. Invalid eligible-combination records are rejected with an explanation. Stage 4A's rejected segments are passed through for reporting. If nothing can be generated, the result uses `no_valid_hybrid`; no path is forced. IDs are based on a stable hash of the ordered segment IDs, so identical inputs produce deterministic output. No success or confidence score is calculated.

This is a synthetic software demonstration only. A generated hybrid is not claimed to be medically better than any source journey and is not a real-world recommendation.

## Stage 5: Synthetic evidence-support scoring

`score_hybrid_evidence(hybrid_generation_result)` in `evidence_scoring.py` calculates a transparent evidence-support summary separately for each hybrid in a Stage 4B output. It uses the original similarity and step outcome provenance. It is strictly a summary of the available synthetic examples; it is **not** a clinical success probability, treatment-effectiveness prediction, or medical recommendation. Scores must not be compared with clinical success rates.

The approved formula and fixed weights are:

```text
Score = 100 × (0.4118S + 0.3529O + 0.2353A)
```

The weights sum to 1.0. With valid components in the range 0–1, the score is between 0 and 100. The returned score is rounded to two decimal places for display; individual component calculations are also returned.

- **S — Similarity support:** mean of the original Stage 2 cosine similarities for unique source patient journeys contributing steps to that hybrid. A patient appearing in multiple steps or segments contributes once. Missing, invalid, out-of-range, or conflicting similarity values make S undefined.
- **O — Outcome support:** for each unique journey contributing a selected Positive segment in that hybrid, collect the unique source step IDs from its selected Positive segments. Average their original outcomes, map each journey mean using `O_individual = (mean outcome + 1) / 2`, then average those journey-level support values. Repeated/overlapping step IDs count once. Missing or invalid selected outcomes make O undefined; no outcome is invented.
- **A — Agreement support:** calculate separately for each hybrid. Start with only the unique source patient journeys contributing steps to that hybrid. For each of those journeys, use its available candidate evidence carried in the Stage 4B result—including positive, negative, mixed, or rejected segments—and deduplicate source segments and source step IDs. Give each journey one positive, negative, or neutral/unclear vote using the approved journey-level outcome thresholds (positive mean ≥ +0.5 with at least one +1; negative mean ≤ −0.5 with at least one −1; otherwise neutral/unclear). Exclude neutral/unclear votes from the denominator. `A = positive votes / (positive votes + negative votes)`. If the denominator is zero, A is undefined.

For a Stage 4B result containing multiple alternative hybrids, each hybrid gets its own S, O, and A. A rejected segment can affect a hybrid's A only when its source patient contributes to that hybrid. An unrelated patient's positive, negative, or neutral vote is excluded, even if that patient appears in a different alternative hybrid or in the rejected-segment report. This keeps agreement evidence aligned with the hybrid being scored while still using available negative/mixed evidence from its own contributing journeys.

### Worked example

Suppose a hybrid uses two unique source journeys with similarities 0.90 and 0.70, so `S = (0.90 + 0.70) / 2 = 0.80`. Its selected positive steps have outcomes +1 for each contributing journey, so `O = 1.00`. The first journey has a positive vote. The second has a selected +1 step plus three unique −1 outcomes from its available rejected candidate segments; its unique-step mean is −0.5, so it receives one negative vote. Thus `A = 1 / (1 + 1) = 0.50`. Any negative patient unrelated to this hybrid is not included.

```text
Score = 100 × (0.4118×0.80 + 0.3529×1.00 + 0.2353×0.50)
      = 80.00 synthetic evidence-support points
```

This number describes only this constructed educational evidence. It is not an 80% chance of success.

### Missing evidence behavior

If S, O, or A is undefined, the result status is `insufficient_evidence`, the final score is `null`, and the output names each missing component and reason. The system never replaces an undefined value with zero. A missing outcome on a selected Positive source step makes O unavailable; a journey with incomplete outcomes cannot receive a positive/negative agreement vote and is treated as neutral/unclear. If no positive/negative votes remain, A is undefined. Duplicate source journeys and source steps are deduplicated before their respective calculations; conflicting duplicate values are not silently resolved.

The result includes the hybrid ID, unique contributing patient IDs, S/O/A values, weights, supporting calculations and source IDs, score, status, and a plain-language explanation.

## Tests

Run the complete Stage 1 through Stage 6F test suite from the project root:

```bash
PYTHONPATH=src pytest -q
```

The suite covers synthetic-only database import, ID/order/outcome preservation, preprocessing and missing text, similarity matching and configured thresholds, treatment segment labels, Stage 4A compatibility/sequence decisions, Stage 4B ordering/provenance/duplicate prevention, Stage 5 component math and hybrid-specific population selection, and dashboard behavior through Stage 6F. Test-only rules fixtures do not alter project rules. The complete suite count is reported in the final dashboard test section below.

## Stage 6A: Streamlit dashboard foundation

The dashboard has six implemented navigation sections: Home, Patient Similarity, Treatment Analysis, Compatibility Analysis, Hybrid Journey, and Evidence Scoring. The final Evidence Scoring page is documented in Stage 6F below.

The Home page introduces TwinCure, displays the warning “Synthetic educational prototype — not for clinical use,” reports whether SQLite is accessible, and shows patient, journey-step, sequence-rule, and compatibility-rule counts read from the existing database. Patient and step totals use the existing `list_patients()` and `list_steps()` functions; rule totals use read-only SQL through the existing `connect()` helper. Counts are not hardcoded.

If the configured database is missing, the page reports it as unavailable without allowing SQLite to create an empty replacement file. If the database exists but is unreadable or lacks expected tables, an understandable error is shown and metric values are displayed as unavailable. Initialize the database first if needed using the earlier instructions in this README.

Streamlit was already present in `requirements.txt`; no dependency-file change was needed. From the project root, install dependencies and launch the dashboard with:

```bash
pip install -r requirements.txt
PYTHONPATH=src streamlit run src/twincure/app.py
```

By default, the app reads `database/twincure.db`. To test or use another database, set `TWINCURE_DB_PATH` to its path. Relative override paths are resolved from the project root. For example:

```bash
TWINCURE_DB_PATH=/path/to/twincure.db PYTHONPATH=src streamlit run src/twincure/app.py
```

Run the complete Stage 1–6F tests from the project root:

```bash
PYTHONPATH=src pytest -q
```

The suite includes Streamlit AppTest checks for live Home metrics, Patient Similarity, Treatment Analysis, Compatibility Analysis, Hybrid Journey ordering/provenance/rejection states, and Evidence Scoring component values, unique-journey votes, insufficient evidence, no-hybrid, and database/rule error states. It also exercises cross-page navigation. The complete suite currently contains 99 passing tests.

## Stage 6B: Patient Similarity page

The Patient Similarity page in `dashboard/patient_similarity.py` submits user-entered synthetic symptom text to the existing Stage 2 `find_similar_patients()` workflow. That function uses `load_preprocessed_histories()` from preprocessing, the TF-IDF vectorizer, cosine similarity, and the configured `similarity_threshold` from `config/settings.toml`. The dashboard does not implement a second matching algorithm or define a separate threshold.

On a match, the page displays the synthetic patient ID, original symptom description, and the Stage 2 cosine similarity score, plus an explanation of TF-IDF/cosine text matching. Empty input, text with no usable words, no matches, and unavailable database states have separate messages. The page explicitly warns that text similarity does not establish medical equivalence or treatment effectiveness and does not display diagnoses, prescriptions, or clinical recommendations. A missing database is detected before the Stage 2 reader is called, avoiding accidental creation of an empty SQLite file.

The dashboard tests compare displayed scores and matching patient IDs with direct calls to Stage 2, and cover empty and invalid inputs, no-match queries, missing-database handling, and navigation to all implemented pages.

## Stage 6C: Treatment Analysis page

The Treatment Analysis page in `dashboard/treatment_analysis.py` accepts synthetic symptom text, calls Stage 2 `find_similar_patients()` to obtain threshold-qualified journey matches, and passes the resulting object and database path to Stage 3 `analyze_treatment_paths()`. Segment construction and classification remain entirely in the existing Stage 3 module; the dashboard adds no second algorithm and does not change its thresholds.

Results are grouped into Positive, Negative, and Mixed / Unclear. Each segment displays its source patient ID and segment ID, Stage 2 similarity, Stage 3 mean outcome, and each original source step ID, step order, description, and outcome. Missing descriptions and outcomes stay explicitly marked as missing; unavailable means are not filled in.

The page displays the existing Stage 3 criteria: Positive means ≥ +0.5 with at least one +1; Negative means ≤ −0.5 with at least one −1; Mixed / Unclear includes all other cases, including missing or invalid outcomes. The page labels these results as classifications of artificial example outcomes only. Positive does not mean a treatment is medically effective, and the page does not give diagnoses, prescriptions, or clinical recommendations.

Empty and unusable queries, no similar journeys, no generated segments, database errors, and unexpected processing errors each receive a clear message. Tests compare page counts and displayed provenance to direct Stage 2/3 results, check every category, cover missing/unclear outcomes and error states, and confirm Home, Patient Similarity, and the remaining later-stage pages still work.

## Stage 6D: Compatibility Analysis page

The Compatibility Analysis page in `dashboard/compatibility_analysis.py` connects the existing modules in order: Stage 2 `find_similar_patients()` finds threshold-qualified synthetic journeys; Stage 3 `analyze_treatment_paths()` produces their contiguous source segments; and Stage 4A `select_candidate_segments()` applies the explicit JSON compatibility and sequence rules. The dashboard reuses these implementations and does not introduce a second similarity, segment-generation, compatibility, or ordering algorithm. It does not modify `data/synthetic_rules.json`.

The page uses compact tables for **eligible Positive candidate segments**, **eligible combinations**, and **rejected/unresolved decisions**. Candidate tables retain source patient ID, segment ID, source step IDs, original order, and descriptions. Each eligible pair displays both source journeys, Stage 4A's validated order, and the compatibility and sequence reasons. Rejected combinations include explicit incompatibility or a prohibited sequence. Unresolved combinations include unknown compatibility, no applicable ordering, or multiple possible orders. For ambiguous pairs, both possibilities are listed and neither is selected. Non-positive segments and positive segments without any eligible pair are shown separately; same-journey pairs are labelled not compared.

Unknown compatibility is never treated as compatible; an ambiguous or unestablished sequence is never accepted. Only Stage 4A's explicitly eligible combinations are described as able to proceed to the later Hybrid Journey stage. These are prototype-rule decisions only, not clinical safety checks or medical approval. The page displays no medical recommendations and reports missing database or rules files without creating an empty database.

The Stage 6D tests exercise the real Stage 2→3→4A pipeline on the starter synthetic dataset, including eligible pairs, unknown compatibility, and non-positive segments. Isolated test-only JSON fixtures exercise explicit incompatibility, prohibited order, no applicable order, and ambiguous order; those fixtures are created only in temporary test directories and leave the project's synthetic rules unchanged. Tests also cover missing database/rule files and regression behavior for Home, Patient Similarity, and Treatment Analysis.

## Stage 6E: Hybrid Journey page

The Hybrid Journey page in `dashboard/hybrid_journey.py` reuses the pipeline in order: Stage 2 `find_similar_patients()` applies the configured TF-IDF/cosine threshold; Stage 3 `analyze_treatment_paths()` creates contiguous source segments; Stage 4A `select_candidate_segments()` checks positive segments against the explicit compatibility and sequence rules; Stage 4B `generate_hybrid_journeys()` generates separate examples only from Stage 4A's `eligible_combinations`. The dashboard does not implement a second hybrid-generation algorithm and does not modify the core stages or rules.

Every generated hybrid shows its deterministic hybrid journey ID, unique contributing patient count, number of included steps, and Stage 4B generation status. A step table displays the hybrid position, original patient ID, source segment ID, source step ID, original step order, description, and synthetic outcome. Segment-level explanations identify each source journey and why the segment was included. Compatibility and sequence explanations are copied from Stage 4A; the dashboard does not recalculate or choose an order. Each eligible Stage 4A combination becomes a separate hybrid example rather than being combined with alternatives.

If Stage 4B reports `no_valid_hybrid`, the page displays that status and its explanation and does not force a journey. It also distinguishes empty queries, no similar patients, no segments, no Stage 4A-eligible combinations, missing database/rule files, and unexpected errors. Stage 4B rejected or duplicate combinations are shown with the original rejection reasons. The page prominently labels all data as synthetic educational examples and states that artificial outcomes and prototype rules are not evidence of medical safety, effectiveness, or suitability. It does not provide diagnoses, prescriptions, or clinical recommendations.

The Stage 6E tests compare displayed step rows with direct Stage 2→3→4A→4B results, verify Stage 4A is the only source of combinations passed to Stage 4B, and exercise ordering, source provenance, compatibility/sequence explanations, invalid duplicate rejection, no eligible combinations, `no_valid_hybrid`, missing database/rules, and navigation regression for all existing pages.

## Stage 6F: Evidence Scoring page

The final dashboard section in `dashboard/evidence_scoring.py` runs the existing pipeline in order: Stage 2 `find_similar_patients()`, Stage 3 `analyze_treatment_paths()`, Stage 4A `select_candidate_segments()`, Stage 4B `generate_hybrid_journeys()`, then Stage 5 `score_hybrid_evidence()`. It uses the existing configured similarity threshold and explicit synthetic rules. Stage 5 owns all component calculations and the score formula; the page only formats and explains the returned results.

For each hybrid, the page displays Stage 5's returned **S**, **O**, and **A** values plus the final 0–100 evidence-support score. S is described as mean cosine similarity across unique source journeys. O shows each unique contributing journey's source-step IDs, outcomes, mean, and mapped outcome support. A displays the source patient IDs and their positive, negative, or neutral/unclear journey votes. Its population is hybrid-specific: only journeys contributing to that hybrid are included. Overlapping source steps are deduplicated; available rejected/non-positive candidate evidence from those contributing journeys may affect their vote, while unrelated patients do not. Neutral/unclear votes are excluded from the positive-plus-negative denominator.

The displayed formula is **`Score = 100 × (0.4118S + 0.3529O + 0.2353A)`**. The UI uses Stage 5's final score as returned and does not round component values before that calculation. If Stage 5 returns `insufficient_evidence`, the affected component is shown as undefined, Stage 5's missing-evidence reasons are displayed, and the final score is marked **Withheld**—missing values are never replaced with zero. If Stage 4B produces no valid hybrid, no score is displayed.

The page displays this limitation: “The evidence-support score measures how strongly the synthetic example data supports a generated hybrid journey under TwinCure's prototype scoring rules. It is not a probability of treatment success, clinical efficacy score, or medical recommendation.” All displayed patient journeys and outcomes are synthetic educational examples. The score is not real-world clinical evidence, a diagnosis, a prescription, a treatment recommendation, or clinically validated probability. Do not enter real patient information.

Install and launch from the project root using the commands in Setup above (`pip install -r requirements.txt`, then `PYTHONPATH=src streamlit run src/twincure/app.py`). Run the complete suite with `PYTHONPATH=src pytest -q`. The Stage 6F tests compare displayed values with direct Stage 5 results, verify hybrid-specific votes and unique-step deduplication, show that rejected evidence from a contributing journey can change A while unrelated-patient evidence cannot change A or the final score, and cover insufficient evidence, no valid hybrids, database/rule failures, and routing for all six dashboard sections. The full suite currently contains 99 passing tests.


## Limitations and possible future enhancements

The starter dataset is deliberately small and fully synthetic. The 0.20 similarity threshold is a configurable demonstration value, not an empirically selected cutoff. The compatibility and sequence rules are handcrafted educational metadata, not clinical protocols. TF-IDF measures text overlap and does not establish clinical equivalence. The S/O/A evidence-support score summarizes only the available constructed examples; it is not a real-world success probability, treatment-effectiveness estimate, or validated confidence measure. Synthetic outcomes do not demonstrate causality. SQLite synchronization currently uses `INSERT OR IGNORE`, so modifying existing source rows requires the documented controlled database rebuild. The dashboard currently shows tables and metrics; Matplotlib is included in dependencies but is not used for charts.

Possible future academic enhancements include a larger, still synthetic and explicitly labelled scenario set; additional automated evaluation of similarity-threshold behavior; richer visualization of provenance and ordering decisions; and a carefully specified CSV-to-SQLite synchronization workflow. Any future use of real health data would require appropriate governance, privacy safeguards, domain review, and a separate validation effort; it is outside this prototype's scope.

## Disclaimer

**TwinCure is a synthetic educational software prototype, not a medical device or clinically validated system. It is not intended for diagnosis, prescribing, treatment planning, or any clinical decision-making. Similarity scores, segment labels, compatibility decisions, hybrid journeys, and evidence-support scores are experimental outputs based solely on constructed examples and rules. They are not clinical evidence, medical advice, or predictions of patient outcomes. Do not enter real patient information or personally identifiable information.**
