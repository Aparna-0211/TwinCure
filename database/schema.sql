PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,
    age_group TEXT,
    disease TEXT NOT NULL,
    symptoms_text TEXT NOT NULL,
    journey_outcome INTEGER CHECK (journey_outcome IN (-1, 0, 1)),
    dataset_label TEXT NOT NULL CHECK (dataset_label = 'SYNTHETIC_EDUCATIONAL_ONLY')
);

CREATE TABLE IF NOT EXISTS journey_steps (
    step_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    step_order INTEGER NOT NULL CHECK (step_order >= 1),
    step_category TEXT NOT NULL,
    step_description TEXT NOT NULL,
    step_outcome INTEGER NOT NULL CHECK (step_outcome IN (-1, 0, 1)),
    outcome_notes TEXT NOT NULL,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE,
    UNIQUE (patient_id, step_order)
);

CREATE TABLE IF NOT EXISTS sequence_rules (
    rule_id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_category TEXT NOT NULL,
    to_category TEXT NOT NULL,
    rule_type TEXT NOT NULL CHECK (rule_type IN ('allowed', 'prohibited')),
    rule_note TEXT NOT NULL,
    UNIQUE (from_category, to_category)
);

CREATE TABLE IF NOT EXISTS compatibility_rules (
    rule_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_a TEXT NOT NULL,
    category_b TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('compatible', 'incompatible', 'unknown')),
    rule_note TEXT NOT NULL,
    UNIQUE (category_a, category_b)
);
