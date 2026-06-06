"""Compare production labels (disambiguation ON) vs the no-disambiguation run.

Joins on task_id. Region membership is deterministic, so each task sits in the same
region in both runs — only the NAMES differ. The key question: without disambiguation,
do the ~134 prize-synonym tasks collapse to one (or a few) names?
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "pipeline")

import pandas as pd
from config import DATA_DIR

pd.set_option("display.max_colwidth", 60)
pd.set_option("display.width", 240)
pd.set_option("display.max_rows", 200)

PROD = DATA_DIR / "toponymy_labels.parquet"
RAW = DATA_DIR / "experiments" / "toponymy_labels_nodisambig.parquet"
RAW_JSON = DATA_DIR / "experiments" / "raw_cluster_names.json"

# The 6 near-synonyms (find-or-acquire prize tasks) + the 1 genuinely-distinct one.
SYNONYMS = [
    "Most [Superlative] Item Challenges",
    "Best Object Bring-In Tasks",
    "Best Thing Challenges",
    "Bring the Most Superlative Object",
    "Superlative Object Showcase Tasks",
    "Best/Nicest Object Superlative Challenges",
]
MAKEBUILD = "Make or Build the Best"

prod = pd.read_parquet(PROD).rename(columns={"label_layer_0": "prod_0", "label_layer_1": "prod_1"})
raw = pd.read_parquet(RAW).rename(columns={"label_layer_0": "raw_0", "label_layer_1": "raw_1"})
m = prod.merge(raw, on="task_id")

print("=" * 78)
print("VOCABULARY SIZE (layer 0)")
print(
    f"  production (disambig ON) : {prod.prod_0.nunique()} distinct names, "
    f"{(prod.prod_0 == 'Unlabelled').sum()} Unlabelled"
)
print(
    f"  no-disambig (raw)        : {raw.raw_0.nunique()} distinct names, {(raw.raw_0 == 'Unlabelled').sum()} Unlabelled"
)

print("=" * 78)
print("RAW per-cluster first-pass names (what the namer produced BEFORE disambiguation)")
rawnames = json.loads(RAW_JSON.read_text())
for layer, names in rawnames.items():
    dupes = pd.Series(names).value_counts()
    dupes = dupes[dupes > 1]
    print(f"  layer {layer}: {len(names)} clusters, {pd.Series(names).nunique()} distinct")
    if len(dupes):
        print(f"    repeated raw names: {dupes.to_dict()}")

print("=" * 78)
print("THE PRIZE DISTRICT: where do the 6 production synonyms land without disambiguation?")
district = m[m.prod_0.isin(SYNONYMS)]
print(f"  {len(district)} tasks across {len(SYNONYMS)} production names")
print(f"  → without disambiguation they collapse to {district.raw_0.nunique()} distinct name(s):")
print(district.raw_0.value_counts().to_string())
print()
print("  crosstab (production synonym  ×  raw name):")
print(pd.crosstab(district.prod_0, district.raw_0).to_string())

print("=" * 78)
print("CONTRAST: Make or Build the Best (the genuinely-distinct one)")
mb = m[m.prod_0 == MAKEBUILD]
print(f"  {len(mb)} tasks → raw names: {mb.raw_0.value_counts().to_dict()}")
