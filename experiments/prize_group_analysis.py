"""Why do the six prize islands separate? Two lenses on the 6 production-label groups.

A. Object-category diagnostic — if the islands split by object/sensory content (and an
   independent signal, the LLM emoji gist, agrees), a different embedding model would
   very likely fragment the same way → "accept" becomes the rational call.
B. Classic NLP metrics — do the groups differ on measurable axes (length, phrasing
   frame, lexical diversity) beyond "best/most object"? Tests whether the cut is
   capturing real sub-structure vs. arbitrary noise.

Uses embed_text (what was actually embedded → what drove the clustering).
"""

from __future__ import annotations

import re
import sys
from collections import Counter

sys.path.insert(0, "pipeline")

import numpy as np
import pandas as pd
from config import STRUCTURED_FIELDS_PARQUET, TASK_ROWS_PARQUET, TOPONYMY_LABELS_PARQUET
from sklearn.feature_extraction.text import TfidfVectorizer

# Order roughly largest→smallest island.
GROUPS = [
    "Most [Superlative] Item Challenges",
    "Best Object Bring-In Tasks",
    "Bring the Most Superlative Object",
    "Best Thing Challenges",
    "Superlative Object Showcase Tasks",
    "Best/Nicest Object Superlative Challenges",
]
# Shared "prize" vocabulary to strip so the *distinguishing* content shows through.
SHARED = {
    "best",
    "most",
    "thing",
    "things",
    "item",
    "items",
    "nice",
    "nicest",
    "good",
    "better",
    "object",
    "impressive",
    "the",
    "a",
    "an",
    "your",
    "you",
    "that",
    "of",
    "to",
    "with",
}


def main():
    lab = pd.read_parquet(TOPONYMY_LABELS_PARQUET)
    rows = pd.read_parquet(TASK_ROWS_PARQUET, columns=["task_id", "embed_text"])
    sf = pd.read_parquet(STRUCTURED_FIELDS_PARQUET, columns=["task_id", "task_emoji"])
    df = lab.merge(rows, on="task_id").merge(sf, on="task_id")
    df = df[df.label_layer_0.isin(GROUPS)].copy()
    df["text"] = df.embed_text.fillna("").str.strip()
    df["n_words"] = df.text.str.split().apply(len)
    df["n_chars"] = df.text.str.len()
    df["first_word"] = df.text.str.extract(r"^([A-Za-z']+)")[0].str.lower()
    df["has_rel_clause"] = df.text.str.contains(r"\b(that|which|who|makes|making)\b", case=False, regex=True)

    # ── A. distinctive content words per group (TF-IDF over 6 group-docs) ──
    docs = [" ".join(df.loc[df.label_layer_0 == g, "text"]) for g in GROUPS]
    vec = TfidfVectorizer(stop_words="english", token_pattern=r"[a-z']{3,}", lowercase=True)
    X = vec.fit_transform(docs)
    vocab = np.array(vec.get_feature_names_out())
    print("=" * 92)
    print("A. OBJECT-CATEGORY DIAGNOSTIC — top distinctive words + emoji gist per island")
    print("=" * 92)
    for i, g in enumerate(GROUPS):
        sub = df[df.label_layer_0 == g]
        scores = X[i].toarray().ravel()
        top = [w for w in vocab[scores.argsort()[::-1]] if w not in SHARED][:8]
        emojis = Counter("".join(sub.task_emoji.fillna("").tolist()))
        top_emoji = " ".join(e for e, _ in emojis.most_common(8))
        print(f"\n[{len(sub):>2}] {g}")
        print(f"     top words : {', '.join(top)}")
        print(f"     emoji gist: {top_emoji}")
        print(f"     samples   : {' | '.join(sub.text.head(4))}")

    # ── B. classic NLP metrics per group ──
    print("\n" + "=" * 92)
    print("B. CLASSIC NLP METRICS per island")
    print("=" * 92)
    agg = df.groupby("label_layer_0").agg(
        n=("task_id", "size"),
        words_mean=("n_words", "mean"),
        words_med=("n_words", "median"),
        words_std=("n_words", "std"),
        chars_mean=("n_chars", "mean"),
        rel_clause_pct=("has_rel_clause", lambda s: 100 * s.mean()),
    )
    # lexical diversity (type-token ratio) per group
    ttr = {}
    for g in GROUPS:
        toks = re.findall(r"[a-z']+", " ".join(df.loc[df.label_layer_0 == g, "text"]).lower())
        ttr[g] = len(set(toks)) / max(len(toks), 1)
    agg["ttr"] = pd.Series(ttr)
    agg = agg.reindex(GROUPS).round(2)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 40)
    print(agg.to_string())

    # first-word (phrasing frame) distribution
    print("\nphrasing frame — first-word distribution per island:")
    for g in GROUPS:
        fw = df.loc[df.label_layer_0 == g, "first_word"].value_counts().head(4)
        print(f"  {g[:42]:<42} {dict(fw)}")

    # significance: does word-count differ across the 6 groups?
    try:
        from scipy.stats import kruskal

        samples = [df.loc[df.label_layer_0 == g, "n_words"].values for g in GROUPS]
        h, p = kruskal(*samples)
        print(f"\nKruskal-Wallis on word-count across the 6 islands: H={h:.2f}, p={p:.4f}")
    except Exception as e:  # noqa: BLE001
        print(f"\n(kruskal skipped: {e})")


if __name__ == "__main__":
    main()
