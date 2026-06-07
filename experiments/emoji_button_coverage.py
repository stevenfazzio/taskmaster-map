"""How well does each of the 16 emoji filter buttons cover its concept?

The buttons (06_visualize.py) match a task iff the button's single glyph is a
substring of the task's gist (05_extract_fields.py). But the gist is generated
open-vocabulary, so a music task may carry 🎭/🎤 and never the canonical 🎵 — a
silent false negative. This script quantifies that gap for all 16 buttons:

  canonical  — gists containing the exact button glyph (what the button matches today)
  family     — gists containing ANY glyph in the concept's synonym set
  gap        — family but NOT canonical (tasks the button silently misses)
  + the alternative glyphs those gap tasks used instead

Run before and after any prompt/matching change as the before/after measurement.
Read-only over data/structured_fields.parquet.
"""

from __future__ import annotations

import sys
from collections import Counter

sys.path.insert(0, "pipeline")

import pandas as pd
from config import DATA_DIR

# (canonical glyph, label, synonym family). Family INCLUDES the canonical.
# Concrete-object buttons have thin families; concept/action buttons have wide ones.
# Known overlaps: 🎩∈clothing&hats, 🎭∈song&film — noted in output.
BUTTONS = [
    ("🥚", "egg", "🥚🍳🐣🐤🐥"),
    ("🎈", "balloon", "🎈🎊"),
    ("🦆", "duck", "🦆🦢🐤🐥🐦"),
    ("🥥", "coconut", "🥥🌴"),
    ("🧻", "loo roll", "🧻🚽🧼"),
    ("🗑️", "wheelie bin", "🗑️🚮♻️🗙"),
    ("👕", "clothing", "👕👚👗👔👖🧥🩳🧦🧣🥼👙👘🦺🥾👞👟🥿👠👡👢🧤"),
    ("🎩", "hats", "🎩👒🧢⛑️👑🪖🎓"),
    ("🎁", "present", "🎁🎀💝"),
    ("🎵", "song", "🎵🎶🎼🎤🎙🎧🎹🎸🎻🥁🎷🎺🪕🪗🎭"),
    ("🎬", "film", "🎬🎥📹📽️🎞️🎦🍿📷📸🤳"),
    ("💥", "smash", "💥🔨🪓💣🧨⚒️🛠️🔥"),
    ("🤢", "gross", "🤢🤮🤧💩🦠🪳🐛🤧😷"),
    ("🙈", "hide/blindfold", "🙈🫣🥷🫥🕵️🔍"),
    ("🤫", "silent", "🤫🔇🔕🙊🤐"),
    ("⚖️", "balancing", "⚖️🤹🧘🩰🤸🛹"),
]


def main() -> None:
    df = pd.read_parquet(DATA_DIR / "structured_fields.parquet")
    em = df["task_emoji"].fillna("")
    n = len(df)
    print(f"corpus: {n} tasks ({(em.str.len() > 0).sum()} with gists)\n")

    hdr = f"{'button':18s} {'canon':>5s} {'family':>6s} {'gap':>4s} {'miss%':>5s}  alternatives used in the gap"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for glyph, label, family in BUTTONS:
        fam = [g for g in family if g != "️"]  # strip variation selectors for membership
        canon = em.str.contains(glyph, regex=False)
        has_fam = em.apply(lambda s, fam=fam: any(g in s for g in fam))
        gap = has_fam & ~canon
        alts = Counter()
        for s in em[gap]:
            for g in fam:
                if g != glyph and g in s:
                    alts[g] += 1
        miss = gap.sum() / has_fam.sum() if has_fam.sum() else 0.0
        alt_str = " ".join(f"{g}{c}" for g, c in alts.most_common(8))
        print(f"{glyph + ' ' + label:18s} {canon.sum():5d} {has_fam.sum():6d} {gap.sum():4d} {miss:4.0%}  {alt_str}")
        rows.append((label, canon.sum(), has_fam.sum(), gap.sum(), miss))

    tot_canon = sum(r[1] for r in rows)
    tot_fam = sum(r[2] for r in rows)
    print(
        f"\ntotals: {tot_canon} canonical matches vs {tot_fam} family matches "
        f"— buttons currently catch {tot_canon / tot_fam:.0%} of in-concept tasks"
    )


if __name__ == "__main__":
    main()
