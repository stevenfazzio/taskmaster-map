"""Unit tests for the stage-00b wiki parser, on small synthetic tmtable wikitext."""

import importlib.util
import math
import sys
from pathlib import Path

PIPELINE = Path(__file__).resolve().parent.parent / "pipeline"
sys.path.insert(0, str(PIPELINE))
_spec = importlib.util.spec_from_file_location("wiki00b", PIPELINE / "00b_fetch_wiki.py")
w = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w)

HEADER = """{| class="tmtable"
|- class="tmtableheader"
! Task
! Description
! {{color|FFF|Ann}}
! {{color|FFF|Bob}}
! {{color|FFF|Cat}}<onlyinclude>
"""


def _page(rows: str) -> str:
    return HEADER + rows + '|- class="tmtabletotal"\n! colspan="2" | Total\n| 1\n| 2\n| 3</onlyinclude>\n|}\n'


def test_simple_rows_prefixes_and_scores():
    contestants, tasks = w.parse_episode(
        _page(
            """|- class="tmtablerow"
| [[Best thing|1]]
| '''Prize:''' The [[w:Thing|best]] thing.<ref>x</ref>
| '''5'''
| -
| 3
|- class="tmtablerow"
| [[Live one|2]]
| '''Team Live:''' Shout the most.
| colspan="2" | 4
| 1
|- class="tmtablerow"
| T
| Guess the number.
| ✔
| ✘
| DQ
"""
        )
    )
    assert contestants == ["Ann", "Bob", "Cat"]
    t1, t2, t3 = tasks
    assert t1["task_num"] == "1" and t1["scores"][0] == 5 and math.isnan(t1["scores"][1]) and t1["scores"][2] == 3
    brief, flags = w.brief_and_flags(t1["desc_raw"], t1["task_num"])
    assert brief == "The best thing." and flags == {"is_prize": 1, "is_live": 0, "is_tiebreak": 0}
    brief, flags = w.brief_and_flags(t2["desc_raw"], t2["task_num"])
    assert brief == "Team Live: Shout the most." and flags["is_live"] == 0  # upstream convention
    assert t2["scores"] == [4, 4, 1]  # colspan shares the score
    assert t3["scores"][:2] == [1.0, 0.0] and math.isnan(t3["scores"][2])  # ticks; DQ -> NaN
    assert w.brief_and_flags(t3["desc_raw"], "T")[1]["is_tiebreak"] == 1


def test_multirow_task_sums_subrows_and_counts_rowspan_once():
    _, tasks = w.parse_episode(
        _page(
            """|- class="tmtablerow"
| rowspan=3 | [[Branch task|4]]
| Pick a path.
| 0
| 0
| rowspan=2 | 2
|- class="tmtablerow"
| Path A.
| 5
| -
|- class="tmtablerow"
| <small>'''Bonus''' for style.</small>
| 1
| 0
| 0
|- class="tmtablerow"
| [[Next|5]]
| Next task.
| 1
| 2
| 3
"""
        )
    )
    assert [t["task_num"] for t in tasks] == ["4", "5"]
    assert tasks[0]["desc_raw"].strip() == "Pick a path."
    assert tasks[0]["scores"] == [6.0, 0.0, 2.0]  # 0+5+1, 0+0, rowspan'd 2 counted once
    assert tasks[1]["scores"] == [1.0, 2.0, 3.0]


def test_clean_markup_and_split():
    assert w.clean_markup("'''Bold''' [[A|b]] {{wp|X|Y}} &amp; <br/>z") == "Bold b Y & z"
    assert w.split_top_level("a|[[b|c]]|{{d|e}}", "|") == ["a", "[[b|c]]", "{{d|e}}"]
    assert w.cell_attrs_and_text("rowspan=4 | [[T|3]]") == ({"rowspan": "4"}, " [[T|3]]")
