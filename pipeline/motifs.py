"""Single source of truth for the 16 emoji filter motifs.

Each motif drives four consumers, so it lives here to keep them from drifting:
  - the EMOJI_SYSTEM literal-use guardrail (stage 05) — uses `glyph` + `sense`
  - the motif-tag classifier prompt (stage 05)        — uses `key` + `definition`
  - the filter buttons in the strip (stage 06)        — uses `glyph` + `label` + `key`
  - the enforcement check (stage 05)                  — uses `glyph` + `key`

`sense` is the terse meaning the witty gist is allowed to draw the glyph for; `definition`
is the fuller rule the classifier tags a task by. They describe the SAME concept at two
altitudes — keep them consistent when editing. `key` is the stable id (never change it
once gists/tags are cached); `glyph`/`label` are user-facing.
"""

from __future__ import annotations

MOTIFS = [
    {
        "glyph": "🥚",
        "key": "egg",
        "label": "egg",
        "sense": "an egg",
        "definition": "A real, fake, or chocolate egg is a central object of the task.",
    },
    {
        "glyph": "🎈",
        "key": "balloon",
        "label": "balloon",
        "sense": "a balloon",
        "definition": "A balloon or balloons are a central object (inflating, popping, transporting…).",
    },
    {
        "glyph": "🦆",
        "key": "duck",
        "label": "duck",
        "sense": "a duck",
        "definition": "A duck — live, rubber, or toy — is a central object.",
    },
    {
        "glyph": "🥥",
        "key": "coconut",
        "label": "coconut",
        "sense": "a coconut",
        "definition": "A coconut is a central object.",
    },
    {
        "glyph": "🧻",
        "key": "loo_roll",
        "label": "loo roll",
        "sense": "a toilet roll",
        "definition": "A toilet/loo roll or its cardboard tube is a central object.",
    },
    {
        "glyph": "👕",
        "key": "clothing",
        "label": "clothing",
        "sense": "clothing/a garment",
        "definition": (
            "Wearing, making, or altering clothing — garments, footwear, gloves, or headwear (hats, "
            "helmets) — as a CENTRAL part of the task. An incidental 'while wearing X' costume "
            "constraint does not count."
        ),
    },
    {
        "glyph": "🎁",
        "key": "present",
        "label": "present",
        "sense": "a gift/present",
        "definition": (
            "Giving, making, wrapping, or choosing a gift or present. NOT abstract 'bring the "
            "best/most X' prize tasks judged purely on a quality."
        ),
    },
    {
        "glyph": "🎵",
        "key": "music",
        "label": "music",
        "sense": "music or singing",
        "definition": (
            "Actual music — singing, a song, playing or making music, an instrument, or composing. "
            "A general performance, dance, presentation, or sketch with NO actual music does not count."
        ),
    },
    {
        "glyph": "🎬",
        "key": "film",
        "label": "film",
        "sense": "film/video/acting a scene",
        "definition": (
            "Making a film or video, photography, or acting out a scene or sketch. A pure song or "
            "dance performance is music, not film; and the mere fact that every task is recorded "
            "does not count."
        ),
    },
    {
        "glyph": "💥",
        "key": "smash",
        "label": "smash / pop",
        "sense": "smashing/popping/bursting",
        "definition": (
            "Smashing, breaking, destroying, popping, or bursting something (a balloon, bubble, "
            "etc.) as the GOAL — not a forbidden side-effect."
        ),
    },
    {
        "glyph": "🤢",
        "key": "gross",
        "label": "gross",
        "sense": "deliberately disgusting (not just messy)",
        "definition": (
            "Deliberately disgusting or revolting — slime, smells, bodily fluids, eating something "
            "gross. Mere mess (paint/food/water) does NOT count unless it is meant to revolt."
        ),
    },
    {
        "glyph": "🙈",
        "key": "hidden",
        "label": "blindfold / hide",
        "sense": "hiding or blindfolded (NOT cringe)",
        "definition": (
            "Physically concealing or hiding an object or person from view, OR doing something "
            "blindfolded / unable to see. NOT general secrecy or stealth, pure guessing, "
            "embarrassment, or 'can't look'."
        ),
    },
    {
        "glyph": "🤫",
        "key": "silent",
        "label": "silent",
        "sense": "silence (NOT 'a secret')",
        "definition": (
            "The contestant is required to stay silent or make no sound — no noise, no talking. "
            "Making, disguising, or identifying a noise is NOT this motif, nor is keeping a secret."
        ),
    },
    {
        "glyph": "⚖️",
        "key": "weigh",
        "label": "weigh / balance",
        "sense": "physical weight/weighing (NOT justice)",
        "definition": (
            "Literal physical weight or balancing weight — weighing things, hitting an exact weight, "
            "making something as heavy/light as possible, balancing on scales. Strength, sturdiness, "
            "or size alone do NOT count, nor does justice/fairness."
        ),
    },
    {
        "glyph": "🎨",
        "key": "art",
        "label": "art / draw",
        "sense": "drawing/painting/art",
        "definition": "Drawing, painting, sculpting, or making a visual artwork.",
    },
    {
        "glyph": "🎯",
        "key": "throwing",
        "label": "throwing / aim",
        "sense": "throwing/aiming at a target (NOT 'goal achieved')",
        "definition": (
            "Physically throwing, launching, flinging, or rolling a projectile (ball, dart, object) "
            "into a container, at a physical target, or for distance — a real object must travel "
            "through the air or across a surface. Word/guessing/prediction tasks, pulling, and "
            "merely having a goal to 'hit' do NOT count."
        ),
    },
]

MOTIF_KEYS = [m["key"] for m in MOTIFS]
GLYPH_TO_KEY = {m["glyph"]: m["key"] for m in MOTIFS}

# Metaphor-prone glyphs: 🎯 goal/target, 💥 pop/wow, 🤫 secret/shush, 🙈 cringe/can't-look.
# For these, if a gist shows the glyph but the classifier didn't tag the motif, the appearance is
# almost always metaphorical — so STRIP the glyph from the gist rather than force-add the tag.
# (Concrete glyphs like 🥚/👕 do the opposite: a literal appearance the classifier missed is added.)
STRIP_POLICY_KEYS = {"throwing", "smash", "silent", "hidden"}
