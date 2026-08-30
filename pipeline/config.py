"""Defect class taxonomy for the YOLO classification pipeline.

The raw dataset in data/set01 and data/set02 has no bounding-box annotations:
each photo is a single high-res macro shot of an oak veneer sample, named
after the defect it shows (e.g. dub_vypadavy_suk = falling-out knot). Per the
project decision, the image filename IS the label, so this is treated as a
whole-image classification problem rather than object detection.

Only defects that appear as a spatially localized, discrete blemish are
included here (a knot, a chip-out, a rot patch) -- these are the ones a
photo-classification model can plausibly key on. Classes describing a
whole-panel characteristic (a grain/figure pattern, an overall color cast, a
sapwood zone) are excluded: nothing in the frame localizes them, so a
classifier would just be learning incidental lighting/crop differences.

INCLUDED_CLASSES maps a short class name -> list of filename prefixes (after
stripping the "dub_" species tag and any trailing set/variant suffix like
"_02", "_velke") that belong to it.
"""

INCLUDED_CLASSES: dict[str, list[str]] = {
    "vypadavy_suk": ["vypadavy_suk"],
    "zarostle_suky": ["zarostle_suky"],
    "cerny_soucek_drevokazny_hmyz": ["cerny_soucek_+_drevokazny_hmyz"],
    "vylomeni": ["vylomeni"],
    "hniloba": ["hniloba"],
}

# Veneer grain/figure-cut types -- whole-panel patterns, classified separately
# by pipeline/grain_pattern.py using classical texture/orientation descriptors
# rather than YOLO (see that module's docstring for why).
GRAIN_PATTERN_CLASSES: dict[str, list[str]] = {
    "fladrova_dyha": ["fladrova_dyha"],
    "polofladrova_dyha": ["polofladrova_dyha"],
    "rovnoleta_dyha": ["rovnoleta_dyha"],
    "podelna_neodlupciva_zrcatka": ["podelna_neodlupciva_zrcatka"],
}

# Documented for the README / future re-evaluation, not used by the code.
EXCLUDED_CLASSES: dict[str, str] = {
    "fladrova_dyha": "whole-panel grain/figure pattern, not a localized defect",
    "polofladrova_dyha": "whole-panel grain/figure pattern, not a localized defect",
    "rovnoleta_dyha": "whole-panel grain/figure pattern, not a localized defect",
    "podelna_neodlupciva_zrcatka": "whole-panel grain/figure pattern, not a localized defect",
    "barevny_rozdil_stredni": "whole-panel color characteristic, no localized region",
    "rozbarvenost": "color mismatch across a seam/whole panel, no localized region",
    "bel": "sapwood is a broad zone/edge characteristic, not a discrete blemish",
    "sval": "whole-panel texture characteristic, no visible localized mark",
    "zabeh": "grain-runout streak spanning the panel, no visible localized mark",
}

# Suffixes stripped from a stem before matching it against the prefixes above.
KNOWN_VARIANT_SUFFIXES: list[str] = [
    "_1", "_2", "_3", "_01", "_02", "_03", "_velke",
]

SPECIES_PREFIX = "dub_"

RAW_DATA_DIRS: list[str] = ["data/set01", "data/set02"]

DATASET_DIR = "dataset"
VAL_HOLDOUT_PER_CLASS = 1  # images held out for val when a class has enough unique samples
MIN_UNIQUE_FOR_VAL_SPLIT = 3  # classes with fewer unique images than this train on everything

RESIZE_MAX_SIDE = 1280  # longest side in px after prepare_dataset preprocessing
JPEG_QUALITY = 90

# Real YOLO *detection* (a box drawn around the defect), once
# annotations/boxes.json has been hand-annotated via `python -m
# pipeline.annotate` -- see pipeline/prepare_detect_dataset.py.
#
# Deliberately NOT just sorted(INCLUDED_CLASSES): detection is finer-grained
# than the classifier's whole-image buckets. INCLUDED_CLASSES lumps
# "cerny_soucek_drevokazny_hmyz" (black knot + wood-boring insect damage)
# into one class because a single photo shows both and classification can
# only assign one whole-image label -- but they're two different defects
# that don't always co-occur, and the insect damage typically shows up as
# several separate holes in one photo, not one. Detection can and should
# tell them apart and box each occurrence separately, so they're split here.
DETECT_DATASET_DIR = "detect_dataset"
DETECT_CLASS_NAMES = [
    "cerny_soucek",
    "drevokazny_hmyz",
    "hniloba",
    "vylomeni",
    "vypadavy_suk",
    "zarostle_suky",
]

# Text prompts for pipeline/infer_zeroshot.py -- a pretrained open-vocabulary
# detector (YOLO-World) that has never seen a wood photo, given plain-English
# phrases for the 5 localized defect classes instead of trained box labels.
# See that module's docstring for why this is worth trying (and its honest
# result on this dataset).
ZEROSHOT_PROMPTS: list[str] = [
    "wood knot",
    "knot hole in wood",
    "insect hole in wood",
    "wood rot",
    "chipped out wood",
    "crack in wood",
]
