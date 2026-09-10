"""Defect class taxonomy for the YOLO classification pipeline.

data/set01 and data/set02 have no box annotations -- filename IS the label
(e.g. dub_vypadavy_suk = falling-out knot), so this is whole-image
classification, not detection.

Only spatially localized, discrete blemishes are included (knot, chip-out,
rot patch) -- whole-panel characteristics (grain pattern, color cast,
sapwood zone) are excluded since nothing in the frame localizes them.

INCLUDED_CLASSES maps a class name -> filename prefixes (after stripping the
"dub_" species tag and variant suffixes like "_02", "_velke").
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

# Real YOLO detection classes, once annotations/boxes.json is hand-annotated
# (see pipeline/prepare_detect_dataset.py). Finer-grained than
# INCLUDED_CLASSES: detection splits "cerny_soucek_drevokazny_hmyz" into two
# classes since they don't always co-occur and insect damage can show up as
# multiple separate holes in one photo.
DETECT_DATASET_DIR = "detect_dataset"
DETECT_CLASS_NAMES = [
    "cerny_soucek",
    "drevokazny_hmyz",
    "hniloba",
    "vylomeni",
    "vypadavy_suk",
    "zarostle_suky",
]

# Plain-English prompts for pipeline/infer_zeroshot.py's pretrained
# open-vocabulary detector -- see that module's docstring for results.
ZEROSHOT_PROMPTS: list[str] = [
    "wood knot",
    "knot hole in wood",
    "insect hole in wood",
    "wood rot",
    "chipped out wood",
    "crack in wood",
]
