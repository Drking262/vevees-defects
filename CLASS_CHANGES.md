# Defect class roadmap — what's needed beyond the Kodytek dataset

Two separate class taxonomies are in play here; this doc maps where they
line up and what still needs work for the classes that get **no help at
all** from the Kodytek/Hugging Face data.

## 1. Our own taxonomy (`pipeline/config.py`)

`DETECT_CLASS_NAMES` plus current annotation coverage
(`annotations/boxes.json`, 5 images total):

| our class | meaning | raw photos | annotated boxes |
|---|---|:---:|:---:|
| `vypadavy_suk` | knot fallen out / knot-hole | 3 | 3 |
| `zarostle_suky` | overgrown / ingrown knot | 2 | 0 |
| `cerny_soucek` | black pith fleck / black knot | 1 | 1 |
| `drevokazny_hmyz` | wood-boring insect damage | 1† | 0 |
| `hniloba` | rot | 2 | 1 |
| `vylomeni` | breakout / chip-out (veneer peeling) | 2 | 0 |

† one raw photo shows both `cerny_soucek` and `drevokazny_hmyz`; only the
`cerny_soucek` box is drawn so far.

**Headline finding**: 5 annotated images across 6 classes isn't enough to
train a detector on, independent of Kodytek.

## 2. Kodytek's taxonomy (`iluvvatar/wood_surface_defects` on HF)

Ten classes (box counts from `YOLOV8-CDC/wood_defects_dataset_analysis.ipynb`):
`Live_Knot` (21,224), `Dead_Knot` (11,985), `resin` (3,455),
`knot_with_crack` (2,276), `Crack` (2,169), `Marrow` (1,181), `Quartzity`
(1,075), `Knot_missing` (503), `Blue_Stain` (96), `overgrown` (10).

Published taxonomy for **sawn timber boards**, a different setup/vocabulary
than our **oak veneer macro shots** -- a domain gap that limits reuse below.

## 3. Class-by-class: what maps, what doesn't

| our class | closest Kodytek class | overlap | action needed |
|---|---|---|---|
| `zarostle_suky` | `overgrown` | Weak -- Kodytek's rarest class (10 boxes), barely anything to transfer | Need our own examples; don't rely on Kodytek pretraining |
| `vypadavy_suk` | `Knot_missing` | Moderate -- same concept, 503 usable boxes | Worth pretraining on, then verify it generalizes across the domain gap |
| `cerny_soucek` | none (`Dead_Knot` visually different) | None | Novel class; needs its own data + annotation |
| `drevokazny_hmyz` | none | None | Biggest gap -- zero Kodytek or local coverage |
| `hniloba` | none | None | No Kodytek rot class; only 1 annotated box today |
| `vylomeni` | none | None | Veneer-specific process defect, not in sawn-board data; annotate the 2 existing photos, then collect more |

## 4. What this actually changes

1. Kodytek pretraining covers a different label set on a different
   substrate -- only `vypadavy_suk` is worth borrowing from.
2. Annotate zero-support classes first: `drevokazny_hmyz` and `vylomeni`
   have no boxes and no fallback source; `zarostle_suky` close behind.
3. `annotations/boxes.json` needs far more coverage before
   `pipeline.prepare_detect_dataset` is worth training on.
4. If Kodytek pretraining is used, scope it narrowly -- warm-starting
   "knot-shaped blob" features via `vypadavy_suk`/`zarostle_suky`, not as
   labeled examples for the other four classes.
5. Use `pipeline/infer_zeroshot.py` as an annotation accelerator: its
   YOLO-World prompts (`"insect hole in wood"`, `"wood rot"`, `"chipped out
   wood"`) can generate candidate boxes for a human to verify.

## Open questions

- Confirm Kodytek's actual photography setup (board vs. veneer) -- if the
  gap is large, even `vypadavy_suk` transfer may not be worth it.
- Decide whether to collect more raw photos before annotating --
  `hniloba`/`vylomeni`/`drevokazny_hmyz` each have only 1-2 source photos,
  capping how much annotation can help regardless of effort.
