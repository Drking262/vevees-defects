# Defect class roadmap — what's needed beyond the Kodytek dataset

This project actually has **two separate class taxonomies** in play, and only
some of the work overlaps between them. This doc lays out where they line up
and — more importantly — what still has to happen for the classes that get
**no help at all** from the Kodytek/Hugging Face data.

## 1. Our own taxonomy (`pipeline/config.py`)

The real detection target, `DETECT_CLASS_NAMES`, plus how much annotated data
each one actually has right now (`annotations/boxes.json`, 5 images total):

| our class          | meaning                              | raw photos | annotated boxes |
|---------------------|---------------------------------------|:---------:|:---------------:|
| `vypadavy_suk`       | knot fallen out / knot-hole           | 3         | 3                |
| `zarostle_suky`      | overgrown / ingrown knot              | 2         | 0                |
| `cerny_soucek`       | black pith fleck / black knot         | 1         | 1                |
| `drevokazny_hmyz`    | wood-boring insect damage             | 1†        | 0                |
| `hniloba`            | rot                                   | 2         | 1                |
| `vylomeni`           | breakout / chip-out (veneer peeling)  | 2         | 0                |

† the one `cerny_soucek_+_drevokazny_hmyz.jpg` raw photo shows both defects
in one frame; only the `cerny_soucek` box has been drawn so far.

**5 annotated images across 6 classes is not enough to train a detector on,
independent of anything to do with Kodytek** — that's the headline finding.

## 2. The Kodytek dataset's taxonomy (`iluvvatar/wood_surface_defects` on HF)

Ten classes, sized per the EDA in `YOLOV8-CDC/wood_defects_dataset_analysis.ipynb`:
`Live_Knot` (21,224 boxes), `Dead_Knot` (11,985), `resin` (3,455),
`knot_with_crack` (2,276), `Crack` (2,169), `Marrow` (1,181), `Quartzity`
(1,075), `Knot_missing` (503), `Blue_Stain` (96), `overgrown` (10).

This is the published Kodytek wood-surface-defects taxonomy for automated
visual inspection of **sawn timber boards** — a different photography setup
and defect vocabulary than our own **oak veneer macro shots**. That domain
gap matters for how much of it is actually reusable below.

## 3. Class-by-class: what maps, what doesn't

| our class          | closest Kodytek class | overlap quality | action needed |
|---------------------|------------------------|------------------|----------------|
| `zarostle_suky`      | `overgrown`             | Weak — conceptually the same defect, but Kodytek itself only has 10 boxes of it (its rarest class), so there's barely anything to transfer from. | Still need our own annotated examples; don't expect Kodytek pretraining to carry this class. |
| `vypadavy_suk`       | `Knot_missing`          | Moderate — same concept (a knot has fallen out, leaving a hole), and Kodytek has a usable 503 boxes. | Worth trying: pretrain/fine-tune a knot-hole detector on Kodytek's `Knot_missing`, then adapt to our veneer photos. Verify it actually generalizes across the domain gap before trusting it. |
| `cerny_soucek`       | none (`Dead_Knot` is the nearest, but visually different — a fully blackened dead knot vs. a black pith fleck) | None | Genuinely novel class for the detector; needs its own data collection + annotation, no transfer source. |
| `drevokazny_hmyz`    | none                    | None | No Kodytek class covers insect damage at all. Zero annotated examples exist locally either — this is the single biggest gap in the whole taxonomy. |
| `hniloba`            | none                    | None | Kodytek has no decay/rot class. Needs dedicated data collection; only 1 annotated box exists today. |
| `vylomeni`           | none                    | None | A veneer-peeling process defect — not something a sawn-board dataset like Kodytek would ever contain. Annotate the 2 existing raw photos at minimum, then collect more. |

## 4. What this actually changes

1. **Don't plan around "Kodytek pretraining covers wood defects broadly."**
   It covers a specific, different label set on a different substrate. Only
   `vypadavy_suk` has a Kodytek class worth borrowing from in any real sense.
2. **Annotation priority should go to the zero-support classes first** —
   `drevokazny_hmyz` and `vylomeni` currently have *no* annotated boxes at
   all and no fallback data source. `zarostle_suky` is close behind (0 boxes,
   and its Kodytek analog is too thin to help).
3. **`annotations/boxes.json` needs far more coverage** before
   `pipeline.prepare_detect_dataset` produces a dataset worth training on —
   this is true regardless of anything Kodytek-related.
4. **If Kodytek pretraining is used, scope it narrowly**: warm-starting
   general "knot-shaped blob" features via `vypadavy_suk`/`zarostle_suky`,
   not as a source of extra labeled examples for the other four classes.
5. **Use `pipeline/infer_zeroshot.py` as an annotation accelerator.** Its
   YOLO-World prompts already include `"insect hole in wood"`, `"wood rot"`,
   and `"chipped out wood"` — running it over the raw `data/set01`/`data/set02`
   photos to generate candidate boxes (for a human to verify/correct) should
   be faster than annotating the zero-support classes from a blank canvas.

## Open questions

- Confirm the Kodytek dataset's actual photography setup (board surface vs.
  veneer) — if the domain gap is large, even the `vypadavy_suk` transfer may
  not be worth the effort.
- Decide whether to collect more raw photos (`data/set03`?) before
  annotation — `hniloba`, `vylomeni`, and `drevokazny_hmyz` each have only
  1–2 source photos today, which caps how much annotation can help regardless
  of effort spent.
