# Revision experiments (Access-2026-34643)

Everything the revised manuscript reports is produced by `run_all.sh`. It writes
`../latex/results_macros.tex` and `../latex/Figures/revised/*`; rebuild the paper
with `../latex/build.sh` and the response letter with `pdflatex` in `../response/`.
Values that have not been measured print as a red **??**.

## What changed versus the original code

| File | Change |
|---|---|
| `ultralytics_ext/segment_msem_pr.py` | **Bug fix:** in training mode Segment26 returns a dict, which `_find_proto()` did not search, so the PR-Head was skipped during training and ran untrained at inference. New `_refine_train_preds()` refines `one2many` prototypes (and gives `one2one` the detached copy, as Segment26 does). `pr_aux` now stores detached tensors (graph tensors break the EMA deepcopy). |
| `ultralytics_ext/segment_msem_pr.py` | Ablation heads: `PlainSegment26` (width-matched baseline), `MSEMOnlySegment26`, `PROnlySegment26`. Old checkpoints still load. |
| `patch_ultralytics.py` | Automates `docs/MANUAL_PATCH.md` (adds the custom heads to `parse_model`). Idempotent and leaves a `.bak` backup. |
| `convert_ceymo.py` | Official CeyMo → YOLO-seg polygons; official test split unchanged; scene-disjoint, class-stratified validation split. |
| `evaluate.py` | One protocol for all models: Box/Mask P, R, mAP; mIoU; Boundary IoU (Cheng et al.) at d = 3 px and 2 % of the diagonal; FP16 batch-1 FPS. |

## Recipe fix (November 2026): direction-aware flip

The original recipe mirrored images (`fliplr=0.5`, and CopyPaste in `flip` mode) without
changing labels, so a mirrored Left Arrow was still labelled LA (same for SLA/SRA). That is
label noise on exactly the weakest classes (test IoU: LA 35, SRA 29). `ultralytics_ext/flip_swap.py`
now swaps LA<->RA and SLA<->SRA whenever an instance is mirrored; `train_variant.py` runs a
self-test at start and stores the recipe in `runs_<tag>/<run>/recipe.json`.

| Setting | Meaning |
|---|---|
| `FLIP=swap` (default) | mirror and swap direction labels (the fix) |
| `FLIP=off` | no mirroring at all |
| `FLIP=plain` | original submission behaviour (for reference only) |
| `IMGSZ=640` / `1280` | training and evaluation size; 1280 keeps thin markings visible (try `BATCH=8`) |

Each recipe writes to its own `runs_flip<FLIP>_<IMGSZ>/` and `results_flip<FLIP>_<IMGSZ>/`,
so earlier runs are never mixed in. `per_class_report.py <results dir> [...]` prints a summary.

**Quick check first** (one run, about 1/10 of the full cost):

```bash
VARIANTS=baseline SEEDS=0 TABLES=0 ./run_all.sh                 # flip fix at 640
VARIANTS=baseline SEEDS=0 TABLES=0 IMGSZ=1280 BATCH=8 ./run_all.sh   # flip fix at 1280
python per_class_report.py results results_flipswap_640 results_flipswap_1280
```

Compare LA, RA, SLA, SRA and mIoU against the old `results/` (baseline: mIoU 67.5, LA 38.9, SRA 30.5).
Then run the full study with the better setting, e.g. `IMGSZ=1280 BATCH=8 ./run_all.sh`.

## Run (GPU workstation)

```bash
pip install ultralytics==8.4.90 pyyaml opencv-python matplotlib
cd experiments
CEYMO="/path/to/ceymo dataset" SEEDS="0 1 2" DEVICE=0 BATCH=16 ./run_all.sh
cd ../latex && ./build.sh          # road_mark_highlighted.pdf + road_mark_clean.pdf
cd ../response && pdflatex response_to_reviewers.tex
```

That is 5 variants × 3 seeds × 150 epochs on 1,770 images. If `/dev/shm` is small,
add `--workers 0` in `run_all.sh`. If you change BATCH, the seeds or the GPU, update
Section IV-B and Table 2 of the paper.

`yolo26s-seg.pt` (COCO-pretrained) must be in this folder, or Ultralytics must be
able to download it.

## Verified on this Mac (CPU) before hand-over

- The five configs build. The full model has 16.78 M params and 83.3 GFLOPs, and the original `run/best.pt` loads into it with every key matching.
- 1–2-epoch smoke training of all five variants works. PR-Head BatchNorm statistics and weights now change during training (they stayed at their initial values in the original checkpoint).
- `convert_ceymo.py` on the real CeyMo release: 1,770 / 316 / 788 images, 566 / 100 / 264 scenes, zero scene overlap between any pair of splits.
- `evaluate.py` and `make_tables.py` run end to end.

## Released files

| Path | Contents |
|---|---|
| `splits/{train,val,test}_seed0.txt` | Exact image IDs (CeyMo file stems) of the leakage-free splits used in the paper: 1,770 / 316 / 788 images. `convert_ceymo.py --seed 0` reproduces them. |
| `splits/train_dropped_scene_overlap_with_test.txt` | The 13 official training images removed because their scene also occurs in the official test split. |
| `split_stats_seed0.json` | Per-split image, scene and per-class instance counts, and the scene-overlap check (all zero). |
| `results_flipswap_640/*.json` | Evaluation output of every run (5 variants x 3 seeds, val and test). |
| `runs_flipswap_640/<run>/` | Training log (`results.csv`), full Ultralytics arguments (`args.yaml`) and recipe (`recipe.json`) of each run. Trained weights (`best.pt`) are not included because of their size. |
| `notebooks/` | Kaggle notebooks used to train and evaluate the variants. |

The CeyMo images and annotations are not redistributed; obtain them from the CeyMo authors.
