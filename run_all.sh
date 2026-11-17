#!/usr/bin/env bash
# End-to-end revision experiments: convert CeyMo -> train 5 variants -> evaluate on val and test -> LaTeX macros.
#
#   CEYMO=/path/to/ceymo SEEDS="0 1 2" DEVICE=0 BATCH=16 WORKERS=2 ./run_all.sh
#
# Recipe switches (new runs and results go to their own folders, so earlier runs are never mixed in):
#   FLIP=swap   mirror images and swap LA<->RA, SLA<->SRA labels (default; fixes the original label noise)
#   FLIP=off    no mirroring;  FLIP=plain  original submission behaviour
#   IMGSZ=640   training/eval image size; 1280 keeps thin markings visible (use BATCH=8 if GPU memory is short)
#   TABLES=0    skip writing ../latex/results_macros.tex and figures (for quick checks / partial runs)
#
# Requires: pip install ultralytics==8.4.90 pyyaml opencv-python  (and yolo26s-seg.pt in this folder or auto-download)
set -euo pipefail
cd "$(dirname "$0")"

CEYMO=${CEYMO:-}
DATA_DIR=${DATA_DIR:-datasets/ceymo_official}
SEEDS=${SEEDS:-"0 1 2"}
DEVICE=${DEVICE:-0}
BATCH=${BATCH:-16}
WORKERS=${WORKERS:-2}   # dataloader workers; use 0 if /dev/shm is small (Docker default 64MB)
VARIANTS=${VARIANTS:-"baseline baseline_p256 msem_only pr_only full"}
FLIP=${FLIP:-swap}
IMGSZ=${IMGSZ:-640}
TAG=${TAG:-flip${FLIP}_${IMGSZ}}
RUNS=${RUNS:-runs_${TAG}}
RESULTS=${RESULTS:-results_${TAG}}
TABLES=${TABLES:-1}

echo "Recipe: FLIP=$FLIP IMGSZ=$IMGSZ BATCH=$BATCH SEEDS='$SEEDS' -> $RUNS/, $RESULTS/"

python patch_ultralytics.py
python make_variant_yamls.py
if [ ! -f "$DATA_DIR/data.yaml" ]; then
  [ -n "$CEYMO" ] || { echo "set CEYMO to the official CeyMo root (contains train/ and test/)"; exit 1; }
  python convert_ceymo.py --ceymo "$CEYMO" --out "$DATA_DIR"
fi

first_seed=$(echo $SEEDS | cut -d' ' -f1)
for seed in $SEEDS; do
  for v in $VARIANTS; do
    w="$RUNS/${v}_s${seed}/weights/best.pt"
    # train_variant.py skips finished runs (<run>/DONE) and resumes interrupted ones from last.pt
    python train_variant.py --variant "$v" --data "$DATA_DIR/data.yaml" --seed "$seed" --batch "$BATCH" \
      --device "$DEVICE" --workers "$WORKERS" --project "$PWD/$RUNS" --imgsz "$IMGSZ" --flip "$FLIP"
    for split in val test; do
      [ -f "$RESULTS/${v}_s${seed}_${split}.json" ] && continue
      fps=""; [ "$split" = test ] && [ "$seed" = "$first_seed" ] && fps="--fps"
      python evaluate.py --weights "$w" --data "$DATA_DIR/data.yaml" --split "$split" --imgsz "$IMGSZ" \
        --variant "${v}_s${seed}" --out "$RESULTS" --device "$DEVICE" --workers "$WORKERS" $fps
    done
  done
done

if [ "$TABLES" = 0 ]; then
  python per_class_report.py "$RESULTS"
  echo "Done (TABLES=0: manuscript not updated)."
  exit 0
fi

python make_tables.py --results "$RESULTS" --stats "$DATA_DIR/split_stats.json" --out ../latex/results_macros.tex

# Figures used by the manuscript (first seed of the full model, official test split)
FIG=../latex/Figures/revised; mkdir -p "$FIG"
U="$RESULTS/ultra/full_s${first_seed}_test"
for f in MaskPR_curve.png MaskF1_curve.png confusion_matrix_normalized.png val_batch0_labels.jpg val_batch0_pred.jpg; do
  cp "$U/$f" "$FIG/$f"
done
python plot_convergence.py "$RUNS/full_s${first_seed}/results.csv" "$FIG/convergence.pdf"
python per_class_report.py "$RESULTS"
echo "Done. Recompile the manuscript: results_macros.tex now holds every number used in the paper."
