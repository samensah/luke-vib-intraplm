#!/bin/bash

# ============================================
# CONFIGURATION
# ============================================
# "studio-ousia/luke-large"
# "roberta-base"
MODEL_NAME="roberta-base"
DATASET="tacrev"
EPOCHS=5
NUM_CLASS=42
SEED=64
DATA_PERCENTAGE=100
BETA=0.5

# VIB Layer Configuration
# -2 = Word embedding layer (before encoder)
# -1 = Last encoder layer
#  0 = First encoder layer
#  1 = Second encoder layer
#  2 = Third encoder layer
# ... and so on
VIB_LAYERS=(-2 -1 0 1 2 3 4 5 6 7 8 9 10)



# ============================================
# AUTO-DERIVED PATHS
# ============================================
DATA_DIR="./data/${DATASET}"

# Auto-detect model prefix (rb, rl, lb, ll)
case "$MODEL_NAME" in
    *luke-large*)   MODEL_PREFIX="ll" ;;
    *luke-base*)    MODEL_PREFIX="lb" ;;
    *roberta-large*) MODEL_PREFIX="rl" ;;
    *roberta-base*)  MODEL_PREFIX="rb" ;;
    *)              MODEL_PREFIX="model" ;;
esac



# ============================================
# DISPLAY CONFIGURATION
# ============================================
echo "========================================"
echo "Model: ${MODEL_NAME} (${MODEL_PREFIX})"
echo "Dataset: ${DATASET}"
echo "VIB Layers: ${VIB_LAYERS[@]}"

echo "========================================"

# ============================================
# TRAINING LOOP
# ============================================
for vib_idx in "${VIB_LAYERS[@]}"; do
    echo "Training VIB Layer ${vib_idx}..."

    OUTPUT_DIR="outputs/${MODEL_PREFIX}_${DATASET}/${vib_idx}"
    echo "Output: ${OUTPUT_DIR}"

    [ -n "$RESUME_FROM" ] && echo "Resume: ${RESUME_FROM}"
    # Optional: Set checkpoint path to resume training
    RESUME_FROM=""  # e.g., "outputs/rb_retacred/checkpoint-1500"
    
    CUDA_VISIBLE_DEVICES=0 python src/train.py \
      --data_dir ${DATA_DIR} \
      --model_name_or_path ${MODEL_NAME} \
      --input_format entity_marker_punct \
      --seed ${SEED} \
      --train_batch_size 32 \
      --eval_batch_size 32 \
      --num_train_epochs ${EPOCHS} \
      --eval_steps 500 \
      --num_class ${NUM_CLASS} \
      --output_dir ${OUTPUT_DIR} \
      --beta ${BETA} \
      --vib_layer_idx ${vib_idx} \
      --data_percentage ${DATA_PERCENTAGE} \
      ${RESUME_FROM:+--resume_from_checkpoint "$RESUME_FROM"}
    
    echo "✓ Completed layer ${vib_idx}"
    echo ""
done

echo "All runs completed! Results in: ${OUTPUT_DIR}"