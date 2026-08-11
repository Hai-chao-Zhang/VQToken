#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$REPO_ROOT"

: "${DATA_YAML:?Set DATA_YAML to a prepared LLaVA training-mixture YAML file}"
: "${IMAGE_FOLDER:?Set IMAGE_FOLDER to the image dataset root}"
: "${VIDEO_FOLDER:?Set VIDEO_FOLDER to the video dataset root}"

for required_path in "$DATA_YAML" "$IMAGE_FOLDER" "$VIDEO_FOLDER"; do
    if [[ ! -e "$required_path" ]]; then
        printf 'Required path does not exist: %s\n' "$required_path" >&2
        exit 2
    fi
done

PRETRAINED_MODEL=${PRETRAINED_MODEL:-lmms-lab/llava-onevision-qwen2-0.5b-ov}
VISION_MODEL=${VISION_MODEL:-google/siglip-so400m-patch14-384}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
MASTER_PORT=${MASTER_PORT:-30017}
RUN_NAME=${RUN_NAME:-vqtoken-onevision-qwen2-0.5b}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/outputs/$RUN_NAME}
LOG_DIR=${LOG_DIR:-$REPO_ROOT/trlogs}
REPORT_TO=${REPORT_TO:-none}
TORCH_COMPILE=${TORCH_COMPILE:-false}
USE_EMBEDDED_VISION=${USE_EMBEDDED_VISION:-true}

IFS=',' read -r -a GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#GPU_LIST[@]}
if (( NUM_GPUS < 1 )); then
    printf 'CUDA_VISIBLE_DEVICES must name at least one GPU\n' >&2
    exit 2
fi

command -v deepspeed >/dev/null || {
    printf 'deepspeed is not installed; run: pip install -e ".[train]"\n' >&2
    exit 2
}

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}

printf 'Model: %s\nOutput: %s\nGPUs: %s\n' "$PRETRAINED_MODEL" "$OUTPUT_DIR" "$CUDA_VISIBLE_DEVICES"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" deepspeed \
    --master_port "$MASTER_PORT" \
    --num_gpus "$NUM_GPUS" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path "$PRETRAINED_MODEL" \
    --version qwen_1_5 \
    --data_path "$DATA_YAML" \
    --image_folder "$IMAGE_FOLDER" \
    --video_folder "$VIDEO_FOLDER" \
    --use_vqtoken true \
    --use_embedded_vision "$USE_EMBEDDED_VISION" \
    --vqtoken_mode centroids \
    --vqtoken_selection_method fixed \
    --vqtoken_min_clusters 12 \
    --vqtoken_max_clusters 32 \
    --mm_tunable_parts "mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr 2e-6 \
    --vision_tower "$VISION_MODEL" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end false \
    --mm_use_im_patch_token false \
    --group_by_modality_length true \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --mm_newline_position one_token \
    --attn_implementation sdpa \
    --bf16 true \
    --run_name "$RUN_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 true \
    --model_max_length 32768 \
    --gradient_checkpointing true \
    --dataloader_num_workers 4 \
    --lazy_preprocess true \
    --report_to "$REPORT_TO" \
    --torch_compile "$TORCH_COMPILE" \
    --dataloader_drop_last true \
    --frames_upbound 32 \
    2>&1 | tee "$LOG_DIR/$RUN_NAME.log"
