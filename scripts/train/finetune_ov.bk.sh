export NUM_GPUS=1
export NNODES=1
export RANK=0
export ADDR=172.17.139.154  # or 'localhost'
export PORT=29500      # or any free port number

# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=docker0 # ens8f0
# export NCCL_P2P_DISABLE=1
# export NCCL_DEBUG=INFO

export WANDB_API_KEY="your_wandb_key"
export WANDB_ENTITY="youraandb_entity"
export WANDB_PROJECT=llavanext
export WANDB_MODE=online
export PYTHONWARNINGS="ignore"

# export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512' #

export HF_HOME="your huggingface cache dir"
export OPENAI_API_KEY="your OPENAI_API_KEY here"
export HF_TOKEN="your huggingface key here"
export HF_HUB_ENABLE_HF_TRANSFER=1

export PYTHONPATH="${PYTHONPATH}:./tmp/LLaVa-Video/LLaVA-NeXT"



IMAGE_FOLDER="/your_dataset_dir/llava_ov/images"
VIDEO_FOLDER="/your_dataset_dir/LLaVA-Video-178K/video/"
DATA_YAML="./tmp/LLaVa-Video/LLaVA-NeXT/scripts/video/train/datasets.yaml" # e.g exp.yaml


LLM_VERSION="Qwen/Qwen2-7B-Instruct" 
# for 7b model we recommend bs=1, accum=2, 16 nodes, 128 gpus, lr=1e-5, warmup=0.03
# for 72b model we recommend bs=1, accum=1, 32 nodes, 256 gpus, lr=1e-5, warmup=0.03
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

BASE_RUN_NAME="llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

############### Finetune ################

# Stage 2
PROMPT_VERSION="qwen_1_5"
RUN_NAME="llava-onevision-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-ov_stage_am9" 
# PREV_STAGE_CHECKPOINT="/mnt/bn/vl-research/checkpoints/onevision/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-mid_to_final_next_3m_am9_july14" # replace it with your last checkpoint training from single image collection
PREV_STAGE_CHECKPOINT="./tmp/llm/llava-onevision-qwen2-0.5b-ov"
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${RUN_NAME}"



# --data_path /mnt/bn/vl-research/workspace/boli01/projects/LLaVA_Next/scripts/i18n/scale_llms/next_ov_stage_july21.yaml \
# --image_folder /mnt/bn/vl-research/data/llava_data \
# --video_folder /mnt/bn/vl-research/data/llava_video \

# ACCELERATE_CPU_AFFINITY=1 torchrun --nnodes="${NNODES}" --node_rank="${RANK}" --nproc_per_node="${NUM_GPUS}"   --master_addr="${ADDR}" --master_port="${PORT}" 

# deepspeed --master_port 30000 --include localhost:4,5,6,7\
ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node=8 --master_port 30000 \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path $DATA_YAML \
    --image_folder $IMAGE_FOLDER \
    --video_folder $VIDEO_FOLDER \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir ./tmp/LLaVa-Video/ckpt/$RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 32
exit 0;

# You can delete the sdpa attn_implementation if you want to use flash attn
