export HF_HOME="your huggingface cache dir"
export OPENAI_API_KEY="your OPENAI_API_KEY here"
export HF_TOKEN="your huggingface key here"
export HF_HUB_ENABLE_HF_TRANSFER=1

export NCCL_P2P_DISABLE="1" 
export NCCL_IB_DISABLE="1"

PRETRAIN=haichaozhang/VQ-Token-llava-ov-0.5b


# run vqtoken
CUDA_VISIBLE_DEVICES=2 accelerate launch --num_processes=1 --main_process_port 29509 \
-m lmms_eval \
--model llava_onevision_vqtoken \
--model_args pretrained=$PRETRAIN,conv_template=qwen_1_5,model_name=llava_qwen \
--tasks activitynetqa --batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs_new/ 

# baseline - llava-onevision
# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --main_process_port 29509 \
# -m lmms_eval \
# --model llava_onevision \
# --model_args pretrained=$PRETRAIN,conv_template=qwen_1_5,model_name=llava_qwen \
# --tasks nextqa_mc_test --batch_size 1 \
# --log_samples \
# --log_samples_suffix llava_onevision \
# --output_path ./logs_new/  