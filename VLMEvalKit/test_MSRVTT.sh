export HF_HOME=your_cache_dir
export OPENAI_API_KEY=sk-your-openai-key
export HF_TOKEN=hf_-you-huggingface-key
export HF_HUB_ENABLE_HF_TRANSFER=1

# IDEFICS-80B-Instruct on MMBench_DEV_EN, MME, and SEEDBench_IMG, Inference only
CUDA_VISIBLE_DEVICES=0 python run.py --data MSRVTT_MC --model llava_onevision_qwen2_0.5b_ov --verbose 


# longvideobench_dataset = {
#     'LongVideoBench_8frame': partial(LongVideoBench, dataset='LongVideoBench', nframe=8),
#     'LongVideoBench_8frame_subs': partial(LongVideoBench, dataset='LongVideoBench', nframe=8, use_subtitle=True),
#     'LongVideoBench_64frame': partial(LongVideoBench, dataset='LongVideoBench', nframe=64),
#     'LongVideoBench_1fps': partial(LongVideoBench, dataset='LongVideoBench', fps=1.0),
#     'LongVideoBench_0.5fps': partial(LongVideoBench, dataset='LongVideoBench', fps=0.5),
#     'LongVideoBench_0.5fps_subs': partial(LongVideoBench, dataset='LongVideoBench', fps=0.5, use_subtitle=True)
# }


# llava_onevision_qwen2_0.5b_ov