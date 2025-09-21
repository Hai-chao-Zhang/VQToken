import torch
import torch.nn.functional as F
import time
import numpy as np
from decord import VideoReader, cpu
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
import copy
import os
import warnings

warnings.filterwarnings("ignore")

# Set device
device = "cuda:3" if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(device)

# Load the LLaVA-OneVision Model
pretrained = "haichaozhang/VQ-Token-llava-ov-0.5b"
model_name = "llava_qwen"
device_map = "auto"
llava_model_args = {"multimodal": True}

# Load model, tokenizer, and image processor
tokenizer, model, image_processor, max_length = load_pretrained_model(
    pretrained, None, model_name, device_map=device_map, attn_implementation="sdpa", **llava_model_args
)
model.eval()

# Function to extract frames from video
def load_video(video_path, max_frames_num):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    frame_idx = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int).tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames  # Shape: (frames, height, width, channels)

# Load and process video
video_path = "sample/P15_cereals.avi"
video_frames = load_video(video_path, 16)  # Load 16 frames
print(f"Video Frames Shape: {video_frames.shape}")  # Expecting (16, 1024, 576, 3)

# Convert frames to tensor and preprocess
image_tensors = []
frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
image_tensors.append(frames)

# Prepare conversation input
conv_template = "qwen_1_5"
question = f"{DEFAULT_IMAGE_TOKEN}\nDescribe what's happening in this video."

conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [frame.shape[:2] for frame in video_frames]  # Extract (height, width) for each frame

# ✅ Ensure attention mask is properly set to avoid errors
attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

# ===== Measure Execution Time =====
torch.cuda.synchronize()
start_time = time.time()

# GPU-based precise timing
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()

# Generate response with a valid attention mask
with torch.no_grad():
    cont = model.generate(
        input_ids,
        images=image_tensors,
        image_sizes=image_sizes,
        attention_mask=attention_mask,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
        modalities=["video"],
    )

torch.cuda.synchronize()
end_time = time.time()
end_event.record()
torch.cuda.synchronize()

# Time measurements
elapsed_time = start_event.elapsed_time(end_event) / 1000  # Convert from ms to seconds
cpu_time = end_time - start_time

# Decode text outputs
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
print("\nGenerated Text Output:\n", text_outputs[0])

# ===== Measure Memory Usage =====
max_memory_used = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # Convert to MB
current_memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # Convert to MB

# ===== Accurate FLOPs Calculation Using PyTorch Profiler =====
def profile_flops(model, input_ids, image_tensors, image_sizes, attention_mask):
    """Measure FLOPs using PyTorch Profiler"""

    flops = 0

    with torch.no_grad():
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_flops=True
        ) as prof:
            _ = model.generate(
                input_ids,
                images=image_tensors,
                image_sizes=image_sizes,
                attention_mask=attention_mask,
                do_sample=False,
                temperature=0,
                max_new_tokens=4096,
                modalities=["video"],
            )

    for evt in prof.key_averages():
        if evt.flops is not None:
            flops += evt.flops  # Accumulate FLOPs

    return flops / (10**12)  # Convert to TFLOPs

# Measure FLOPs
try:
    flop_count = profile_flops(model, input_ids, image_tensors, image_sizes, attention_mask)
except Exception as e:
    print("\n⚠️ Warning: FLOP calculation failed.")
    flop_count = None

# ===== Print Summary =====
print("\n===== Performance Metrics =====")
print(f"Execution Time (CPU): {cpu_time:.4f} sec")
print(f"Execution Time (GPU Event Timing): {elapsed_time:.4f} sec")
# print(f"Max Memory Used (GPU): {max_memory_used:.2f} MB")
# print(f"Current Memory Allocated (GPU): {current_memory_allocated:.2f} MB")
if flop_count is not None:
    print(f"Estimated FLOPs: {flop_count:.2f} TFLOPs")


# import torch
# import time
# from decord import VideoReader, cpu
# from llava.model.builder import load_pretrained_model
# from llava.mm_utils import tokenizer_image_token
# from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
# from llava.conversation import conv_templates
# import copy
# import warnings
# import numpy as np

# warnings.filterwarnings("ignore")

# # Set device
# device = "cuda:4" if torch.cuda.is_available() else "cpu"
# torch.cuda.set_device(device)

# # Load the LLaVA-OneVision Model
# pretrained = "haichaozhang/VQ-Token-llava-ov-0.5b"
# model_name = "llava_qwen"
# device_map = "auto"
# llava_model_args = {"multimodal": True}

# tokenizer, model, image_processor, max_length = load_pretrained_model(
#     pretrained, None, model_name, device_map=device_map, attn_implementation="sdpa", **llava_model_args
# )
# model.eval()

# def load_video(video_path, max_frames_num):
#     vr = VideoReader(video_path, ctx=cpu(0))
#     total_frame_num = len(vr)
#     frame_idx = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int).tolist()
#     spare_frames = vr.get_batch(frame_idx).asnumpy()
#     return spare_frames

# # Load video
# video_path = "sample/P15_cereals.avi"
# video_frames = load_video(video_path, 16)

# # Convert frames to tensor and preprocess
# image_tensors = []
# frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
# image_tensors.append(frames)

# # Prepare input
# conv_template = "qwen_1_5"
# question = f"{DEFAULT_IMAGE_TOKEN}\nDescribe what's happening in this video."

# conv = copy.deepcopy(conv_templates[conv_template])
# conv.append_message(conv.roles[0], question)
# conv.append_message(conv.roles[1], None)
# prompt_question = conv.get_prompt()

# input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
# image_sizes = [frame.shape[:2] for frame in video_frames]

# attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

# # 🔥 Reset Memory Stats Before Execution
# torch.cuda.empty_cache()
# torch.cuda.reset_peak_memory_stats()

# # Start Timing
# torch.cuda.synchronize()
# start_time = time.time()
# start_event = torch.cuda.Event(enable_timing=True)
# end_event = torch.cuda.Event(enable_timing=True)
# start_event.record()

# # Generate response
# with torch.no_grad():
#     cont = model.generate(
#         input_ids,
#         images=image_tensors,
#         image_sizes=image_sizes,
#         attention_mask=attention_mask,
#         do_sample=False,
#         temperature=0,
#         max_new_tokens=4096,
#         modalities=["video"],
#     )

# torch.cuda.synchronize()
# end_time = time.time()
# end_event.record()
# torch.cuda.synchronize()

# elapsed_time = start_event.elapsed_time(end_event) / 1000  # Convert ms to seconds
# cpu_time = end_time - start_time

# # Decode output
# text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
# print("\nGenerated Text Output:\n", text_outputs[0])

# # 🔥 Track GPU Memory Consumption
# max_memory_used = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB
# current_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)  # MB
# total_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)  # MB

# print("\n===== Performance Metrics =====")
# print(f"Execution Time (CPU): {cpu_time:.4f} sec")
# print(f"Execution Time (GPU Event Timing): {elapsed_time:.4f} sec")
# print(f"Max Memory Used (GPU): {max_memory_used:.2f} MB")
# print(f"Current Memory Allocated (GPU): {current_memory:.2f} MB")
# print(f"Total Reserved Memory (GPU): {total_reserved:.2f} MB")

# # 🔥 Deep Memory Breakdown with PyTorch Profiler
# with torch.profiler.profile(
#     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
#     profile_memory=True,
#     record_shapes=True
# ) as prof:
#     _ = model.generate(
#         input_ids,
#         images=image_tensors,
#         image_sizes=image_sizes,
#         attention_mask=attention_mask,
#         do_sample=False,
#         temperature=0,
#         max_new_tokens=4096,
#         modalities=["video"],
#     )

# print("\nDetailed Memory Report:")
# print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
