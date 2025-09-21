import torch
import torch.nn.functional as F
import time, os
import numpy as np
from decord import VideoReader, cpu
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
import copy
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
    return vr.get_batch(frame_idx).asnumpy()  # Shape: (frames, height, width, channels)

# Load and process video
videofolder_path = "sample/"
for video in os.listdir(videofolder_path):
    video_path = os.path.join(videofolder_path, video)
    video_frames = load_video(video_path, 16)  # Load 16 frames

    # Convert frames to tensor and preprocess
    frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
    image_tensors = [frames]

    # Prepare conversation input
    conv_template = "qwen_1_5"
    question = f"{DEFAULT_IMAGE_TOKEN}\nDescribe what's happening in this video."

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [frame.shape[:2] for frame in video_frames]  # Extract (height, width) for each frame

    # Ensure attention mask is properly set
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    # Measure Execution Time
    torch.cuda.synchronize()
    start_time = time.time()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    # Generate response
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
            vis=True
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

    # Print Performance Summary
    print("\n===== Performance Metrics =====")
    print(f"Execution Time (CPU): {cpu_time:.4f} sec")
    print(f"Execution Time (GPU Event Timing): {elapsed_time:.4f} sec")
