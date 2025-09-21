import torch
import time
import threading
from decord import VideoReader, cpu
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
import copy
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# Set device
device = "cuda:2" if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(device)

# Load LLaVA-OneVision Model
pretrained = "haichaozhang/VQ-Token-llava-ov-0.5b"
model_name = "llava_qwen"
device_map = "auto"
llava_model_args = {"multimodal": True}

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

# Load video frames
video_path = "sample/P15_cereals.avi"
video_frames = load_video(video_path, 16)
print(f"Video Frames Shape: {video_frames.shape}")  # Expecting (16, 1024, 576, 3)

# Preprocess video frames
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
image_sizes = [frame.shape[:2] for frame in video_frames]

# Ensure valid attention mask
attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

# ===== Track Only Max GPU Memory While Checking Every 0.05s =====
max_memory_used = 0  # Store the max memory

def monitor_gpu_memory(interval=0.05, stop_event=None):
    """Continuously updates max GPU memory usage every 0.05s."""
    global max_memory_used
    while not stop_event.is_set():
        torch.cuda.synchronize()
        current_mem = torch.cuda.memory_allocated(device) / (1024 ** 2)  # Convert to MB
        max_memory_used = max(max_memory_used, current_mem)
        time.sleep(interval)

# Start GPU monitoring thread
stop_event = threading.Event()
monitor_thread = threading.Thread(target=monitor_gpu_memory, args=(0.05, stop_event), daemon=True)
monitor_thread.start()

# ====== Run Model ======
torch.cuda.empty_cache()  # Clear cache before measurement
torch.cuda.reset_peak_memory_stats(device)

# Measure execution time
torch.cuda.synchronize()
start_time = time.time()

# GPU timing events
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
    )

end_event.record()
torch.cuda.synchronize()
end_time = time.time()

# Stop GPU monitoring
stop_event.set()
monitor_thread.join()

# Time measurements
elapsed_time = start_event.elapsed_time(end_event) / 1000  # Convert from ms to sec
cpu_time = end_time - start_time

# Decode output
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
print("\nGenerated Text Output:\n", text_outputs[0])

# Final GPU memory report
print(f"\n🔹 Final Peak GPU Memory Used: {max_memory_used:.2f} MB")
