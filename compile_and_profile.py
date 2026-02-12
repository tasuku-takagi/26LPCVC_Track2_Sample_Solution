import qai_hub
import onnx
import os
import sys

# --- Configuration ---
ONNX_DIR = "exported_onnx"
VIDEO_ONNX_NAME = "r2plus1dQEVD"   # change if you named it differently
DEVICE_NAME = "Dragonwing IQ-9075 EVK"

# Must match your export dummy input
BATCH = 1
C = 3
T = 8           # frames (8 if that’s what you exported)
H = 112         # common for r2plus1d_18
W = 112
# ---------------------

def run_profile(model, device):
    profile_job = qai_hub.submit_profile_job(
        model=model,
        device=device,
        options="--max_profiler_iterations 100"
    )
    return profile_job.job_id

def compile_model(model, device, input_specs):
    compile_job = qai_hub.submit_compile_job(
        model=model,
        device=device,
        input_specs=input_specs,
        options="--target_runtime onnx"
    )
    return compile_job.job_id


VIDEO_ONNX_PATH = os.path.join(ONNX_DIR, VIDEO_ONNX_NAME)

if not os.path.exists(VIDEO_ONNX_PATH):
    print(f"Error: '{VIDEO_ONNX_PATH}' not found. Run export_onnx.py first.")
    sys.exit(1)

print(f"Loading ONNX video model from {VIDEO_ONNX_PATH}...")
onnx_video_model = onnx.load(VIDEO_ONNX_PATH)

try:
    onnx.checker.check_model(onnx_video_model)
    print("Video ONNX model is valid ✅")
except onnx.checker.ValidationError as e:
    print("Video ONNX model validation failed ❌")
    print(e)
    sys.exit(1)

device = qai_hub.Device(DEVICE_NAME)

# IMPORTANT: input name must match torch.onnx.export(input_names=[...])
# If you used input_names=["video"] in export, keep "video" here.
input_specs = {
    "video": (BATCH, C, T, H, W)  # float32 by default
}

print("\nSubmitting compilation job to QAI Hub...")
compile_id = compile_model(
    model=onnx_video_model,
    device=device,
    input_specs=input_specs
)
print(f"Compilation job ID: {compile_id}")

print("\nSubmitting profiling job to QAI Hub...")
target_model = qai_hub.get_job(compile_id).get_target_model()
profile_id = run_profile(target_model, device)
print(f"Profiling job ID: {profile_id}")
print("Done.")
