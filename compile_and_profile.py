import qai_hub
import onnx
import os
import sys

# =============================================================================
# USER CONFIGURATION — update these values before running
# =============================================================================

# Path to the directory containing your exported ONNX model.
ONNX_DIR = ""
VIDEO_ONNX_NAME = "model.onnx"
DEVICE_NAME = "Dragonwing IQ-9075 EVK"

# Input dimensions — must match the model's expected input.
BATCH = 1
C = 3
T = 16
H = 112
W = 112

# =============================================================================


def run_profile(model, device):
    """Submit a profiling job to QAI Hub and return the job ID."""
    profile_job = qai_hub.submit_profile_job(
        model=model,
        device=device,
        options="--max_profiler_iterations 100"
    )
    return profile_job.job_id


def compile_model(model, device, input_specs):
    """Compile an ONNX model on QAI Hub and return the compile job ID."""
    options = "--target_runtime qnn_context_binary"
    compile_job = qai_hub.submit_compile_job(
        model=model,
        device=device,
        input_specs=input_specs,
        options=options
    )
    return compile_job.job_id


def main():

    if not ONNX_DIR:
        print("Error: 'ONNX_DIR' is not set. Update it at the top of this script.")
        sys.exit(1)

    VIDEO_ONNX_PATH = os.path.join(ONNX_DIR, VIDEO_ONNX_NAME)
    if not os.path.exists(VIDEO_ONNX_PATH):
        print(f"Error: '{VIDEO_ONNX_PATH}' not found. Run your export script first.")
        sys.exit(1)

    print(f"Loading ONNX model from {VIDEO_ONNX_PATH}...")
    onnx_video_model = onnx.load(VIDEO_ONNX_PATH)

    try:
        onnx.checker.check_model(onnx_video_model)
        print("ONNX model is valid ✅")
    except onnx.checker.ValidationError as e:
        print("ONNX model validation failed ❌")
        print(e)
        sys.exit(1)

    device = qai_hub.Device(DEVICE_NAME)

    # Input name must match the name used in torch.onnx.export(input_names=[...])
    input_specs = {
        "video": ((BATCH, C, T, H, W), "float32")
    }

    print("\nSubmitting compilation job to QAI Hub...")
    compile_id = compile_model(
        model=onnx_video_model,
        device=device,
        input_specs=input_specs,
    )
    print(f"Compilation job ID: {compile_id}")

    print("\nWaiting for compilation to finish before submitting profiling job...")
    target_model = qai_hub.get_job(compile_id).get_target_model()

    print("Submitting profiling job to QAI Hub...")
    profile_id = run_profile(target_model, device)
    print(f"Profiling job ID: {profile_id}")
    print("Done.")


if __name__ == "__main__":
    main()
