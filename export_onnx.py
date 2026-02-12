import os
import torch
import torchvision
import onnxscript
from qai_hub_models.models.resnet_2plus1d import Model
from torch import nn

try:
    from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
    TORCHVISION_AVAILABLE = True
except Exception:
    TORCHVISION_AVAILABLE = False

# MAKE SURE TO RUN THIS FILE FIRST BEFORE compile_and_profile.py IF YOU DO NOT HAVE exported_onnx directory

# -----------------------------
# 0. Configuration
# -----------------------------
ONNX_DIR = "exported_onnx"
ONNX_NAME = "r2plus1dQEVD"

device = torch.device("cpu")  # export on CPU to avoid device issues
os.makedirs(ONNX_DIR, exist_ok=True)
onnx_path = os.path.join(ONNX_DIR, ONNX_NAME)

# torchvision video models expect (N, C, T, H, W)
BATCH = 1
CHANNELS = 3
T = 8          # frames (set to what your model expects; common is 8, 16, 32)
H = 112       
W = 112

#for proper compile and profile, a dummy input must be given resembling an example input to the model
DUMMY_VIDEO = torch.randn(BATCH, CHANNELS, T, H, W, dtype=torch.float32, device=device)

# -----------------------------
# 1. Load your model
# -----------------------------
def load_model():
    """
    Replace this with your repo's model loading.
    Goal: return a torch.nn.Module in eval mode, float32, on CPU.
    """
    if not TORCHVISION_AVAILABLE:
        raise RuntimeError(
            "torchvision not available. Replace load_model() with your own model import."
        )

    # Option A: pretrained torchvision weights 
    # model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)

    # Option B: your finetuned checkpoint:
    num_classes = 92  #adjust to the proper amount of output classes
    model = torchvision.models.get_model("r2plus1d_18")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    ckpt = torch.load("./model_29.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=True)
    
    return model


print(f"Saving ONNX to: {os.path.abspath(onnx_path)}")
print("Loading model...")
model = load_model().to(device)
model = model.to(torch.float32)
model.eval()

## commented this out as this may not be needed 
# -----------------------------
# 2. (Optional) Wrap forward if your repo returns extra stuff
# -----------------------------
# class LogitsOnlyWrapper(torch.nn.Module):
#     """
#     Some repos return dicts/tuples (e.g., logits + aux outputs).
#     ONNX export is happiest with a single Tensor output.
#     """
#     def __init__(self, m: torch.nn.Module):
#         super().__init__()
#         self.m = m

#     def forward(self, video):
#         out = self.m(video)
#         # Handle common patterns:
#         if isinstance(out, dict):
#             # pick a sensible key; adjust if needed
#             if "logits" in out:
#                 return out["logits"]
#             # fallback: first value
#             return next(iter(out.values()))
#         if isinstance(out, (tuple, list)):
#             return out[0]
#         return out

export_model = model
# export_model = LogitsOnlyWrapper(model).to(device).eval()


# -----------------------------
# 3. Export settings
# -----------------------------
USE_DYNAMIC_AXES = False

dynamic_axes = None
if USE_DYNAMIC_AXES:
    dynamic_axes = {
        "video": {0: "batch", 2: "time"},
        "logits": {0: "batch"},
    }

# -----------------------------
# 4. Export
# -----------------------------
print("Exporting...")
with torch.no_grad():
    # Always sanity-run once to catch shape/dtype errors early
    y = export_model(DUMMY_VIDEO)
    assert isinstance(y, torch.Tensor), f"Model output must be Tensor for this exporter, got {type(y)}"
    print(f"Sanity output shape: {tuple(y.shape)}") #should give number of class labels 

    # Try dynamo=True first, fallback to classic exporter if it fails.
    # will save onnx verison of the model to exported_onnx directory
    try:
        torch.onnx.export(
            export_model,
            DUMMY_VIDEO,
            onnx_path,
            input_names=["video"],
            output_names=["logits"],
            opset_version=18,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
            verbose=False,
            export_params=True,
            training=torch.onnx.TrainingMode.EVAL,
            dynamo=True,
        )
        print("Exported with dynamo=True")
    except Exception as e:
        print(f"dynamo export failed ({type(e).__name__}: {e})")
        print("Retrying with dynamo=False (classic exporter)...")

        torch.onnx.export(
            export_model,
            DUMMY_VIDEO,
            onnx_path,
            input_names=["video"],
            output_names=["logits"],
            opset_version=18,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
            verbose=False,
            export_params=True,
            training=torch.onnx.TrainingMode.EVAL,
            dynamo=False,
        )
        print("Exported with dynamo=False")


print("Export complete and successful.")
