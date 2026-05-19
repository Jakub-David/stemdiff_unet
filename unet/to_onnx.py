from model import ResidualUNet
from data import AugmentedDataset
import torch
import onnxruntime as ort
import numpy as np

dataset = AugmentedDataset("dataset1.1/train.h5")

x, y = dataset[0]
x = x[None]

print(x.shape)
output_name = "model_resized.onnx"
# model, config = ResidualUNet.load("runs/", "20260417_154943_*/*epoch40.pt")
# model, config = ResidualUNet.load("runs/", "20260506_135318*/*epoch40.pt")
model, config = ResidualUNet.load("runs/20260511_174507_resized_augmented_gaussian/residual_unet_epoch23.pt")
model = model.eval()

with torch.no_grad():
    torch_output = model.predict(x)


batch_dim = torch.export.Dim("dim")
side_dim = torch.export.Dim("side", min=256, max=4096)
sc = torch.export.ShapesCollection()
sc[x] = (batch_dim, 1, side_dim, side_dim)

onnx_program = torch.onnx.export(model, x, dynamic_shapes=sc)
onnx_program.save(output_name)


# Create ONNX Runtime session
ort_session = ort.InferenceSession(output_name)
# Prepare input: convert to numpy, ensure correct type
ort_inputs = {"x": x.numpy().astype(np.float32)}
ort_output = ort_session.run(None, ort_inputs)[0].squeeze()

# Compare with relative (rtol) and absolute (atol) tolerance
np.testing.assert_allclose(torch_output, ort_output, rtol=1e-02, atol=1e-03)
print("✓ Export validated within rtol=1e-02, atol=1e-03")