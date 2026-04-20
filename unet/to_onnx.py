from model import ResidualUNet
from data import AugmentedDataset
import torch
import onnxruntime as ort
import numpy as np

dataset = AugmentedDataset("dataset1.1/train.h5")

x, y = dataset[0]
x = x[None]

print(x.shape)

model, config = ResidualUNet.load("runs/", "20260417_154943_*/*epoch40.pt")
model = model.eval()

with torch.no_grad():
    torch_output, _ = model.predict(x)


dim = torch.export.Dim("dim")
sc = torch.export.ShapesCollection()
sc[x] = (dim, 1, 256, 256)

onnx_program = torch.onnx.export(model, x, dynamic_shapes=sc)
onnx_program.save("model.onnx")


# Create ONNX Runtime session
ort_session = ort.InferenceSession("model.onnx")
# Prepare input: convert to numpy, ensure correct type
ort_inputs = {"x": x.numpy().astype(np.float32)}
ort_output = ort_session.run(None, ort_inputs)[0].squeeze()

# Compare with relative (rtol) and absolute (atol) tolerance
np.testing.assert_allclose(torch_output, ort_output, rtol=1e-02, atol=1e-03)
print("✓ Export validated within rtol=1e-02, atol=1e-03")