from model import ResidualUNet
import torch
import onnxruntime as ort
import numpy as np
import h5py

dataset = h5py.File("dataset/train.h5", 'r')

x = dataset["laf3"][0]
x = torch.from_numpy(x[None, None]).float()

print(x.shape)
output_name = "model_self_sup.onnx"
# model, config = ResidualUNet.load("runs/", "20260417_154943_*/*epoch40.pt")
# model, config = ResidualUNet.load("runs/", "20260506_135318*/*epoch40.pt")
# model, config = ResidualUNet.load("runs/20260511_174507_resized_augmented_gaussian/residual_unet_epoch23.pt")
# model, params = ResidualUNet.load("runs/", "20260520_163333_preprocessed_gaussian_2x/*epoch40.pt")
# model, params = ResidualUNet.load("runs/20260520_190549_profile_2x_gaussian_v2/residual_unet_epoch40.pt")
# model, params = ResidualUNet.load("runs/20260527_175506_combined_g2x_precalc_cal_const/residual_unet_epoch40.pt")
# model, params = ResidualUNet.load("runs/20260602_191131_preprocessed_bc2_logspace/residual_unet_epoch40.pt")
# model, params = ResidualUNet.load("runs2/20260616_135901_bc2_norm_logspace_norm_lr8e-05_only_reg_gauss/residual_unet_epoch100.pt")
model, params = ResidualUNet.load("runs2/20260616_145357_bc2_norm_logspace_norm_lr8e-05_only_reg_gauss_tv/residual_unet_epoch100.pt")
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
np.testing.assert_allclose(torch_output, ort_output, rtol=1e-01, atol=1e-02)
print("✓ Export validated within rtol=1e-01, atol=1e-02")