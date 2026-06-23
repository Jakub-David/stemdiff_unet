import torch
import onnxruntime as ort
import numpy as np
import h5py

def convert(model, output_name, h5_data, verbose=True):
    dataset = h5_data

    x = dataset["laf3"][0]
    x = torch.from_numpy(x[None, None]).float()

    model = model.eval()

    with torch.no_grad():
        torch_output = model.predict(x)


    batch_dim = torch.export.Dim("dim")
    side_dim = torch.export.Dim("side", min=256, max=4096)
    sc = torch.export.ShapesCollection()
    sc[x] = (batch_dim, 1, side_dim, side_dim)

    onnx_program = torch.onnx.export(model, x, dynamic_shapes=sc, verbose=verbose)
    onnx_program.save(output_name)


    # Create ONNX Runtime session
    ort_session = ort.InferenceSession(output_name)
    # Prepare input: convert to numpy, ensure correct type
    ort_inputs = {"x": x.numpy().astype(np.float32)}
    ort_output = ort_session.run(None, ort_inputs)[0].squeeze()

    # Compare with relative (rtol) and absolute (atol) tolerance
    np.testing.assert_allclose(torch_output, ort_output, rtol=1e-01, atol=1e-02, verbose=verbose)
    if verbose:
        print("✓ Export validated within rtol=1e-01, atol=1e-02")

if __name__ == "__main__":
    from model import ResidualUNet
    model, params = ResidualUNet.load("runs/20260622_132820_self_sup_ncNone_lcw0.01_l1w0.001/residual_unet_epoch20.pt")
    convert(model, "model_self_sup.onnx", h5py.File("dataset/train.h5", 'r'))