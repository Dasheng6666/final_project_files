import os
import sys
import types
import math
import importlib
from typing import Tuple

import numpy as np
import pytest

import torch
import torch.nn as nn
from PIL import Image

# -------------------------------
# torchsummary  piq
# -------------------------------
if "torchsummary" not in sys.modules:
    torchsummary_stub = types.ModuleType("torchsummary")

    def _summary(*args, **kwargs):
      
        return None

    torchsummary_stub.summary = _summary
    sys.modules["torchsummary"] = torchsummary_stub

if "piq" not in sys.modules:
    piq_stub = types.ModuleType("piq")

    def _psnr(gen, org, data_range=1.0):
        
        return torch.tensor(42.0)

    def _ssim(gen, org, data_range=1.0):
        return torch.tensor(0.99)

    piq_stub.psnr = _psnr
    piq_stub.ssim = _ssim
    sys.modules["piq"] = piq_stub

# -------------------------------
# file
# -------------------------------
MODULE_NAME = os.getenv("RECON_MODULE", "network")
mod = importlib.import_module(MODULE_NAME)


# -------------------------------
# Fixtures
# -------------------------------
@pytest.fixture(scope="session", autouse=True)
def _fix_seed():
    # seed
    mod.set_seed(1234)


@pytest.fixture()
def cpu_device():
    return torch.device("cpu")


@pytest.fixture()
def tiny_image(tmp_path) -> str:
    """ 16x16 image"""
    h, w = 16, 16
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            arr[y, x] = [
                (x * 17 + y * 5) % 256,
                (y * 31) % 256,
                (x * 13) % 256,
            ]
    p = tmp_path / "test_image.png"
    Image.fromarray(arr).save(p)
    return str(p)


@pytest.fixture()
def small_patch() -> Tuple[np.ndarray, np.ndarray, int, int]:
    """返回一个 8x8 的随机彩色 patch 的 (coords, rgb, h, w)。"""
    img = (np.random.rand(8, 8, 3) * 255).astype(np.uint8)
    pil = Image.fromarray(img)
    coords, rgb_values, h, w = mod.extract_patch_data(pil)
    return coords, rgb_values, h, w


# -------------------------------
# function
# -------------------------------

def test_set_seed_determinism():
    mod.set_seed(777)
    a1 = torch.randn(5)
    n1 = np.random.rand(5)
    mod.set_seed(777)
    a2 = torch.randn(5)
    n2 = np.random.rand(5)
    assert torch.allclose(a1, a2)
    assert np.allclose(n1, n2)


def test_positional_encoding_shape_and_range():
    enc = mod.positional_encoding(0.5, 0.25, num_frequencies=10)
    assert isinstance(enc, torch.Tensor)
    assert enc.ndim == 1 and enc.shape[0] == 40  # 2 * 2 * num_frequencies
    assert torch.all(enc <= 1.0) and torch.all(enc >= -1.0)


def test_early_stopping_behavior():
    es = mod.EarlyStopping(patience=2, delta=0.1, stop_threshold=1e-12)
   
    assert es(1.0) is False
    #  delta 
    assert es(0.95) is False
    # ->  patience 
    assert es(0.96) is True

    # threshold
    es2 = mod.EarlyStopping(patience=100, delta=0.1, stop_threshold=1e-3)
    assert es2(5e-4) is True


def test_sine_layer_init_bounds():
    # first layer: bound = 1/in_features
    f = 10
    layer_first = mod.SineLayer(f, 5, is_first=True, omega_0=30.0)
    w = layer_first.linear.weight.detach().cpu().numpy()
    bound_first = 1.0 / f
    assert np.max(w) <= bound_first + 1e-6
    assert np.min(w) >= -bound_first - 1e-6

    # hidden: bound = sqrt(6/in_features)/omega_0
    layer_mid = mod.SineLayer(f, 5, is_first=False, omega_0=30.0)
    w2 = layer_mid.linear.weight.detach().cpu().numpy()
    bound_mid = math.sqrt(6.0 / f) / 30.0
    assert np.max(w2) <= bound_mid + 1e-6
    assert np.min(w2) >= -bound_mid - 1e-6


def test_frequency_init_applies_bounds():
    lin = nn.Linear(10, 5)
    init = mod.frequency_init(25)
    lin.apply(init)
    w = lin.weight.detach().cpu().numpy()
    bound = math.sqrt(6.0 / 10) / 25.0
    assert np.max(w) <= bound + 1e-6
    assert np.min(w) >= -bound - 1e-6


# -------------------------------
# model
# -------------------------------

def test_mlp_forward_shape_and_range():
    model = mod.MLP()
    # ps test
    pts = [(i, i * 0.5) for i in range(7)]
    enc = torch.stack([mod.positional_encoding(x, y) for x, y in pts]).float()
    out = model(enc)
    assert out.shape == (7, 3)
    assert torch.all(out >= 0) and torch.all(out <= 1)


def test_siren_forward_shape_and_range(cpu_device):
    model = mod.Siren().to(cpu_device)
    coords = torch.rand(9, 2, device=cpu_device)
    out = model(coords)
    assert out.shape == (9, 3)
    assert torch.all(out >= 0) and torch.all(out <= 1)


def test_filmsiren_forward_shape_and_range(cpu_device):
    model = mod.FilmSIREN().to(cpu_device)
    coords = torch.rand(11, 2, device=cpu_device)
    out = model(coords)
    assert out.shape == (11, 3)
    assert torch.all(out >= 0) and torch.all(out <= 1)
    # z 
    assert hasattr(model, "z") and isinstance(model.z, nn.Parameter)


# -------------------------------
# data and mask
# -------------------------------

def test_extract_patch_data_and_mask():
    #  4x4 image
    img = np.array([
        [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]],
        [[255, 0, 255], [0, 255, 255], [128, 128, 128], [64, 64, 64]],
        [[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120]],
        [[130, 140, 150], [160, 170, 180], [190, 200, 210], [220, 230, 240]],
    ], dtype=np.uint8)
    pil = Image.fromarray(img)
    coords, rgb, h, w = mod.extract_patch_data(pil)
    assert coords.shape == (h * w, 2)
    assert rgb.shape == (h * w, 3)
    assert h == 4 and w == 4
    assert (rgb >= 0).all() and (rgb <= 1).all()

    mask = mod.create_weight_mask(8, 4)
    assert mask.shape == (8, 8, 3)
    assert np.isclose(mask.max(), 1.0)
    assert np.isclose(mask.min(), 0.0)
    # edge 0
    assert np.all(mask[0, :, :] == 0) and np.all(mask[:, 0, :] == 0)
    
from network import extract_patch_data    
def test_patch_data_shapes():
    patch = Image.new("RGB", (64, 64), (255, 0, 0))
    coords, rgb_values, h, w = extract_patch_data(patch)
    assert coords.shape == (h*w, 2)
    assert rgb_values.shape == (h*w, 3)


# -------------------------------
# train and generate
# -------------------------------

def test_train_model_runs_minimal_steps_each_model(small_patch, cpu_device):
    coords, rgb, h, w = small_patch

    # MLP（ PE）
    mlp = mod.MLP().to(cpu_device)
    mlp_trained = mod.train_model(mlp, coords, rgb, cpu_device, total_steps=2, steps_til_summary=1)
    assert isinstance(mlp_trained, mod.MLP)

    # Siren
    siren = mod.Siren().to(cpu_device)
    siren_trained = mod.train_model(siren, coords, rgb, cpu_device, total_steps=2, steps_til_summary=1)
    assert isinstance(siren_trained, mod.Siren)

    # FilmSIREN
    filmsiren = mod.FilmSIREN().to(cpu_device)
    filmsiren_trained = mod.train_model(filmsiren, coords, rgb, cpu_device, total_steps=2, steps_til_summary=1)
    assert isinstance(filmsiren_trained, mod.FilmSIREN)


def test_generate_patch_shape_and_dtype(small_patch, cpu_device):
    coords, rgb, h, w = small_patch
    model = mod.Siren().to(cpu_device)
    # littke step
    model = mod.train_model(model, coords, rgb, cpu_device, total_steps=1, steps_til_summary=1)
    out = mod.generate_patch(model, cpu_device, patch_size=8, actual_size=(h, w))
    assert out.shape == (h, w, 3)
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255


# -------------------------------
# main (end to end)
# -------------------------------
@pytest.mark.parametrize("model_type", ["mlp", "siren", "filmsiren"])
def test_main_end_to_end(tmp_path, tiny_image, model_type, monkeypatch):
    # work fold path
    monkeypatch.chdir(tmp_path)
    out = mod.main(
        model_type=model_type,
        image_path=tiny_image,
        patch_size=8,
        overlap=4,
        total_steps=1,
    )
    assert isinstance(out, np.ndarray)
    assert out.shape == (16, 16, 3)
    assert out.dtype == np.uint8

    # oytput
    out_dir = tmp_path / f"output_{model_type}"
    out_img = out_dir / f"reconstructed_{model_type}.png"
    assert out_img.exists()


def test_main_unknown_model_type_raises(tmp_path, tiny_image, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        mod.main(
            model_type="unknown",
            image_path=tiny_image,
            patch_size=8,
            overlap=4,
            total_steps=1,
        )


def test_main_generates_model_output(tmp_path):
    """
    Ensure that `main` returns the model-generated output,
    not a direct copy of the original image.
    This helps prevent data leakage between "training" and "test" data.
    """
    # 1. Create a random small test image
    h, w = 8, 8
    img_array = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    img_path = tmp_path / "original.png"
    Image.fromarray(img_array).save(img_path)

    # 2. Import main function (change to your module name if different)
    from network import main  # If main is in network.py, use `from network import main`

    # 3. Run main with small parameters for quick testing
    reconstructed = main(
        model_type="mlp",
        image_path=str(img_path),
        patch_size=8,
        overlap=2,
        total_steps=10
    )

    # 4. Validate the output
    assert isinstance(reconstructed, np.ndarray), "Output must be a numpy array"
    assert reconstructed.shape == (h, w, 3), "Output shape must match input image"
    assert reconstructed.dtype == np.uint8, "Output must be uint8"
    assert not np.array_equal(
        reconstructed, img_array
    ), "Output is identical to the original image — possible data leakage"

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
