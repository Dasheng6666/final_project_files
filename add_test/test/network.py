import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchsummary import summary
import piq

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Early stopping mechanism
class EarlyStopping:
    def __init__(self, patience=2000, delta=0.00001, stop_threshold=0.000009):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.stop_threshold = stop_threshold

    def __call__(self, current_loss):
        if current_loss < self.stop_threshold:
            print(f"Loss reached threshold {self.stop_threshold}, early stopping")
            return True
        if current_loss < self.best_loss - self.delta:
            self.best_loss = current_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False
    
def positional_encoding(x, y, num_frequencies=10):
    device = x.device if torch.is_tensor(x) else 'cpu'
    frequencies = 2.0 ** torch.arange(num_frequencies, device=device) * np.pi

    if not torch.is_tensor(x):
        x = torch.tensor(x, device=device)
    if not torch.is_tensor(y):
        y = torch.tensor(y, device=device)

    x_enc = torch.cat([torch.sin(frequencies * x), torch.cos(frequencies * x)])
    y_enc = torch.cat([torch.sin(frequencies * y), torch.cos(frequencies * y)])
    return torch.cat([x_enc, y_enc])


# ====================== MODEL DEFINITIONS ======================
class MLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dims=[32]*4+[16], output_dim=3):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return torch.sigmoid(self.net(x))


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)

        with torch.no_grad():
            if self.is_first:
                bound = 1 / in_features
            else:
                bound = np.sqrt(6 / in_features) / omega_0
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class Siren(nn.Module):
    def __init__(self, in_features=2, hidden_features=32, hidden_layers=6, out_features=3):
        super().__init__()
        self.net = nn.ModuleList()

        # first
        self.net.append(SineLayer(in_features, hidden_features, is_first=True))

        # middle
        for _ in range(hidden_layers-2):
            self.net.append(SineLayer(hidden_features, hidden_features))
        self.net.append(SineLayer(hidden_features, 16))

        # last correct to half
        self.final_layer = nn.Sequential(
            nn.Linear(16, out_features),
            nn.Sigmoid()
        )
        with torch.no_grad():
            self.final_layer[0].weight.uniform_(-np.sqrt(6/hidden_features),
                                             np.sqrt(6/hidden_features))

    def forward(self, coords):
        x = coords
        for layer in self.net:
            x = layer(x)
        return self.final_layer(x)

# FiLM-modulated SIREN model
def frequency_init(freq):
    def init(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                bound = np.sqrt(6 / num_input) / freq
                m.weight.uniform_(-bound, bound)
    return init

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, freq, phase_shift):
        return torch.sin(freq * self.layer(x) + phase_shift)

class CustomMappingNetwork(nn.Module):
    def __init__(self, z_dim, map_output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(z_dim, 6),
            nn.LeakyReLU(0.2),
            nn.Linear(6, 6),
            nn.LeakyReLU(0.2),
            nn.Linear(6, 6),
            nn.LeakyReLU(0.2),
            nn.Linear(6, map_output_dim)
        )
        self.network.apply(frequency_init(25))
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def forward(self, z):
        out = self.network(z)
        return out[..., :out.shape[-1]//2], out[..., out.shape[-1]//2:]

class FilmSIREN(nn.Module):
    """Film-modulated SIREN architecture"""
    def __init__(self, input_dim=2, z_dim=64, hidden_dim=36, out_features=3, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.z = nn.Parameter(torch.randn(1, z_dim))
        self.network = nn.ModuleList([
            FiLMLayer(input_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        self.final_layer = nn.Linear(hidden_dim, out_features)
        self.mapping_network = CustomMappingNetwork(z_dim, len(self.network)*hidden_dim*2)
        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))

    def forward(self, coords):
        freqs, phases = self.mapping_network(self.z)
        freqs = freqs * 15 + 30
        x = coords
        for idx, layer in enumerate(self.network):
            start = idx * self.network[0].layer.out_features
            end = (idx+1) * self.network[0].layer.out_features
            x = layer(x, freqs[:, start:end], phases[:, start:end])
        return torch.sigmoid(self.final_layer(x))

# ====================== HELPER FUNCTIONS ======================
def extract_patch_data(patch_img):
    """Extract coordinates and RGB values from image patch"""
    patch_np = np.array(patch_img)
    h, w, _ = patch_np.shape
    y_coords, x_coords = np.indices((h, w))
    coords = np.stack((x_coords, y_coords), axis=-1).reshape(-1, 2)
    rgb_values = patch_np.reshape(-1, 3) / 255.0
    return coords, rgb_values, h, w

def create_weight_mask(patch_size, overlap):
    """Create weight mask for seamless blending"""
    mask = np.ones((patch_size, patch_size, 3), dtype=np.float32)
    for i in range(overlap):
        ratio = i / (overlap - 1)
        mask[:, i] *= ratio
        mask[:, -(i+1)] *= ratio
    for j in range(overlap):
        ratio = j / (overlap - 1)
        mask[j, :] *= ratio
        mask[-(j+1), :] *= ratio
    return mask

# ====================== TRAINING FUNCTIONS ======================
def train_model(model, coords, rgb_values, device, total_steps=10000, steps_til_summary=2000):
    """Train model function"""
    coords_tensor = torch.tensor(coords, dtype=torch.float32).to(device)

    #PS
    if isinstance(model, MLP):
        coords_tensor = torch.stack([
            positional_encoding(c[0], c[1], num_frequencies=10) for c in coords_tensor
        ])

    rgb_tensor = torch.tensor(rgb_values, dtype=torch.float32).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
    early_stop = EarlyStopping(patience=2000)
    
    for step in range(total_steps):
        optimizer.zero_grad()
        pred = model(coords_tensor)
        loss = criterion(pred, rgb_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if early_stop(loss.item()):
            print(f"Early stopping at step {step}")
            break
        if step % steps_til_summary == 0:
            print(f"Step {step}, Loss: {loss.item():.6f}")
    return model

def generate_patch(model, device, patch_size, actual_size):
    """Generate image patch using trained model"""
    h, w = actual_size
    y_coords, x_coords = np.indices((h, w))
    coords = np.stack((x_coords, y_coords), axis=-1).reshape(-1, 2)
    coords_tensor = torch.tensor(coords, dtype=torch.float32).to(device)
    if isinstance(model, MLP):
        coords_tensor = torch.stack([
            positional_encoding(c[0], c[1], num_frequencies=10) for c in coords_tensor
        ])
    with torch.no_grad():
        rgb_output = model(coords_tensor).cpu().numpy()
    rgb_output = (np.clip(rgb_output, 0, 1) * 255).astype(np.uint8)
    return rgb_output.reshape(h, w, 3)
def main(model_type='mlp', image_path='test_image.png', patch_size=64, overlap=16, total_steps=15000):
    """Main function for image reconstruction"""
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Selected model: {model_type.upper()}")
    output_folder = f"output_{model_type}"
    os.makedirs(output_folder, exist_ok=True)

    full_img = Image.open(image_path).convert('RGB')
    original_width, original_height = full_img.size
    original_np = np.array(full_img)
    stride = patch_size - overlap
    num_x = (original_width - overlap + stride - 1) // stride
    num_y = (original_height - overlap + stride - 1) // stride
    regenerated_sum = np.zeros((original_height, original_width, 3), dtype=np.float32)
    weight_sum = np.zeros((original_height, original_width, 3), dtype=np.float32)
    weight_mask = create_weight_mask(patch_size, overlap)
    psnr_values = []
    ssim_values = []
    prev_state_dict = None
    patch_count = 0
    
    for i in range(num_x):
        for j in range(num_y):
            left = i * stride
            upper = j * stride
            right = min(left + patch_size, original_width)
            lower = min(upper + patch_size, original_height)
            if i == num_x - 1:
                left = original_width - patch_size
            if j == num_y - 1:
                upper = original_height - patch_size

            patch = full_img.crop((left, upper, right, lower))
            coords, rgb_values, h_patch, w_patch = extract_patch_data(patch)
            actual_size = (h_patch, w_patch)
            
            if model_type == 'mlp':
                model = MLP().to(device)
            elif model_type == 'siren':
                model = Siren().to(device)
            elif model_type == 'filmsiren':
                model = FilmSIREN(device=device).to(device)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            if prev_state_dict is not None:
                if model_type == 'filmsiren':
                    current_state_dict = model.state_dict()
                    for key in prev_state_dict:
                        if key != 'z':
                            current_state_dict[key] = prev_state_dict[key]
                    model.load_state_dict(current_state_dict)
                else:
                    model.load_state_dict(prev_state_dict)
            
            if i == 0 and j == 0:
                if model_type == "mlp":
                    # MLP（positional_encoding）
                    print(f"\n{model_type} model summary:")
                    summary(model, input_size=(40,), device=device.type)
                else:    
                    print(f"\n{model_type} model summary:")
                    summary(model, input_size=(2,), device=device.type)
            
            print(f"\nTraining {model_type} patch ({i},{j}) at [{left}:{right}, {upper}:{lower}]...")
            trained_model = train_model(model, coords, rgb_values, device, total_steps=total_steps)
            
            if model_type == 'filmsiren':
                state_to_save = {k: v for k, v in trained_model.state_dict().items() if k != 'z'}
                prev_state_dict = copy.deepcopy(state_to_save)
            else:
                prev_state_dict = copy.deepcopy(trained_model.state_dict())
            
            model_path = os.path.join(output_folder, f"model_{i}_{j}.pth")
            torch.save(trained_model.state_dict(), model_path)
            patch_array = generate_patch(trained_model, device, patch_size, actual_size)
            original_patch = np.array(patch)
            gen_tensor = torch.from_numpy(patch_array/255.).permute(2,0,1).unsqueeze(0).float()
            org_tensor = torch.from_numpy(original_patch/255.).permute(2,0,1).unsqueeze(0).float()
            psnr = piq.psnr(gen_tensor, org_tensor, data_range=1.0).item()
            ssim = piq.ssim(gen_tensor, org_tensor, data_range=1.0).item()
            psnr_values.append(psnr)
            ssim_values.append(ssim)
            print(f"Patch {patch_count} metrics - PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
            actual_h, actual_w = lower - upper, right - left
            cropped_mask = weight_mask[:actual_h, :actual_w, :]
            patch_float = patch_array.astype(np.float32)
            regenerated_sum[upper:lower, left:right] += patch_float * cropped_mask
            weight_sum[upper:lower, left:right] += cropped_mask
            patch_count += 1
    
    regenerated = np.zeros_like(regenerated_sum)
    valid_mask = weight_sum > 1e-6
    regenerated[valid_mask] = regenerated_sum[valid_mask] / weight_sum[valid_mask]
    regenerated = np.clip(regenerated, 0, 255).astype(np.uint8)
    output_path = os.path.join(output_folder, f"reconstructed_{model_type}.png")
    Image.fromarray(regenerated).save(output_path)
    print("\n=== FINAL RESULTS ===")
    print(f"Model: {model_type.upper()}")
    print(f"Patches processed: {patch_count}")
    print(f"Local PSNR (avg): {np.mean(psnr_values):.2f} ± {np.std(psnr_values):.2f} dB")
    print(f"Local SSIM (avg): {np.mean(ssim_values):.4f} ± {np.std(ssim_values):.4f}")
    print(f"Reconstructed image saved to: {output_path}")
    return regenerated
