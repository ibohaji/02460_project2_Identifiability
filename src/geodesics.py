import torch 
import random 
from configs import TrainingConfig
from src.curves import CubicCurve, compute_curve_energy
from tqdm import tqdm 




def optimize_geodesic(c0, c1, curve,  decoders, T=16, steps=500, lr=1e-2, device='cuda',
                      early_stopping_n=100, early_stopping_delta=1e-4):
    """
    Optimize a geodesic curve while ensuring the objective function remains unchanged during optimization.

    Parameters:
    - c0: Starting point
    - c1: Endpoint
    - decoders: List of decoder modules
    - T: Number of time steps
    - steps: Number of optimization iterations
    - lr: Learning rate
    - device: Computing device
    - early_stopping_n: Number of steps to check for early stopping
    - early_stopping_delta: Minimum required improvement to continue

    Returns:
    - Optimized curve
    - Logged energy values
    """
    curve = curve(c0, c1).to(device)  # Initialize the curve
    optimizer = torch.optim.Adam(curve.parameters(), lr=lr)  # Adam optimizer
    energy_log = []  # Store energy values

    # **Pre-generate fixed decoder indices**
    fixed_indices = [(torch.randint(0, len(decoders), (1,), device=device).item(),
                      torch.randint(0, len(decoders), (1,), device=device).item())
                     for _ in range(T)]

    best_energy = float('inf')
    no_improve_count = 0  # Counter for early stopping

    with tqdm(range(steps)) as pbar:
        for step in pbar:
            optimizer.zero_grad()  # Clear gradients
            energy = compute_curve_energy(curve, decoders, T=T, fixed_indices=fixed_indices, device=device)  # Compute energy
            energy.backward()  # Backpropagation
            optimizer.step()  # Update parameters

            energy_value = energy.item()
            energy_log.append(energy_value)  # Store energy value

            # Early Stopping Logic
            if energy_value < best_energy - early_stopping_delta:
                best_energy = energy_value
                no_improve_count = 0  # Reset counter
            else:
                no_improve_count += 1

            if no_improve_count >= early_stopping_n:
                print(f"Early stopping at step {step}, energy: {energy_value:.6f}")
                break  # Stop training if no improvement

            # Update progress bar
            pbar.set_description(f"Energy: {energy_value:.6f}")

    return curve, energy_log

def compute_all_geodesic_distances(models_dict, test_image_pairs, max_decoder_num=3, num_vaes=10):
    device = TrainingConfig.device
    T = 5  # Time steps for segmenting the interval to calculate energy
    steps = 10  # Optimization steps
    lr = 1e-2  # Optimal learning rate
     # Used for testing
    distances = {}  # number_of_decoders → pair_idx → list of distances

    # Iterate over the number of decoders from 1 to max_decoder_num
    for number_of_decoders in range(1, max_decoder_num + 1):
        distances[number_of_decoders] = []

        for pair_idx, (x1, x2) in enumerate(test_image_pairs):
            dij_list = []

            for m in range(num_vaes):
                model_name = f"vae_d{max_decoder_num}_seed{1000 + m}"
                model = models_dict[model_name]
                model.eval()

                # Encode to latent space (mean)
                with torch.no_grad():
                    z1 = model.encoder(x1.unsqueeze(0).to(device)).base_dist.loc.squeeze(0)
                    z2 = model.encoder(x2.unsqueeze(0).to(device)).base_dist.loc.squeeze(0)

                # Randomly select number_of_decoders decoders
                selected_decoders = random.sample(list(model.decoders), number_of_decoders)

                # Compute geodesic energy (use the last value from energy_log)
                curve, energy_log = optimize_geodesic(
                    z1, z2, decoders=selected_decoders, T=T, steps=steps, lr=lr, device=device
                )
                energy = energy_log[-1]  # Use the last energy value from optimization

                dij_list.append(energy)

            distances[number_of_decoders].append(dij_list)  # shape: [10 models]

    return distances
