import torch 
import torch.nn as nn 


class CubicCurve(nn.Module):
    def __init__(self, c0, c1):
        """
        Cubic polynomial curve module with fixed endpoints and parameterized middle section.

        Parameters:
        - c0: [d] Tensor representing the start point.
        - c1: [d] Tensor representing the end point.
        """
        super().__init__()
        self.register_buffer("c0", c0)
        self.register_buffer("c1", c1)

        d = c0.shape[0]
        # Learnable parameters: w1, w2 ∈ R^d
        self.w1 = nn.Parameter(torch.zeros(d, requires_grad=True))
        self.w2 = nn.Parameter(torch.zeros(d, requires_grad=True))

    def forward(self, t):
        """
        Forward pass to compute the cubic curve.

        Parameters:
        - t: [B] or [B, 1], curve parameter t ∈ [0, 1].

        Returns:
        - c(t): [B, d], computed points on the cubic curve.
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)  # Reshape to [B, 1],so that t can * c0

        # [B, d] broadcasting
        t1 = t
        t2 = t ** 2
        t3 = t ** 3

        w1 = self.w1  # [d]
        w2 = self.w2  # [d]
        w3 = -w1 - w2  # Ensure smooth transition

        # Linear interpolation between c0 and c1
        linear = (1 - t) * self.c0 + t * self.c1  # [B, d]

        # Residual polynomial component
        residual = w1 * t1 + w2 * t2 + w3 * t3  # [B, d]

        return linear + residual  # [B, d], final cubic curve output



def compute_curve_energy(curve, decoders, T=16, num_samples=1, fixed_indices=None, device='cuda'):
    """
    Compute the energy of a curve using fixed decoder indices to ensure the objective function remains consistent.

    Parameters:
    - curve: An instance of CubicCurve
    - decoders: List of decoder modules
    - T: Number of time steps,I set T=20, maybe too small? but too large require longer time
    - num_samples: Number of Monte Carlo samples,just follow the paper using one-monte carlo estimate
    - fixed_indices: Pre-fixed but still random (seems weird will explain in the optimize part)
      decoder indices [(idx1_t0, idx2_t0), (idx1_t1, idx2_t1), ...]
    - device: Computing device

    Returns:
    - Scalar energy value
    """
    total_energy = 0.0  # Accumulate energy over all time steps

    for i in range(T):
        t0 = torch.tensor([i / T], device=device, dtype=torch.float32)
        t1 = torch.tensor([(i + 1) / T], device=device, dtype=torch.float32)

        x0 = curve(t0)  # γ(t0), shape [1, d]
        x1 = curve(t1)  # γ(t1)

        energy = 0.0  # Energy for the current time step

        for _ in range(num_samples):
            # recall the decoder:return td.Independent(td.Normal(loc=means, scale=1e-1), 3)
            # And I have asked TA we can directly use mean to calculate the norm differences
            # He said we don't need to do KL divergence like page 104 in textbook
            # **Compute mean outputs for all decoders**
            decoded_x0_means = torch.stack([decoder(x0).mean for decoder in decoders], dim=0)
            decoded_x1_means = torch.stack([decoder(x1).mean for decoder in decoders], dim=0)

            # **Use fixed decoder indices**
            idx1, idx2 = fixed_indices[i]  # Retrieve fixed indices
            sampled_mean_x0 = decoded_x0_means[idx1]  # Select decoder mean for x0
            sampled_mean_x1 = decoded_x1_means[idx2]  # Select decoder mean for x1

            # Compute L2 norm
            energy += torch.norm(sampled_mean_x1 - sampled_mean_x0, p=2)

        # Take Monte Carlo average, one sample monte carlo
        total_energy += energy / num_samples

    return T * total_energy  # Return total energy over all time steps，although T is just a constant







class PieceWiseCurve:

    """ 
    Line segments by the two points c0 and c1, with num_segments number of points.
    """
    def __init__(self, c0, c1, num_segments, data_points, poly=True, sigma=0.1, epsilon=1e-4):
        self.c0 = c0
        self.c1 = c1
        self.sigma = 0.1 
        self.data_points = data_points
        self.epsilon = 1e-4
        self.num_segments = num_segments
        self.control_points = torch.stack([
            c0 + (i/(num_segments + 1)) * (c1-c0)
            for i in range(1, num_segments + 1) 
        ])
        self.control_points.requires_grad = True

    def __call__(self, t):
        return self.points[t]
    
        
    def calculate_density(self):
        densities = []
        for cp in self.control_points:
            # Squared distance from data points to this control point
            sq_dist = torch.sum((self.data_points - cp)**2, dim=1)
            gaussian = torch.exp(-sq_dist / (2 * self.sigma**2))
            density = torch.mean(gaussian)
            densities.append(density)
        return torch.stack(densities)     
    
    def calculate_metric(self):
        """
        Calculate the metric of the curve.
        """
        density = self.calculate_density() 
        metric = 1 / (density + self.epsilon)
        return metric
    
    def compute_energy(self):
        metrics = self.calculate_metric()
        velocities = self.control_points[1:] - self.control_points[:-1]
        segment_metrics = metrics[:-1]
        energy = torch.sum(torch.norm(velocities, dim=1)**2 * segment_metrics)
        return energy
    

    def optimize_control_points(self):
        optimizer = torch.optim.Adam([self.control_points], lr=0.01)
        loss = []
        for i in range(1000):
            optimizer.zero_grad()
            energy = self.compute_energy()
            energy.backward()
            optimizer.step()

            if i%100 == 0:
                print(f"iteration: {i}")

                print(f"loss: {energy}")
            loss.append(energy.item())
