import random
import numpy as np
import torch 
import os 



def set_seed(seed=42):
    random.seed(seed)                           # Python built-in random module
    np.random.seed(seed)                        # NumPy random generator
    torch.manual_seed(seed)                     # PyTorch CPU random seed
    torch.cuda.manual_seed(seed)                # PyTorch current GPU random seed
    torch.cuda.manual_seed_all(seed)            # PyTorch all GPUs random seed
    torch.backends.cudnn.deterministic = True   # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.benchmark = False      # Disable auto-optimization to prevent non-deterministic behavior
    os.environ["PYTHONHASHSEED"] = str(seed)    # Control hash-based randomness in Python

set_seed(42)



def subsample(data, targets, num_data, num_classes):
    """ subsample the data to num_data images """
    idx = targets < num_classes  # Select samples with class labels less than num_classes (e.g., only classes 0, 1, 2)
    new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255  # Select the first num_data images and normalize to [0,1]
    new_targets = targets[idx][:num_data]  # Select corresponding labels for the subsampled images
    return torch.utils.data.TensorDataset(new_data, new_targets)  # Create a TensorDataset with the filtered images and labels



def compute_all_euclidean_distances(models_dict, test_image_pairs, max_decoder_num=3, num_vaes=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    distances = {}  # Dictionary to store distances: number_of_decoders → pair_idx → list of distances

    # Iterate over the number of decoders from 1 to max_decoder_num
    for number_of_decoders in range(1, max_decoder_num + 1):
        distances[number_of_decoders] = []

        for pair_idx, (x1, x2) in enumerate(test_image_pairs):
            dij_list = []

            for m in range(num_vaes):  # Iterate over num_vaes
                model_name = f"vae_d{max_decoder_num}_seed{1000 + m}"
                model = models_dict[model_name]
                model.eval()

                # Encode to latent space (mean)
                with torch.no_grad():
                    z1 = model.encoder(x1.unsqueeze(0).to(device)).base_dist.loc.squeeze(0)
                    z2 = model.encoder(x2.unsqueeze(0).to(device)).base_dist.loc.squeeze(0)

                # Compute Euclidean distance
                euclidean = torch.norm(z1 - z2, p=2).item()
                dij_list.append(euclidean)

            distances[number_of_decoders].append(dij_list)  # shape: [num_vaes models]

    return distances


def compute_avg_cov_per_decoder(distances_dict):
    """
    Input:
    - distances_dict: decoder_num → list of list of dij (see the previous step output)

    Output:
    - decoder_nums: [1, 2, 3]
    - avg_cov_list: Average CoV for each decoder
    """
    decoder_nums = []
    avg_cov_list = []
    for d, pair_dists in distances_dict.items():  # Each decoder number d corresponds to a list of lists
        cov_list = []  # CoV list for this decoder
        for dij_list in pair_dists:  # Each image pair corresponds to distances from different models
            dij_array = np.array(dij_list)
            mean_dij = np.mean(dij_array)  # Mean distance across different models
            std_dij = np.std(dij_array)
            if mean_dij > 0:
                cov_ij = std_dij / mean_dij
                cov_list.append(cov_ij)  # Append CoV for this decoder and test point

        avg_cov = np.mean(cov_list)  # Compute the average CoV for this decoder
        decoder_nums.append(d)
        avg_cov_list.append(avg_cov)

    return decoder_nums, avg_cov_list
