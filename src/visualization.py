
from matplotlib import pyplot as plt
import seaborn as sns
from loguru import logger


def plot_cov_comparison(decoder_nums, avg_cov_geo, avg_cov_euc, save_path="cov_compare.png"):
    plt.figure(figsize=(6, 4))
    plt.plot(decoder_nums, avg_cov_geo, marker='o', label="Geodesic CoV")
    plt.plot(decoder_nums, avg_cov_euc, marker='s', label="Euclidean CoV")
    plt.xlabel("Number of Decoders")
    plt.ylabel("Average CoV")
    plt.title("CoV vs Number of Decoders")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()



def plot_latent_variables(latent_variables, labels, save_path="latent_variables.png"):

    latent_variables = latent_variables.detach().cpu().numpy().reshape(-1, 2)
    labels = labels.detach().cpu().numpy() 
    logger.info(f"Plotting latent variables to {save_path}")
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    plt.figure(figsize=(10, 8))
    plt.scatter(latent_variables[:, 0], latent_variables[:, 1], c=labels)
    plt.xlabel('Latent Variable 1')
    plt.ylabel('Latent Variable 2')
    plt.title('Latent Variables')
    plt.savefig(save_path, dpi=300)
    logger.info(f"Saved latent variables to {save_path}")
    plt.close()



