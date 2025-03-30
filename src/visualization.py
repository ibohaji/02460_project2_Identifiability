

def plot_cov_comparison(decoder_nums, avg_cov_geo, avg_cov_euc, save_path="cov_compare.png"):
    import matplotlib.pyplot as plt
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

