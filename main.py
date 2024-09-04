import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import font_manager
import matplotlib
import time

# Set the font to Times New Roman
font_path = "/home/zihan/.local/share/fonts/Times New Roman.ttf"
font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"] = "Times New Roman"

# Enable LaTeX rendering
plt.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{times}"


def generate_random_rotation_matrix(size):
    Q, _ = np.linalg.qr(np.random.randn(size, size))
    return Q


def generate_random_scaling_matrix(size):
    D = np.diag(np.random.rand(size) + 0.5)
    return D


def generate_low_rank_diagonal_matrix(size, rank):
    D = np.zeros((size, size))
    D[:rank, :rank] = np.diag(np.random.rand(rank) + 0.1)
    return D


def generate_and_save_data(num_experiments=1000):
    matrix_size = 100
    low_rank_range = range(5, 26)
    num_matrices = 10
    sigma = 0.0001

    A1 = [generate_random_rotation_matrix(matrix_size) for _ in range(num_matrices)]
    A2 = [generate_random_scaling_matrix(matrix_size) for _ in range(num_matrices)]
    A1 = np.array(A1)
    A2 = np.array(A2)

    r_values = []
    k_values_95 = []
    k_values_99 = []

    for _ in range(num_experiments):
        rank = np.random.choice(low_rank_range)
        I_i = generate_low_rank_diagonal_matrix(matrix_size, rank)

        A = A1[np.random.randint(0, len(A1))]
        B = A1[np.random.randint(0, len(A1))]
        C = A2[np.random.randint(0, len(A2))]
        D = A2[np.random.randint(0, len(A2))]

        M = A @ B @ I_i @ C @ D
        M_prime = M + np.random.normal(0, sigma, M.shape)

        U, S, Vt = np.linalg.svd(M_prime)

        total_variance = np.sum(S)
        variance_sum = 0
        k_95 = 0
        k_99 = 0
        for i, singular_value in enumerate(S):
            variance_sum += singular_value
            if k_95 == 0 and variance_sum / total_variance >= 0.95:
                k_95 = i + 1
            if variance_sum / total_variance >= 0.99:
                k_99 = i + 1
                break

        r_values.append(rank)
        k_values_95.append(int(k_95 * 1.4))
        k_values_99.append(int(k_99 * 1.4))

    df = pd.DataFrame({"r": r_values, "k_95": k_values_95, "k_99": k_values_99})
    df.to_csv("experiment_data.csv", index=False)
    print("Data saved to experiment_data.csv")


def plot_heatmap(data, cmap, output_filename, variance, title):
    r_values = data["r"]
    k_values = data[f"k_{variance}"]

    heatmap, xedges, yedges = np.histogram2d(
        r_values, k_values, bins=(30, 30), range=[[0, 30], [0, 30]]
    )

    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    sns.heatmap(
        heatmap,
        cmap=cmap,
        cbar_kws={"label": "Frequency"},
        xticklabels=5,
        yticklabels=5,
        ax=ax,
    )

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel(r"Number of ellipses ($\beta$)", fontsize=12)
    ax.set_ylabel(r"$k$ (scaled by 1.4)", fontsize=12)

    # Adding only the red y=x line
    ax.plot([0, 30], [0, 30], "r-", label=r"$$k = \beta$$")

    ax.legend(fontsize=10)

    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=600, bbox_inches="tight")
    print(f"Figure saved as {output_filename}")
    plt.close()


if __name__ == "__main__":
    tick = time.time()
    # generate_and_save_data()
    tock = time.time()
    print(f"Time taken to generate data: {tock - tick:.2f} seconds")

    tick = time.time()
    data = pd.read_csv("experiment_data.csv")
    tock = time.time()
    print(f"Time taken to read data: {tock - tick:.2f} seconds")

    plot_heatmap(data, "coolwarm", "heatmap_95_percent.png", "95", "")
    # plot_heatmap(data, "coolwarm", "heatmap_99_percent.png", "99")
