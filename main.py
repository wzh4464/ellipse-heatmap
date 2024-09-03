import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def generate_random_rotation_matrix(size):
    """Generate a random rotation matrix of given size."""
    Q, _ = np.linalg.qr(np.random.randn(size, size))
    return Q

def generate_random_scaling_matrix(size):
    """Generate a random scaling matrix of given size."""
    D = np.diag(np.random.rand(size) + 0.5)  # Scaling factors between 0.5 and 1.5
    return D

def generate_low_rank_diagonal_matrix(size, rank):
    """Generate a low-rank diagonal matrix."""
    D = np.zeros((size, size))
    D[:rank, :rank] = np.diag(np.random.rand(rank) + 0.1)  # Diagonal elements between 0.1 and 1.1
    return D

# Parameters
matrix_size = 100
low_rank_range = range(5, 26)  # Rank range from 5 to 25
num_matrices = 10  # Number of matrices to generate in sets A1 and A2
sigma = 0.0001  # Noise level for Gaussian perturbation
num_experiments = 10000  # Number of experiments to perform

# Generate sets A1 (rotation matrices) and A2 (scaling matrices)
A1 = [generate_random_rotation_matrix(matrix_size) for _ in range(num_matrices)]
A2 = [generate_random_scaling_matrix(matrix_size) for _ in range(num_matrices)]
A1 = np.array(A1)
A2 = np.array(A2)

# Initialize lists to store k and r values
r_values = []
k_values = []

# Experiment
for _ in range(num_experiments):
    rank = np.random.choice(low_rank_range)
    I_i = generate_low_rank_diagonal_matrix(matrix_size, rank)
    
    A = A1[np.random.randint(0, len(A1))]
    B = A1[np.random.randint(0, len(A1))]
    C = A2[np.random.randint(0, len(A2))]
    D = A2[np.random.randint(0, len(A2))]
    
    M = A @ B @ I_i @ C @ D
    
    # Add Gaussian noise
    M_prime = M + np.random.normal(0, sigma, M.shape)
    
    # Perform SVD
    U, S, Vt = np.linalg.svd(M_prime)
    
    # Calculate the number of singular values that capture 95% of the total variance
    total_variance = np.sum(S)
    variance_sum = 0
    k = 0
    for singular_value in S:
        variance_sum += singular_value
        k += 1
        if variance_sum / total_variance >= 0.95:
            break
    
    r_values.append(rank)
    k_values.append(int(k * 1.4))  # Scale k by 1.4

# Save data to CSV
df = pd.DataFrame({'r': r_values, 'k': k_values})
df.to_csv('experiment_data.csv', index=False)
print("Data saved to experiment_data.csv")

# Create 2D histogram (heatmap)
heatmap, xedges, yedges = np.histogram2d(r_values, k_values, bins=(30, 30), range=[[0, 30], [0, 30]])

# Plotting with seaborn
plt.figure(figsize=(16, 8))

# Function to set up axes
def setup_axes(ax):
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xlabel('Rank (r) of I_i')
    ax.set_ylabel('k (scaled by 1.4)')
    ax.plot([0, 30], [0, 30], 'r-', label='y=x')  # Updated red line
    ax.legend()

# Original heatmap
ax1 = plt.subplot(1, 2, 1)
sns.heatmap(heatmap, cmap='coolwarm', cbar_kws={'label': 'Frequency'}, xticklabels=5, yticklabels=5, ax=ax1)
setup_axes(ax1)
ax1.set_title('Heatmap of k vs Rank (r)')

# Reversed heatmap
ax2 = plt.subplot(1, 2, 2)
sns.heatmap(heatmap, cmap='coolwarm_r', cbar_kws={'label': 'Frequency'}, xticklabels=5, yticklabels=5, ax=ax2)
setup_axes(ax2)
ax2.set_title('Reversed Heatmap of k vs Rank (r)')

plt.tight_layout()

# Save the figure
plt.savefig('experiment_heatmaps.png', dpi=300, bbox_inches='tight')
print("Figure saved as experiment_heatmaps.png")

plt.show()