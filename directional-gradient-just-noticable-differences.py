## Noise matrix (500x500 matrix of Gaussian noise)
import numpy as np
noise_matrix = np.random.normal(size=(500, 500))

## Display the shape of the matrix and a small portion of it
print(f"Shape of the noise matrix: {noise_matrix.shape}")
print("First 5x5 submatrix of Gaussian noise:")
print(noise_matrix[:5, :5])

## Normalize the noise_matrix to a range between 0 and 1
min_val = noise_matrix.min()
max_val = noise_matrix.max()
normalized_noise_matrix = (noise_matrix - min_val) / (max_val - min_val)

## Display the shape and a small portion of the normalized matrix
print(f"Shape of the normalized noise matrix: {normalized_noise_matrix.shape}")
print("First 5x5 submatrix of normalized Gaussian noise:")
print(normalized_noise_matrix[:5, :5])

## Directional intensity gradient.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure noise_matrix is defined. If not, create a default one (fallback).
# This makes the cell more robust if previous cells were not run or their state was lost.
if 'noise_matrix' not in locals() and 'noise_matrix' not in globals():
    print("Warning: 'noise_matrix' not found. Creating a default 500x500 matrix for robustness.")
    noise_matrix = np.random.normal(size=(500, 500))

## Always re-normalize the noise_matrix to a range between 0 and 1 within this cell
## to ensure 'normalized_noise_matrix' is consistently defined here, regardless of prior executions.
min_val = noise_matrix.min()
max_val = noise_matrix.max()

## Handle case where min_val == max_val to avoid division by zero
if min_val == max_val:
    normalized_noise_matrix = np.zeros_like(noise_matrix) # Or handle as an error if appropriate
else:
    normalized_noise_matrix = (noise_matrix - min_val) / (max_val - min_val)

## N is derived from the shape of the normalized noise matrix
N = normalized_noise_matrix.shape[0]

## Create a linear directional intensity gradient (e.g., left to right)
## The gradient will range from 0 to 1 across the columns
gradient_1d = np.linspace(0, 1, N)
directional_gradient = np.tile(gradient_1d, (N, 1))

## Combine the gradient with the normalized Gaussian noise
## We'll blend them, giving more weight to the gradient for a clear directional intensity.
## The normalized_noise_matrix ranges from 0 to 1.

## Blending with 70% gradient and 30% noise
blended_matrix = (directional_gradient * 0.7) + (normalized_noise_matrix * 0.3)

## Re-normalize the blended matrix to ensure its values are between 0 and 1
min_blended = blended_matrix.min()
max_blended = blended_matrix.max()
scaled_blended_matrix = (blended_matrix - min_blended) / (max_blended - min_blended)

print("Gradient blended with normalized noise and re-normalized.")
## print("First 5x5 submatrix of scaled blended matrix:")
## print(scaled_blended_matrix[:5, :5])

print("Directional gradient created and normalized_noise_matrix ensured.")
## print("First 5x5 submatrix of directional gradient:")
## print(directional_gradient[:5, :5])

## 3. Apply 'just-noticeable differences' by quantizing the intensity levels
## This will create discrete steps in the intensity, making changes more 'noticeable'.

num_jnd_levels = 16  # Changed to 16 distinct intensity levels for JND

## Create bins for quantization over the 0-1 range
bins = np.linspace(0, 1, num_jnd_levels + 1)

## Quantize the scaled_blended_matrix values into these bins
## np.digitize returns the index of the bin each value falls into (1-indexed by default)
quantized_matrix_indices = np.digitize(scaled_blended_matrix, bins)

## To make the values suitable for visualization and represent actual intensity levels,
## map the bin indices to a scaled range (e.g., 0 to num_jnd_levels-1 or 0 to 1).
## We'll use the midpoints of the bins for a representative value.
jnd_levels_values = np.array([(bins[i] + bins[i+1])/2 for i in range(num_jnd_levels)])

## Adjust indices to be 0-based for mapping, handling edge case where value == bins[-1]
quantized_matrix_indices = np.clip(quantized_matrix_indices - 1, 0, num_jnd_levels - 1)

## Create the final quantized matrix using the representative JND values
jnd_matrix = jnd_levels_values[quantized_matrix_indices]

print(f"Matrix quantized into {num_jnd_levels} just-noticeable difference levels.")
## print("First 5x5 submatrix of JND matrix:")
## print(jnd_matrix[:5, :5])

## Visualize the resulting JND matrix using plt.imshow
plt.figure(figsize=(10, 8))
plt.imshow(jnd_matrix, cmap='viridis', origin='lower', extent=[0, N, 0, N])
plt.title(f'Directional Intensity Gradient with {num_jnd_levels} Just-Noticeable Difference Levels')
plt.xlabel('Gradient Direction')
plt.ylabel('Matrix Row')
plt.colorbar(label='JND Level') # Add a color bar for better interpretation
plt.show()

## Comparing Gaussian noise vs. directional intensity gradient
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

## Plotting the original Gaussian noise matrix
sns.heatmap(noise_matrix, cmap='viridis', cbar=True, ax=axes[0],
            square=True, xticklabels=False, yticklabels=False)
axes[0].set_title('Original Gaussian Noise Matrix')
axes[0].set_xlabel('Column Index')
axes[0].set_ylabel('Row Index')

## Plotting the JND matrix
im = axes[1].imshow(jnd_matrix, cmap='viridis', origin='lower', extent=[0, N, 0, N])
axes[1].set_title(f'Directional Intensity Gradient with {num_jnd_levels} JND Levels')
axes[1].set_xlabel('Gradient Direction')
axes[1].set_ylabel('Matrix Row')
fig.colorbar(im, ax=axes[1], label='JND Level')

plt.tight_layout()
plt.show()
