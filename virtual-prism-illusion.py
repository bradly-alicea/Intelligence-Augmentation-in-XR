## Defining the simulation parameters

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## Simulation Parameters
## Actual target position in 3D space
actual_target = np.array([2, 3, 1])

## Prism shift vector (how much the perception is shifted)
prism_shift = np.array([3.0, -2.0, 1.0]) # Increased shift vector

## Perceived target position
perceived_target = actual_target + prism_shift

## Starting position of the 'hand' or 'pointer'
hand_start_pos = np.array([0, 0, 0])

## Number of steps in the adaptation process for each path
num_adaptation_steps = 50

## Number of individual paths to simulate
num_paths = 100

## Adaptation rate multiplier: >1 for faster adaptation, <1 for slower
adaptation_rate_multiplier = 1.0 # Default: normal adaptation rate

print("VR Prism Illusion parameters defined with larger prism shift and adaptation rate multiplier.")

## Simulating Hand Movements and Visual Adaptation

def generate_single_hand_path(actual_target, perceived_target, hand_start_pos, num_adaptation_steps, adaptation_rate_multiplier):
    hand_path = []
    current_hand_pos = hand_start_pos.copy().astype(float) # Ensure current_hand_pos is float
    hand_path.append(current_hand_pos.copy())

    ## Initial movement towards the perceived target (misreaching)
    ## The 'hand' initially aims directly at where it *sees* the target.
    for i in range(num_adaptation_steps // 2): # First half, primarily guided by perceived target
        ## Calculate direction vector, handle division by zero if target is at current position
        direction_vector = perceived_target - current_hand_pos
        norm_direction = np.linalg.norm(direction_vector)
        if norm_direction == 0:
            direction = np.array([0.0, 0.0, 0.0]) # Or break, if target reached
        else:
            direction = direction_vector / norm_direction
        current_hand_pos += direction * 0.1 # Move a small step
        hand_path.append(current_hand_pos.copy())

    ## Adaptation phase: the 'hand' starts correcting towards the actual target
    ## This is a simplified model of motor adaptation.
    for i in range(num_adaptation_steps // 2):
        ## Gradually shift aim from perceived_target towards actual_target
        ## The 'adaptation_factor' increases over time from 0 to 1
        ## Now scaled by adaptation_rate_multiplier
        adaptation_factor = min(1.0, ((i + 1) / (num_adaptation_steps // 2)) * adaptation_rate_multiplier)
        
        ## Blended target: initially perceived, then gradually actual
        blended_target = (1 - adaptation_factor) * perceived_target + adaptation_factor * actual_target
        
        ## Calculate direction vector, handle division by zero if target is at current position
        direction_vector = blended_target - current_hand_pos
        norm_direction = np.linalg.norm(direction_vector)
        if norm_direction == 0:
            direction = np.array([0.0, 0.0, 0.0]) # Or break, if target reached
        else:
            direction = direction_vector / norm_direction
        current_hand_pos += direction * 0.1 # Move a small step
        hand_path.append(current_hand_pos.copy())

    return np.array(hand_path)
  
all_hand_paths = []
# Introduce a small random perturbation to hand_start_pos for each trial
start_pos_perturbation_strength = 0.1 # Small perturbation

for _ in range(num_paths):
    ## Apply a random perturbation to the initial hand position for each trial
    perturbed_hand_start_pos = hand_start_pos + np.random.uniform(-start_pos_perturbation_strength, start_pos_perturbation_strength, 3)
    all_hand_paths.append(generate_single_hand_path(actual_target, perceived_target, perturbed_hand_start_pos, num_adaptation_steps, adaptation_rate_multiplier))

## Plotting the Simulation

fig = plt.figure(figsize=(12, 10)) # Slightly larger figure for multiple paths
ax = fig.add_subplot(111, projection='3d')

## Plot the actual target
ax.scatter(*actual_target, color='green', marker='o', s=250, label='Actual Target (True Position)', edgecolors='black', linewidth=1)
ax.text(actual_target[0] + 0.1, actual_target[1] + 0.1, actual_target[2], 'Actual Target', color='green', fontsize=10)

## Plot the perceived target
ax.scatter(*perceived_target, color='red', marker='X', s=250, label='Perceived Target (Prism Effect)', edgecolors='black', linewidth=1)
ax.text(perceived_target[0] + 0.1, perceived_target[1] + 0.1, perceived_target[2], 'Perceived Target', color='red', fontsize=10)

## Plot the hand's paths
for i, hand_path in enumerate(all_hand_paths):
    ## Plot each path with a slightly transparent blue to see overlaps
    ax.plot(hand_path[:, 0], hand_path[:, 1], hand_path[:, 2], color='blue', linestyle='-', linewidth=0.8, alpha=0.1)
    if i == 0: # Only label one path for the legend to avoid clutter
        ax.plot(hand_path[:, 0], hand_path[:, 1], hand_path[:, 2], color='blue', linestyle='-', linewidth=0.8, alpha=0.1, label='Hand Paths (Initial Misreach then Adaptation)')

## Add a single representation of the mean start position (original hand_start_pos)
ax.scatter(*hand_start_pos, color='cyan', marker='^', s=150, label='Mean Hand Start Position', edgecolors='black', linewidth=0.8)

## Add a cluster of final positions to show the adapted region
final_positions = np.array([path[-1] for path in all_hand_paths])
ax.scatter(final_positions[:, 0], final_positions[:, 1], final_positions[:, 2],
           color='purple', marker='*', s=100, alpha=0.5, label='Final Adapted Positions (Cluster)')

## Set labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title(f'Virtual Reality Prism Illusion Simulation ({num_paths} Separate Trials, Rate: {adaptation_rate_multiplier})')

## Set limits to ensure all elements are visible (recalculate based on all paths)
## Need to include all initial perturbed positions in the limits calculation
all_initial_positions = np.array([path[0] for path in all_hand_paths])
all_coords_for_limits = np.vstack([actual_target, perceived_target, all_initial_positions])
for path in all_hand_paths:
    all_coords_for_limits = np.vstack([all_coords_for_limits, path])

max_range = np.array([all_coords_for_limits[:,0].max()-all_coords_for_limits[:,0].min(),
                      all_coords_for_limits[:,1].max()-all_coords_for_limits[:,1].min(),
                      all_coords_for_limits[:,2].max()-all_coords_for_limits[:,2].min()]).max()

mid_x = (all_coords_for_limits[:,0].max()+all_coords_for_limits[:,0].min()) * 0.5
mid_y = (all_coords_for_limits[:,1].max()+all_coords_for_limits[:,1].min()) * 0.5
mid_z = (all_coords_for_limits[:,2].max()+all_coords_for_limits[:,2].min()) * 0.5

## Add a small buffer to the limits
buffer = max_range * 0.1
ax.set_xlim(mid_x - max_range/2 - buffer, mid_x + max_range/2 + buffer)
ax.set_ylim(mid_y - max_range/2 - buffer, mid_y + max_range/2 + buffer)
ax.set_zlim(mid_z - max_range/2 - buffer, mid_z + max_range/2 + buffer)

plt.legend(loc='lower left') # Adjust legend position
plt.grid(True)
plt.show()

print(f"VR Prism Illusion simulation visualized with {num_paths} separate trials. Observe how the paths initially cluster towards the red 'Perceived Target' and then adapt to converge around the green 'Actual Target'.")
print(f"Generated {num_paths} hand paths with initial misreach and adaptation, each starting from a slightly perturbed position.")
