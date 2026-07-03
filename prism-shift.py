## Particle Swarm Simulation with a Shifted Reference Frame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

## Simulation Parameters
num_particles = 50
dimensions = 2  # 2D for easier visualization
search_space_min = -10
search_space_max = 10
max_iterations = 100

## Particle Swarm Optimization coefficients
w = 0.5   # Inertia weight
c1 = 1.5  # Cognitive coefficient (personal best)
c2 = 1.5  # Social coefficient (global best)

## Perturbation parameter for visual reference frame (standard deviation of Gaussian noise)
perturbation_strength = 0.5

## Prism Illusion Parameter
## This vector represents the constant shift in the perceived global best position
## e.g., if the true gbest is at (0,0), particles will perceive it at (0,0) + prism_shift_vector
prism_shift_vector = np.array([3.0, 2.0]) # Example: shift perceived target by 3 units in x, 2 in y

print("Simulation parameters defined, including prism shift.")

## Fitness Function for Particle Swarm Optimization
## Fitness Function
## A simple 2D quadratic function (e.g., Sphere function) with minimum at (0,0)
def fitness_function(position):
    return np.sum(position**2, axis=-1)

## Particle Initialization
## Initialize positions and velocities randomly within the search space
positions = np.random.uniform(search_space_min, search_space_max, (num_particles, dimensions))
velocities = np.random.uniform(-1, 1, (num_particles, dimensions))

## Initialize personal best (pbest) positions and fitnesses
pbest_positions = positions.copy()
pbest_fitness = fitness_function(positions)

## Initialize global best (gbest) position and fitness
gbest_index = np.argmin(pbest_fitness)
gbest_position = pbest_positions[gbest_index].copy()
gbest_fitness = pbest_fitness[gbest_index]

print("Particles initialized and fitness function defined.")

## Particle Swarm Simulation
history = [] # To store positions for visualization

for iteration in range(max_iterations):
    r1 = np.random.rand(num_particles, dimensions)
    r2 = np.random.rand(num_particles, dimensions)

    ## Perturb the Global Best (Visual Reference Frame) with Prism Effect
    ## Each particle perceives the global best position as shifted by 'prism_shift_vector'
    ## and then with additional Gaussian noise.
    effective_gbest_for_perception = gbest_position + prism_shift_vector
    gbest_position_perturbed = effective_gbest_for_perception + np.random.normal(0, perturbation_strength, (num_particles, dimensions))

    ## Update velocities
    velocities = (w * velocities) + \
                 (c1 * r1 * (pbest_positions - positions)) + \
                 (c2 * r2 * (gbest_position_perturbed - positions))

    ## Update positions
    positions = positions + velocities

    ## Clamp positions to search space limits
    positions = np.clip(positions, search_space_min, search_space_max)

    ## Evaluate current fitness
    current_fitness = fitness_function(positions)

    ## Update personal best (pbest)
    mask = current_fitness < pbest_fitness
    pbest_positions[mask] = positions[mask].copy()
    pbest_fitness[mask] = current_fitness[mask]

    ## Update global best (gbest)
    ## The actual global best is still based on the true fitness function, not the perceived one.
    if np.min(current_fitness) < gbest_fitness:
        gbest_index = np.argmin(current_fitness)
        gbest_position = positions[gbest_index].copy()
        gbest_fitness = current_fitness[gbest_index]

    history.append(positions.copy())

    if (iteration + 1) % 10 == 0:
        print(f"Iteration {iteration + 1}/{max_iterations}: Best fitness = {gbest_fitness:.4f}")

print("Simulation complete. Final Actual Global Best Position:", gbest_position, "Fitness:", gbest_fitness)

## Visualizing Particle Swarm Motion
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(search_space_min, search_space_max)
ax.set_ylim(search_space_min, search_space_max)
ax.set_title('Particle Swarm with Prism Illusion (Shifted Perception)')
ax.set_xlabel('X-coordinate')
ax.set_ylabel('Y-coordinate')

## Plot the actual global best (true target) position
ax.plot(0, 0, 'r*', markersize=15, label='Actual Target (True Global Best)')

## Plot the perceived global best (illusory target) position
## This is where particles *think* the target is due to the prism effect
illusory_target = np.array([0,0]) + prism_shift_vector # Calculate the fixed illusory target for context
ax.plot(illusory_target[0], illusory_target[1], 'k*', markersize=12, label='Illusory Target (Perceived Global Best)')

## Initialize particle scatter plot
scat = ax.scatter(history[0][:, 0], history[0][:, 1], color='blue', alpha=0.7, label='Particles')

def update(frame):
    scat.set_offsets(history[frame])
    return scat,

ani = FuncAnimation(fig, update, frames=len(history), blit=True)
plt.legend()
plt.show()

print("Animation of particle swarm movement with prism illusion displayed.")
