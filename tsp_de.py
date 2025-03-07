import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


np.random.seed(42)
num_cities = 20
cities = np.random.rand(num_cities, 2) * 100 

distance_matrix = squareform(pdist(cities))

pop_size = 20
generations = 1000
F = 0.7 
CR = 0.9

population = np.array([np.random.permutation(num_cities) for _ in range(pop_size)])

def tsp_cost(order):
    total_distance = sum(distance_matrix[order[i], order[i + 1]] for i in range(num_cities - 1))
    total_distance += distance_matrix[order[-1], order[0]]
    return total_distance


for gen in range(generations):
    new_population = []
    for i in range(pop_size):
       
        a, b, c = population[np.random.choice(pop_size, 3, replace=False)]
        mutant = np.array(a)

        diff = set(b) - set(c)
        if len(diff) > 0:
            idx = np.random.choice(list(diff))
            mutant[np.where(mutant == idx)] = c[np.where(c == idx)]

      
        crossover_mask = np.random.rand(num_cities) < CR
        trial = np.where(crossover_mask, mutant, population[i])

        unique, counts = np.unique(trial, return_counts=True)
        if any(counts > 1):
            missing = set(range(num_cities)) - set(trial)
            for j, count in enumerate(counts):
                if count > 1:
                    trial[np.where(trial == unique[j])[0][1:]] = missing.pop()

        if tsp_cost(trial) < tsp_cost(population[i]):
            new_population.append(trial)
        else:
            new_population.append(population[i])

    population = np.array(new_population)

best_order = min(population, key=tsp_cost)

plt.figure(figsize=(6, 6))
plt.scatter(cities[:, 0], cities[:, 1], c='red', marker='o', label="Cities")


for i in range(num_cities - 1):
    plt.plot([cities[best_order[i], 0], cities[best_order[i + 1], 0]],
             [cities[best_order[i], 1], cities[best_order[i + 1], 1]], 'b-')

plt.plot([cities[best_order[-1], 0], cities[best_order[0], 0]],
         [cities[best_order[-1], 1], cities[best_order[0], 1]], 'b-')

plt.title(f"Total Distance: {tsp_cost(best_order):.2f}")
plt.legend()
plt.show()
