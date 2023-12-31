import numpy as np


def billsfit(pop):
    maxSize = 500
    # sizes = np.array([193.71,60.15,89.08,88.98,15.39,238.14,68.78,107.47,119.66,183.70])

    sizes = np.array(
        [109.60, 125.48, 52.16, 195.55, 58.67, 61.87, 92.95, 93.14, 155.05, 110.89, 13.34, 132.49, 194.03, 121.29,
         179.33, 139.02, 198.78, 192.57, 81.66, 128.90])

    # Penalises the fitness by twice the amount it goes over $500
    fitness = np.sum(sizes * pop, axis=1)
    fitness = np.where(fitness > maxSize, 500 - 2 * (fitness - maxSize), fitness)

    return fitness

