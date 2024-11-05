import numpy as np

def add_noise(variable_value, standard_deviation, noise_on):
    random_variation = np.random.normal(0, standard_deviation) * noise_on
    variable_with_noise = variable_value + random_variation
    return variable_with_noise