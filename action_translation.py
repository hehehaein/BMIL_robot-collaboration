import numpy as np
def translate_action(action):
    array = np.zeros(4, dtype=int)
    for i in range(4):
        array[i] = action % 3
        action = action / 3
    for i in range(4):
        array[i] = array[i] - 1
    return array
print(translate_action(62))