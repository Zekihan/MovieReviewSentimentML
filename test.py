import numpy as np

def shuffle_arrays(arrays, set_seed=-1):
    assert all(len(arr) == len(arrays[0]) for arr in arrays)
    seed = np.random.randint(0, 2**(32 - 1) - 1) if set_seed < 0 else set_seed

    for arr in arrays:
        rstate = np.random.RandomState(seed)
        rstate.shuffle(arr)

if __name__ == "__main__":
    data = [[1], [2], [3], [4], [5], [6], [7], [8]]
    labels = [1, 2, 3, 4, 5, 6, 7, 8]

    shuffle_arrays([data,labels])
    print(a)
