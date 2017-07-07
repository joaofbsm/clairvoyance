import sys
import numpy as np

def slice_dataset(dataset_name, slice_size):
    """Slice dataset according to slice_size index.
    
    Arguments:
        dataset_name -- String name of the dataset without the extention.
        slice_size -- Integer size of slice.
    """
    dataset = np.loadtxt(dataset_name + ".csv", delimiter=",")
    sliced = dataset[0:slice_size,]

    np.savetxt(dataset_name + str(slice_size) + ".csv", sliced, delimiter=",", fmt="%.5g")


def main(args):
    dataset_name = args[0]
    slice_size = int(args[1])
    slice_dataset(dataset_name, slice_size)


if __name__ == "__main__":
    main(sys.argv[1:])