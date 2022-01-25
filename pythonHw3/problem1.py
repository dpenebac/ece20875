from matplotlib.colors import same_color
import numpy as np
import matplotlib.pyplot as plt
#hi

def norm_histogram(hist):
    """
    takes a histogram of counts and creates a histogram of probabilities

    :param hist: a numpy ndarray object
    :return: list
    """

    l = [0] * len(hist)
    t = sum(hist)

    c = 0
    for i in hist:
        l[c] = i / t
        c += 1
    

    return(l)


def compute_j(histo, width):
    """
    takes histogram of counts, uses norm_histogram to convert to probabilties, it then calculates compute_j for one bin width

    :param histo: list 
    :param width: float
    :return: float
    """

    hist = norm_histogram(histo) #list of probabilities

    m = sum(histo)
    w = width
    for i in range(len(hist)):
        hist[i] *= hist[i]
    
    J = 2 / ((m - 1) * w) - (((m + 1) / ((m - 1) * w)) * sum(hist))

    return(J)


def sweep_n(data, minimum, maximum, min_bins, max_bins):
    """
    find the optimal bin
    calculate compute_j for a full sweep [min_bins to max_bins]
    please make sure max_bins is included in your sweep

    :param data: list
    :param minimum: int
    :param maximum: int
    :param min_bins: int
    :param max_bins: int
    :return: list
    """

    #ch = compute_j(plt.hist(data, 5, (lo, hi))[0], (hi - lo) / 5)
    # compute_j(histogram, binwidth)

    optimal = [0] * max_bins

    c = 0
    for i in range(min_bins,max_bins + 1):
        optimal[c] = compute_j(plt.hist(data, i,(minimum, maximum))[0],(maximum - minimum) / i)
        c += 1

    return(optimal)


def find_min(l):
    """
    Generic function that takes a list of numbers and returns the smallest number in that list and its index in the list.
    It will return the optimal value and the index of the optimal value as a tuple.

    :param l: list
    :return: tuple
    """

    smol = 1

    for i in l:
        if i < smol:
            smol = i
    
    index = l.index(smol)

    return((smol, index))


if __name__ == '__main__':
    data = np.loadtxt('input.txt')  # reads data from input.txt
    lo = min(data)
    hi = max(data)
    bin_l = 1
    bin_h = 100
    js = sweep_n(data, lo, hi, bin_l, bin_h)
    """
    the values bin_l and bin_h represent the lower and higher bounds of the range of bins.
    They will change when we test your code and you should be mindful of that.
    """
    print(find_min(js))
