from load_cifar_10 import load_cifar_10_data
import numpy as np
import matplotlib.pyplot as plt
######################################################################
def load_cifar_3(train_data, train_labels, test_data, test_labels):

    zipped_train = list(zip(train_data,train_labels))
    zipped_test = list(zip(test_data, test_labels))

    filtering_criteria = lambda x: x[1]<3
    filtered_train = list(filter(filtering_criteria, zipped_train))
    filtered_test = list(filter(filtering_criteria, zipped_test))

    filtered_trained_data = np.array([i for i,_ in filtered_train])
    filtered_trained_labels = np.array([j for _,j in filtered_train])

    filtered_test_data = np.array([i for i,_ in filtered_test])
    filtered_test_labels = np.array([j for _,j in filtered_test])

    return filtered_trained_data, filtered_trained_labels, filtered_test_data, filtered_test_labels

######################################################################
if __name__ == "__main__":

    cifar_10_dir = 'cifar-10-batches-py'

    # loading CIFAR-10 dataset
    train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = \
            load_cifar_10_data(cifar_10_dir)

    # filtering CIFAR-10 to make it CIFAR-3
    train_data, train_labels, test_data, test_labels = load_cifar_3(train_data, train_labels, test_data, test_labels)

    # display some random training images in a 25x25 grid
    num_plot = 5
    f, ax = plt.subplots(num_plot, num_plot)
    for m in range(num_plot):
        for n in range(num_plot):
            idx = np.random.randint(0, train_data.shape[0])
            ax[m, n].imshow(train_data[idx])
            ax[m, n].get_xaxis().set_visible(False)
            ax[m, n].get_yaxis().set_visible(False)
    f.subplots_adjust(hspace=0.1)
    f.subplots_adjust(wspace=0)
    plt.show()

    # You can do the same for test images.
    # You can even show more/less number of images in your plot.
    # Explore the images in the dataset and see what they look like.
