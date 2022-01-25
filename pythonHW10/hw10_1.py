from load_cifar_10 import load_cifar_10_data
import numpy as np
from skimage.color import rgb2gray
#Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
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

    # converting rgb input images to gray then flattening the input images.
    train_data = rgb2gray(train_data).reshape(len(train_data),-1)
    test_data = rgb2gray(test_data).reshape(len(test_data),-1)
    
    ### fill in any necessary code below to perform the task outlined in README.md document. ###
    
    # create KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)

    # train the model using the training sets
    knn.fit(train_data,train_labels)

    # predict the response for test dataset
    test_pred = knn.predict(test_data)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(test_labels, test_pred))

    # calcuating the confusion matrix
    confusion_matrix = metrics.confusion_matrix(test_labels, test_pred)
    print(confusion_matrix)

    fo = open("hw10_1.txt", "w")
    fo.write("Accuracy: ")
    i = metrics.accuracy_score(test_labels, test_pred)
    fo.write(str(i))
    for i in confusion_matrix:
        fo.write("\n")
        for j in i:
            fo.write(str(j))
            fo.write(" ")   
    fo.close()
