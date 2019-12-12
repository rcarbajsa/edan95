from sklearn import metrics, datasets
from sklearn.naive_bayes import GaussianNB
from MNIST import MNISTData
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

def gaussian_nb(mnsit) :

    print('---------------------MNIST_Light---------------------')

    #train_features, test_features, train_labels, test_labels = mnist.get_data()
#
    #mnist.visualize_random()
#
    gnb = GaussianNB()
    #gnb.fit(train_features, train_labels)
    #y_pred = gnb.predict(test_features)
#
    #print("Classification report SKLearn GNB:\n%s\n"
    #  % (metrics.classification_report(test_labels, y_pred)))
    #print("Confusion matrix SKLearn GNB:\n%s" % metrics.confusion_matrix(test_labels, y_pred))
#
    #mnist.visualize_wrong_class(y_pred, 8)
#
    print('---------------------SciKitLearn Digits---------------------')
    
    digits = datasets.load_digits()
    data = digits.data
    #n_samples = len(digits.images)
    #data = digits.images.reshape((n_samples, -1))
    #train_features, test_features, train_labels, test_labels = train_test_split(data, digits.target, test_size=0.3)
#
    #gnb.fit(train_features, train_labels)
    #y_pred = gnb.predict(test_features)
#
    #print("Classification report SKLearn GNB:\n%s\n"
    #  % (metrics.classification_report(test_labels, y_pred)))
    #print("Confusion matrix SKLearn GNB:\n%s" % metrics.confusion_matrix(test_labels, y_pred))
#
    print('---------------------SciKitLearn Digits Summarized---------------------')
    print(data[0])
    
    data_summarized = []
    for image in data:
        image_sum = []
        for pixel in image:
            if pixel < 5:
                image_sum.append(0) #Black
            elif pixel < 10:
                image_sum.append(1) #Gray
            else:
                image_sum.append(2) #White
        data_summarized.append(image_sum)
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    train_features, test_features, train_labels, test_labels = train_test_split(data, digits.target, test_size=0.3)
    gnb.fit(train_features, train_labels)
    y_pred = gnb.predict(test_features)

    print("Classification report SKLearn GNB:\n%s\n"
      % (metrics.classification_report(test_labels, y_pred)))
    print("Confusion matrix SKLearn GNB:\n%s" % metrics.confusion_matrix(test_labels, y_pred))

def ncc():
    print


if __name__ == "__main__":

    mnist = MNISTData('/home/rcarbajsa/Escritorio/edan95_datasets/MNIST_Light/*/*.png')
    gaussian_nb(mnist)


