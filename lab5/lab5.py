from __future__ import division
from sklearn import metrics, datasets
from sklearn.naive_bayes import GaussianNB
from MNIST import MNISTData
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import pdb
import numpy as np
import collections

def gaussian_nb(mnsit) :

    print('---------------------MNIST_Light---------------------')

    train_features, test_features, train_labels, test_labels = mnist.get_data()

    mnist.visualize_random()

    gnb = GaussianNB()
    gnb.fit(train_features, train_labels)
    y_pred = gnb.predict(test_features)

    print("Classification report SKLearn GNB:\n%s\n"
      % (metrics.classification_report(test_labels, y_pred)))
    print("Confusion matrix SKLearn GNB:\n%s" % metrics.confusion_matrix(test_labels, y_pred))

    mnist.visualize_wrong_class(y_pred, 8)

    print('---------------------SciKitLearn Digits---------------------')
    
    digits = datasets.load_digits()
    data = digits.data
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    train_features, test_features, train_labels, test_labels = train_test_split(data, digits.target, test_size=0.3)

    gnb.fit(train_features, train_labels)
    y_pred = gnb.predict(test_features)

    print("Classification report SKLearn GNB:\n%s\n"
      % (metrics.classification_report(test_labels, y_pred)))
    print("Confusion matrix SKLearn GNB:\n%s" % metrics.confusion_matrix(test_labels, y_pred))

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
    data_summarized = np.stack(data_summarized)
    train_features, test_features, train_labels, test_labels = train_test_split(data_summarized, digits.target, test_size=0.3)
    gnb.fit(train_features, train_labels)
    y_pred = gnb.predict(test_features)

    print("Classification report SKLearn GNB:\n%s\n"
      % (metrics.classification_report(test_labels, y_pred)))
    print("Confusion matrix SKLearn GNB:\n%s" % metrics.confusion_matrix(test_labels, y_pred))

def ncc(mnist):

    def fit(x, y):

        classes = set(y)
        pixels = {k: np.zeros(len(x[0])) for k in classes}
        class_count = {k: 0 for k in classes}
        for i in range(len(x)):
            for j in range(len(x[i])):
                pixels[y[i]][j] += x[i][j]
            class_count[y[i]] += 1
        
        centroids = []
        for k in pixels:
            centroids.append(pixels[k]/class_count[k])
        return np.array(centroids)

    def predict(x, centroids):
        
        results = []
        for image in x:
            distances = []
            for mean in centroids:
                distances.append(np.linalg.norm(np.subtract(image, mean)))
            results.append(distances.index(min(distances)))
        return results

    print('---------------------MNIST_Light---------------------')
    train_features, test_features, train_labels, test_labels = mnist.get_data()
    centroids = fit(train_features, train_labels)
    y_pred = predict(test_features, centroids)
    
    print("Classification report SKLearn GNB:\n%s\n"
      % (metrics.classification_report(test_labels, y_pred)))
    print("Confusion matrix SKLearn GNB:\n%s" % metrics.confusion_matrix(test_labels, y_pred))

    print('---------------------SciKitLearn Digits---------------------')
    
    digits = datasets.load_digits()
    data = digits.data
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    train_features, test_features, train_labels, test_labels = train_test_split(data, digits.target, test_size=0.3)

    centroids = fit(train_features, train_labels)
    y_pred = predict(test_features, centroids)

    print("Classification report SKLearn GNB:\n%s\n"
      % (metrics.classification_report(test_labels, y_pred)))
    print("Confusion matrix SKLearn GNB:\n%s" % metrics.confusion_matrix(test_labels, y_pred))

    print('---------------------SciKitLearn Digits Summarized---------------------')
    
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
    data_summarized = np.stack(data_summarized)
    train_features, test_features, train_labels, test_labels = train_test_split(data_summarized, digits.target, test_size=0.3)
    centroids = fit(train_features, train_labels)
    y_pred = predict(test_features, centroids)

    print("Classification report SKLearn GNB:\n%s\n"
      % (metrics.classification_report(test_labels, y_pred)))
    print("Confusion matrix SKLearn GNB:\n%s" % metrics.confusion_matrix(test_labels, y_pred))

def nbc():

    def classes_prob(labels, classes):
        #pdb.set_trace()
        result = {k: 0.0 for k in classes}
        for label in classes:
            result[label] = float(collections.Counter(labels)[label]/len(labels))
        return result

    def cpt(features, train_labels):
        #CPTijk = P(Pixeli = vj | Number = k)
        features_class = dict()
        classes = []
        for i, feauture in enumerate(features):
            if train_labels[i] not in classes:
                classes.append(train_labels[i])
                features_class[train_labels[i]] = list()
            features_class[train_labels[i]].append(feauture)
        return features_class

    def count(col, val):
        
        count = 0.0
        for v in col:
            if v == val:
                count +=1
        return count
    
    def predict(train_features, cpt_table, test_features, classes_prob):

        prediction=[]
        for feature in test_features:
            probs = np.zeros(len(cpt_table))
            for i,label in enumerate(cpt_table):
                temp = list()
                features = np.stack(cpt_table[label])
                for i, pixel in enumerate(feature):
                    count_value_total = count(train_features[:, i], pixel)
                    count_value_per_class = count(features[:, i], pixel)
                    if count_value_total == 0:
                        temp.append(0.0)
                    else:
                        temp.append(count_value_per_class/ count_value_total)
                probs[label] = np.prod(temp) * classes_prob[label]
            prediction.append(np.argmax(probs, axis=0))
        return prediction
        
    print('---------------------SciKitLearn Digits---------------------')
    digits = datasets.load_digits()
    data = digits.data
    classes = digits.target_names
    train_features, test_features, train_labels, test_labels = train_test_split(data, digits.target, test_size=0.3)
    print(train_features[:, 1])
    class_prob = classes_prob(train_labels, classes)
    print(class_prob)
    cpt_table = cpt(train_features, train_labels)
    y_pred = predict(train_features, cpt_table, test_features, class_prob)
    print("Classification report SKLearn GNB:\n%s\n"
      % (metrics.classification_report(test_labels, y_pred)))
    print("Confusion matrix SKLearn GNB:\n%s" % metrics.confusion_matrix(test_labels, y_pred))
    print('---------------------SciKitLearn Digits Summarized---------------------')
    
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
    data_summarized = np.stack(data_summarized)
    train_features, test_features, train_labels, test_labels = train_test_split(data_summarized, digits.target, test_size=0.3)
    class_prob = classes_prob(train_labels, classes)
    print(class_prob)
    cpt_table = cpt(train_features, train_labels)
    y_pred = predict(train_features, cpt_table, test_features, class_prob)
    print("Classification report SKLearn GNB:\n%s\n"
      % (metrics.classification_report(test_labels, y_pred)))
    print("Confusion matrix SKLearn GNB:\n%s" % metrics.confusion_matrix(test_labels, y_pred))
    
def gnb(mnist):

    def prob(x, mean, var):
        #pdb.set_trace()
        if var == 0:
            if x == mean:
                return 1.0
            else:
                return 0.01
        else:
            return (1 / np.sqrt(2*np.pi*var)) * np.exp(-(1/(2*var))*(x-mean)**2)
    def cpt(features, train_labels):
        #CPTijk = P(Pixeli = vj | Number = k)
        features_class = {}
        classes = []
        for i, feauture in enumerate(features):
            if train_labels[i] not in classes:
                classes.append(train_labels[i])
                features_class[train_labels[i]] = []
            features_class[train_labels[i]].append(feauture)
        class_prob = {k: 0.0 for k in classes} 
        cpt_table = {k: [] for k in classes}
        for class_ in classes:
            feauture = np.stack(features_class[class_])
            for i in range(feauture.shape[1]):
                mean = np.mean(feauture[:, i])
                var = np.var(feauture[:, i])
                cpt_table[class_].append((mean, var))
            class_prob[class_] = prob(class_, np.mean(train_labels), np.var(train_labels))
        return cpt_table, class_prob

    def predict(cpt, class_prob, test_features):
        
        prediction = []
        for feature in test_features:
            label_probs = np.zeros(len(class_prob))
            for class_ in cpt:
                probs = []
                for i in range(len(feature)):
                    probs.append(prob(feature[i], cpt[class_][i][0], cpt[class_][i][1]))
                label_probs[class_] = class_prob[class_] * np.prod(probs)
            prediction.append(np.argmax(label_probs, axis=0))
        return prediction

    print('---------------------SciKitLearn Digits---------------------')
    digits = datasets.load_digits()
    data = digits.data
    classes = digits.target_names
    train_features, test_features, train_labels, test_labels = train_test_split(data, digits.target, test_size=0.3)
    cpt_table, class_prob = cpt(train_features, train_labels)
    y_pred = predict(cpt_table, class_prob, test_features)
    print("Classification report SKLearn GNB:\n%s\n"
      % (metrics.classification_report(test_labels, y_pred)))
    print("Confusion matrix SKLearn GNB:\n%s" % metrics.confusion_matrix(test_labels, y_pred))
    print('---------------------SciKitLearn Digits Summarized---------------------')
    
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
    data_summarized = np.stack(data_summarized)
    train_features, test_features, train_labels, test_labels = train_test_split(data_summarized, digits.target, test_size=0.3)
    cpt_table, class_prob = cpt(train_features, train_labels)
    y_pred = predict(cpt_table, class_prob, test_features)
    print("Classification report SKLearn GNB:\n%s\n"
      % (metrics.classification_report(test_labels, y_pred)))
    print("Confusion matrix SKLearn GNB:\n%s" % metrics.confusion_matrix(test_labels, y_pred))
    
    print('---------------------MNIST_Light---------------------')
    train_features, test_features, train_labels, test_labels = mnist.get_data()
    cpt_table, class_prob = cpt(train_features, train_labels)
    y_pred = predict(cpt_table, class_prob, test_features)
    print("Classification report SKLearn GNB:\n%s\n"
      % (metrics.classification_report(test_labels, y_pred)))
    print("Confusion matrix SKLearn GNB:\n%s" % metrics.confusion_matrix(test_labels, y_pred))
    
if __name__ == "__main__":

    mnist = MNISTData('/home/rcarbajsa/Escritorio/edan95_datasets/MNIST_Light/*/*.png')
    #gaussian_nb(mnist)
    #ncc(mnist)
    #nbc()
    gnb(mnist)


