import ToyData as td
import ID3
import pdb
import numpy as np
from sklearn import tree, metrics, datasets, preprocessing
import matplotlib.pyplot as plt


def main():
    attributes, classes, data, target, data2, target2 = td.ToyData().get_data()
    enc = preprocessing.LabelEncoder()
    labels = ["y", "g", "b","s", "l","r", "i"]
    enc.fit(labels)
    data_transform, data2_transform = [], []
    for item in data:
        data_transform.append(enc.transform(item))
    print(data_transform)
    for item in data2:
        print(item)
        data2_transform.append(enc.transform(item))
    print(data2_transform)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(data_transform,target)
    #plt.figure()
    tree.plot_tree(clf)
    #plt.show()
    result = clf.predict(data2_transform)
    print(result)
    report = metrics.classification_report(target2, result)
    print(report)
    report = metrics.confusion_matrix(target2, result)
    print(report)
    id3 = ID3.ID3DecisionTreeClassifier()
    myTree = id3.fit(data, target, attributes, classes)
    print(myTree)
    plot = id3.make_dot_data()
    plot.render("testTree")
    #pdb.set_trace()
    result = id3.predict(data2)
    print(result)
    report = metrics.classification_report(target2, result)
    print(report)
    report = metrics.confusion_matrix(target2, result)
    print(report)
    

if __name__ == "__main__": main()