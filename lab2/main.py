import ToyData as td
import ID3
import pdb
import numpy as np
from sklearn import tree, metrics, datasets, preprocessing
import matplotlib.pyplot as plt
import graphviz

def main():
    digits = datasets.load_digits()
    data_digits = digits.data
    target = digits.target
    sep = int(len(data_digits) * 0.7)
    train_data = data_digits[:sep]
    test_data = data_digits[sep:]
    train_target = target[:sep]
    test_target = target[sep:]
    
    print("------------------- SciKitLearn Tree -------------------")
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_data,train_target)
    dot_data = tree.export_graphviz(clf)
    graph = graphviz.Source(dot_data)
    graph.render('ScikitLearnTree')
    result = clf.predict(test_data)
    print(result)
    report = metrics.classification_report(test_target, result)
    print(report)
    report = metrics.confusion_matrix(test_target, result, labels=digits.target_names)
    print(report)

    print("------------------- SciKitLearn Tree2 -------------------")
    clf = tree.DecisionTreeClassifier(min_samples_leaf=3)
    clf = clf.fit(train_data,train_target)
    dot_data = tree.export_graphviz(clf)
    graph = graphviz.Source(dot_data)
    graph.render('ScikitLearnTree')
    result = clf.predict(test_data)
    print(result)
    report = metrics.classification_report(test_target, result)
    print(report)
    report = metrics.confusion_matrix(test_target, result, labels=digits.target_names)
    print(report)

    print("------------------- My tree with ToyData -------------------")
    attributes, classes, data, target, data2, target2 = td.ToyData().get_data()
    id3 = ID3.ID3DecisionTreeClassifier()
    myTree = id3.fit(data, target, attributes, classes)
    plot = id3.make_dot_data()
    plot.render("testTree")
    #pdb.set_trace()
    result = id3.predict(data2)
    print("Predicted" + str(result))
    report = metrics.classification_report(target2, result)
    print(report)
    report = metrics.confusion_matrix(target2, result)
    print(report)

    print("------------------- My tree with digits -------------------")
    id3 = ID3.ID3DecisionTreeClassifier()
    att_values = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    attributes = {}
    for i in range(64):
        attributes[i] = att_values
    myTree = id3.fit(train_data, train_target, attributes, classes)
    plot = id3.make_dot_data()
    plot.render("testTree")
    #pdb.set_trace()
    result = id3.predict(test_data)
    print("Predicted" + str(result))
    report = metrics.classification_report(test_target, result)
    print(report)
    report = metrics.confusion_matrix(test_target, result)
    print(report)

    print("------------------- My tree with d-g-l -------------------")
    id3 = ID3.ID3DecisionTreeClassifier()
    att_values = ['d','g','l']
    attributes = {}
    for i in range(64):
        attributes[i] = att_values
    data = []
    for item in data_digits:
        row = []
        for d in item:
            if d < 5:
                row.append('d')
            elif d < 10:
                row.append('g')
            else:
                row.append('l')
        data.append(row)
    train_data = data[:sep]
    test_data = data[sep:]
    myTree = id3.fit(train_data, train_target, attributes, classes)
    plot = id3.make_dot_data()
    plot.render("testTree")
    #pdb.set_trace()
    result = id3.predict(test_data)
    print("Predicted" + str(result))
    report = metrics.classification_report(test_target, result)
    print(report)
    report = metrics.confusion_matrix(test_target, result)
    print(report)
    
    

if __name__ == "__main__": main()