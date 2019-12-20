from __future__ import division
from math import log
from collections import Counter, OrderedDict
from graphviz import Digraph
from sklearn import tree, metrics, datasets, preprocessing
import pdb
import ToyData as td

def most_common_class(target):
    if not target:
        return '+'
    else:
        return max(set(target), key=target.count)
def main():

    digits = datasets.load_digits()
    data_digits = digits.data
    target = digits.target
    sep = int(len(data_digits) * 0.7)
    train_data = data_digits[:sep]
    test_data = data_digits[sep:]
    train_target = target[:sep]
    test_target = target[sep:]
    classes = digits.target_names
    id3 = ID3DecisionTreeClassifier(0)
    att_values = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    attributes = {}
    for i in range(64):
        attributes[i] = att_values
    myTree = id3.fit(train_data, train_target, attributes, classes)
    plot = id3.make_dot_data()
    plot.render("testTree_digits")
    #pdb.set_trace()
    result = id3.predict(test_data)
    print("Predicted" + str(result))
    report = metrics.classification_report(test_target, result)
    print(report)
    report = metrics.confusion_matrix(test_target, result)
    print(report)



class ID3DecisionTreeClassifier:

    def __init__(self, toy, minSamplesLeaf=1, minSamplesSplit=2):

        self.__nodeCounter = 0

        # the graph to visualise the tree
        self.__dot = Digraph(comment='The Decision Tree')
        self.__root = None
        # suggested attributes of the classifier to handle training parameters
        self.__minSamplesLeaf = minSamplesLeaf
        self.__minSamplesSplit = minSamplesSplit
        self.nodes = []
        self.toy = toy

    # Create a new node in the tree with the suggested attributes for the visualisation.
    # It can later be added to the graph with the respective function
    def new_ID3_node(self, data):
        node = {'id': self.__nodeCounter, 'label': None, 'attribute': None, 'entropy': None, 'samples': None,
                'classCounts': None, 'nodes': [], 'att_value': None}
        self.nodes.append(node)
        self.__nodeCounter += 1
        if data:
            node = self.set_node_attr(node, data)
        return node

    def set_node_attr(self, node, data):
        nodeString = ''
        for k in node:
            if ((node[k] != None) and (k != 'nodes')):
                nodeString += "\n" + str(k) + ": " + str(node[k])
        for item in data:
            node[item] = data[item]
            nodeString += "\n" + str(item) + ": " + str(node[item])
        self.__dot.node(str(node['id']), label=nodeString)
        self.nodes[node['id']] = node
        return node

    # adds the node into the graph for visualisation (creates a dot-node)
    def add_node_to_graph(self, node, parentid):
        nodeString = ''
        for k in node:
            if ((node[k] != None) and (k != 'nodes')):
                nodeString += "\n" + str(k) + ": " + str(node[k])

        self.__dot.node(str(node['id']), label=nodeString)
        if (parentid != -1):
            self.__dot.edge(str(parentid), str(node['id']))
            nodeString += "\n" + str(parentid) + " -> " + str(node['id'])

        print(nodeString)

        return

    # make the visualisation available
    def make_dot_data(self):
        return self.__dot

    # the entry point for the recursive ID3-algorithm, you need to fill in the calls to your recursive implementation
    def fit(self, data, target, attributes, classes):
        root = self.new_ID3_node(data={})
        self.__root = root
        self.add_node_to_graph(root, parentid=-1)
        tree = self.rec_id3(root, data, target, attributes, classes)
        return tree

    def entropy(self, classes, target):
        entropy = 0
        for label in classes:
            if self.toy:
                if(target.count(label)):
                    aux = target.count(label) / len(target)
                    entropy += (target.count(label) / len(target)) * log(aux, 2)
            else:
                temp = Counter(target)[label]
                if(temp):
                    aux = temp / len(target)
                    entropy += (temp / len(target)) * log(aux, 2)
        return -entropy

    def info_gain(self, entropy, data, attribute, target, classes):
        item_entropies = []
        info_gain = 0
        for item in attribute:
            item_target = []
            item_count = 0
            for i in range(len(data)):
                if item in data[i]:
                    item_target.append(target[i])
                    item_count += 1
            if item_count:
                ent = self.entropy(classes, item_target)
                item_entropies.append(ent)
                info_gain += ent * item_count / len(data)
        return entropy - info_gain

    def rec_id3(self, node, data, target, attributes, classes):
        # No attributes left or all samples belong to one class
        if not attributes or len(set(target)) == 1 or data == []:
            return self.set_node_attr(node, data={'entropy': self.entropy(classes, target),'label': most_common_class(target), 'samples': len(target)})
        else:
             
            ent = self.entropy(classes, target)
            info_gain = []
            for att in attributes:
                info_gain.append(self.info_gain(ent, data, attributes[att], target, classes))
            max_info_gain = max(info_gain)
            #print(info_gain)
            #pdb.set_trace()
            attribute = list(attributes.items())[info_gain.index(max_info_gain)][0]
            #print(attribute)
            values = list(attributes.items())[info_gain.index(max_info_gain)][1]
            self.set_node_attr(node, data={'attribute': attribute, 'entropy': ent, 'samples': len(target)})
            attributes_cpy = OrderedDict(attributes)
            del attributes_cpy[attribute]
            for value in values:
                branch_node = self.new_ID3_node(data={'att_value': value})
                self.add_node_to_graph(branch_node, node['id'])
                new_target = []
                new_data = []
                node['nodes'].append(branch_node)
                for i in range(len(data)):
                    if value in data[i]:
                        new_data.append(data[i])
                        new_target.append(target[i])
                if not new_target:
                    #pdb.set_trace()
                    self.set_node_attr(branch_node, data={'label': most_common_class(target), 'samples': 0})
                else:
                    self.rec_id3(branch_node, new_data, new_target, attributes_cpy, classes)

    def predict(self, data):
        predicted = list()
        print(self.nodes)
        for example in data:
            predicted.append(self.predict_rek(self.__root, example))
        return predicted

    def predict_rek(self, node, example):
        if len(node['nodes']) is 0:
            return node['label']
        for branch in node['nodes']:
            if self.nodes[branch['id']]['att_value'] in example:
                return self.predict_rek(self.nodes[branch['id']], example)
if __name__ == "__main__": main()