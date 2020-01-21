from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics, datasets, cluster
import scipy.stats as st

#GMM
class GMM:
    #Estado inicial
    def __init__(self,data, X,classes,iterations):
        self.data =data
        self.iterations = iterations
        self.classes = classes
        self.X = X
        self.theta = dict()

    #EM Algorithm
    def fit(self):
        """ 1. Set the initial mu, covariance and pi values"""
        indexes = np.random.randint(len(self.data), size=int(len(self.data)*0.1))
        temp = dict()
        for i in indexes:
            k = digits_target[i]
            pixels = self.data[i]
            if k not in temp:
                temp[k] = list()
            temp[k].append(pixels)
            
        for k in temp:
            pi = 0.1
            values = np.array(temp[k])
            mu = np.zeros(self.data.shape[1])
            cov = np.zeros(self.data.shape[1])
            epsilon = 0.01
            for i in range(len(values[0])):
                mu[i] = np.mean(values[:,i])
                cov[i] = np.var(values[:,i]) + epsilon
            self.theta[k] = np.array([pi, mu, cov])
               
        r = self.E_step()
        self.theta = self.M_step(r)
        for i in range(self.iterations):               
            """E Step"""
            r = self.E_step()
            """M Step"""
            self.theta = self.M_step(r)
    
    def E_step(self):
        r = np.zeros((self.X.shape[0],self.classes)) 
        for i in range(len(self.X)):
            prob = np.prod([st.norm.pdf(self.X[i], self.theta[k][1], np.sqrt(self.theta[k][2])) for k in range(self.classes)], axis = 1)
            prod = [self.theta[k][0]*prob[k] for k in range(self.classes)]
            den = np.sum(prod)
            r[i,:] = prod/den    
        return r

    def M_step(self, r):
        r_k = {k:np.sum(r[:,k]) for k in range(10)}
        self.theta = dict()
        epsilon = 0.01
        for k in r_k:
            pi = r_k[k]/len(self.X)
            mu = np.sum([r[i][k]*self.X[i] for i in range(len(self.X))], axis=0)/r_k[k]
            cov = np.sum([r[i][k]*(self.X[i]**2) for i in range(len(self.X))], axis=0)/r_k[k] - mu**2 + epsilon
            self.theta[k] = np.array([pi, mu, cov])
        return self.theta 

    def predict(self, X):
        probs = np.zeros((len(X), self.classes))
        for k in self.theta:
            pi = self.theta[k][0]
            mu = self.theta[k][1]
            cov = self.theta[k][2]
            probs[:,k] = np.sum(np.log(st.norm.pdf(X, mu, cov)) + np.log(pi), axis=1)
        return np.argmax(probs,axis=1)

    def repair(self, y_true, y_pred):
        k_map = dict()
        for k in range(self.classes):
            idxs = [i for i in range(len(y_test)) if y_pred[i]==k]
            unique, counts = np.unique(y_true[idxs], return_counts=True)
            k_map[k] = unique[np.argmax(counts)]
        y_real = list()
        for y in y_pred:
            y_real.append(k_map[y])
        return y_real



#Dataset 
digits = load_digits()
data = digits.data
digits_split = int(len(data)*0.7)
x_train = data[:digits_split]
x_test = data[digits_split:]
digits_target = digits.target
y_train = digits_target[:digits_split]
y_test = digits_target[digits_split:]

print('Training data:', len(x_train), '\nTraining Labels:', len(y_train), '\nTesting Data:', 
      len(x_test), '\nTesting Labels:', len(y_test), '\nCheck:', 
      len(data) == len(x_train) + len(x_test))
print(x_train.shape)
print(y_train.shape)
x_train /= 16
x_test /= 16

GMM = GMM(data,x_train,10,50)     
GMM.fit()

y_pred_train = GMM.predict(x_train)

print("Classification report EM:\n%s\n" % 
      (metrics.classification_report(y_train, y_pred_train)))
print("Confusion matrix EM:\n%s" % metrics.confusion_matrix(y_train, y_pred_train))
print()
h_c_v = metrics.homogeneity_completeness_v_measure(y_train, y_pred_train)
print('Homogenity:',h_c_v[0])
print('Completeness:',h_c_v[1])
print('V-measure:',h_c_v[2])

y_pred_test = GMM.predict(x_test)

print("Classification report EM:\n%s\n" % 
      (metrics.classification_report(y_test, y_pred_test)))
print("Confusion matrix EM:\n%s" % metrics.confusion_matrix(y_test, y_pred_test))
print()
h_c_v = metrics.homogeneity_completeness_v_measure(y_test, y_pred_test)
print('Homogenity:',h_c_v[0])
print('Completeness:',h_c_v[1])
print('V-measure:',h_c_v[2])


y_pred_repair = GMM.repair(y_train, y_pred_train)
print("Classification report SKLearn K-Means:\n%s\n" % 
      (metrics.classification_report(y_train, y_pred_repair)))
print("Confusion matrix SKLearn EM:\n%s" % metrics.confusion_matrix(y_train, y_pred_repair))
print()
h_c_v = metrics.homogeneity_completeness_v_measure(y_train, y_pred_repair)
print('Homogenity:',h_c_v[0])
print('Completeness:',h_c_v[1])
print('V-measure:',h_c_v[2])
print()

y_pred_test_repair = GMM.repair(y_test, y_pred_test)
print("Classification report SKLearn k-Means:\n%s\n" % 
      (metrics.classification_report(y_test, y_pred_test_repair)))
print("Confusion matrix SKLearn EM:\n%s" % metrics.confusion_matrix(y_test, y_pred_test_repair))
print()
h_c_v = metrics.homogeneity_completeness_v_measure(y_test, y_pred_test_repair)
print('Homogenity:',h_c_v[0])
print('Completeness:',h_c_v[1])
print('V-measure:',h_c_v[2])


k_means = cluster.KMeans(n_clusters=10).fit(x_train)
y_pred_kmeans = k_means.predict(x_train)
y_pred_kmeans_real = GMM.repair(y_train, y_pred_kmeans)
print("Classification report SKLearn K-Means:\n%s\n" % 
      (metrics.classification_report(y_train, y_pred_kmeans_real)))
print("Confusion matrix SKLearn EM:\n%s" % metrics.confusion_matrix(y_train, y_pred_kmeans_real))
print()
h_c_v = metrics.homogeneity_completeness_v_measure(y_train, y_pred_kmeans_real)
print('Homogenity:',h_c_v[0])
print('Completeness:',h_c_v[1])
print('V-measure:',h_c_v[2])
print()

y_pred_kmeans_test = k_means.predict(x_test)
y_pred_kmeans_test_real = GMM.repair(y_test, y_pred_kmeans_test)
print("Classification report SKLearn k-Means:\n%s\n" % 
      (metrics.classification_report(y_test, y_pred_kmeans_test_real)))
print("Confusion matrix SKLearn EM:\n%s" % metrics.confusion_matrix(y_test, y_pred_kmeans_test_real))
print()
h_c_v = metrics.homogeneity_completeness_v_measure(y_test, y_pred_kmeans_test_real)
print('Homogenity:',h_c_v[0])
print('Completeness:',h_c_v[1])
print('V-measure:',h_c_v[2])
