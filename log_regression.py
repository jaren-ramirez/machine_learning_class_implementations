import numpy as np
from sklearn import datasets, model_selection
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import StandardScaler

def predict(X, theta_0, weights, threshold=0.5):
    y_hat_prob = theta_0 + np.dot(X, weights)
    y_hat_prob = 1 /(1+np.exp(-1*y_hat_prob))
    
    y_pred = (y_hat_prob >= threshold).astype(int)
    return y_pred

def calculate_confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP, TN, FP, FN

def calculate_scoring_metrics(TP,TN,FP,FN):
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = (2 * precision * recall) / (precision + recall)
    accuracy = (TP + TN) / (TP + FN + TN + FP)
    return precision, recall, F1, accuracy

def binary_cross_entropy(X_test, y_test, theta_0, weights,N):
    y_hat_arr = theta_0 + np.dot(X_test,weights)

    y_hat_arr = 1 / (1 + np.exp(-1 * y_hat_arr)) 
    epsilon = 1e-15
    y_hat_arr = np.clip(y_hat_arr, epsilon, 1 - epsilon)      
    cross_entropy = (-1/N) * np.sum(y_test * np.log(y_hat_arr) + (1-y_test) * np.log(1-y_hat_arr))
    return cross_entropy

def grad_descent(x_train, y_train, theta_0, weights, N, learning_rate):
    
    num_features = len(weights)
    y_hat_arr = theta_0 + np.dot(x_train,weights)
    epsilon = 1e-15
    y_hat_arr = np.clip(y_hat_arr, epsilon, 1 - epsilon)                                              
    y_hat_arr = 1 / (1 + np.exp(-1 * y_hat_arr))
    
    error_arr = y_hat_arr - y_train
    
    new_theta_0 = theta_0 - learning_rate * (1/N) * np.sum(error_arr) 

    gradient = np.dot(x_train.T, error_arr) / N
    new_weights = weights - learning_rate * gradient
        
    return (new_theta_0, new_weights)

# breast_cancer = datasets.load_breast_cancer(as_frame=True)

# breast_cancer = breast_cancer.frame

# breast_cancer.head().to_csv('breast_cancer_test.csv', index=False, mode="w")

# y = breast_cancer[["target"]].to_numpy()
# X = breast_cancer.loc[:, breast_cancer.columns != 'target'].to_numpy()

breast_cancer = datasets.load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

X_train = np.vstack((X[y==1],X[y==0][:1]))
y_train = np.hstack((y[y==1],y[y==0][:1]))
X_test = np.vstack((X[y==1],X[y==0][:10]))
y_test = np.hstack((y[y==1],y[y==0][:10]))

# X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.20,random_state=0)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

N = len(X_train)
test_N = len(X_test)
num_features = 30
iterations = 10000
learning_rate = 0.0001

theta_0 = 0
weights = np.zeros((num_features,1))



errors = []

for iteration in range(iterations):
    errors.append(binary_cross_entropy(X_test,y_test,theta_0,weights,test_N))
    theta_0, weights = grad_descent(X_train,y_train,theta_0,weights,N,learning_rate)

y_pred_test = predict(X_test, theta_0, weights)
TP, TN, FP, FN = calculate_confusion_matrix(y_test,y_pred_test)

precision, recall, f1, accuracy = calculate_scoring_metrics(TP,TN,FP,FN)


print(f"Length of Test Set: {len(y_test)}\nTP: {TP}\nTN: {TN}\nFP: {FP}\nFN: {FN}\n")

print(f"Precision: {precision}\nRecall: {recall}\nF1: {f1}\nAccuracy: {accuracy}")

plt.plot(errors)
plt.title('Cross Entropy Iterations')
plt.xlabel('Iteration')
plt.ylabel('Cross Entropy')
plt.savefig(fname="LogRegression")