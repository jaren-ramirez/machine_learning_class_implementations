import numpy as np
from sklearn import datasets, model_selection
import matplotlib.pyplot as plt


def mean_squared_error(X_test, y_test, theta_0, weights,N):
    y_hat_arr = theta_0 + np.dot(X_test,weights)
    error = y_hat_arr - y_test
    error_squared = np.power(error, 2)
    mean_square_error = np.sum(error_squared) * (1/(2*N))
    return mean_square_error

def grad_descent(x_train, y_train, theta_0, weights, N, learning_rate):
    
    num_features = len(weights)
    y_hat_arr = theta_0 + np.dot(x_train,weights) # we compute all the estimated y values for each sample by adding the intercept to our dot product of sample_i
                                                  # and scalar_i 
        
    error_arr = y_hat_arr - y_train
    new_theta_0 = theta_0 - learning_rate * (1/N) * np.sum(error_arr) 
    new_weights = np.zeros((num_features,1))

    gradient = np.dot(x_train.T, error_arr) / N  # we transpose the array here and get the dot product of each feature then divide by our total sample size
    new_weights = weights - learning_rate * gradient # now we can take all weights and subtract the weights from our gradient and learning rate
        
    return (new_theta_0, new_weights) # we return our new intercept and scalar values

breast_cancer = datasets.load_breast_cancer(as_frame=True)

breast_cancer = breast_cancer.frame

breast_cancer.head().to_csv('breast_cancer_test.csv', index=False, mode="w")

# y = diabetes[["target"]].to_numpy()
# X = diabetes[["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]].to_numpy()

# y = diabetes[["target"]].to_numpy()

# X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.20,random_state=0)

# N = len(X_train)
# test_N = len(X_test)
# num_features = 10
# iterations = 100000
# learning_rate = 0.01

# theta_0 = 0
# weights = np.zeros((num_features,1))



# errors = []

# for iteration in range(iterations):
#     errors.append(mean_squared_error(X_test,y_test,theta_0,weights,test_N))
#     theta_0, weights = grad_descent(X_train,y_train,theta_0,weights,N,learning_rate)

# plt.plot(errors)
# plt.title('Mean Squared Error over Iterations')
# plt.xlabel('Iteration')
# plt.ylabel('Mean Squared Error')
# plt.savefig(fname="LogRegression")