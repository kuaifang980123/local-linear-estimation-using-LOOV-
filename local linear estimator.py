import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from statsmodels.nonparametric import kernel_regression

boston = datasets.load_boston()
labels = boston.feature_names
X = boston.data[:, 0]
Y = boston.target
n = np.shape(X)[0]


X = np.log(X)
Y = np.log(Y)

plt.plot(X, Y, "og")
plt.show()


def main(X, Y, h):
    range_h = np.arange(np.min(X), np.max(X), h)
    X_predict = []
    Y_predict = []
    n = np.shape(Y)[0]
    for i in range_h:
        xtilde = np.sqrt((X - i) ** 2) / h
        K = (xtilde < 1) * (1 - xtilde)
        Ytilde = Y * np.sqrt(K)
        Ytilde=Ytilde.reshape(n, 1)
        Xtilde = np.vstack((np.sqrt(K), X * np.sqrt(K))).T
        try:
            gamma = np.linalg.inv(Xtilde.T @ Xtilde) @ (Xtilde.T @ Ytilde)
        except:
            True
        else:
            a = gamma[0]
            b = gamma[1]
            for j in np.arange(i - h / 2, i + h / 2, 0.001):
                X_predict.append(j)
                Y_predict.append((a + b * j)[0])
    return np.asarray(X_predict), np.asarray(Y_predict)


def find_nearest(array, value):
    array = np.asarray(array)
    n = np.shape(array)[0]
    array = array.reshape(n, 1)
    idx = (np.abs(array - value)).argmin()
    return idx


MSE = []
for h in np.arange(0.1, 1, 0.1):
    SE = []
    for i in range(len(X)):
        X_1 = np.delete(X, i)
        Y_1 = np.delete(Y, i)
        X_pre, Y_pre = main(X_1, Y_1, h)
        Y_p = Y_pre[find_nearest(X_pre, X[i])]
        SE.append((Y_p - Y[i]) ** 2)
    MSE.append(np.mean(SE))
print(MSE.index(min(MSE)))

MSE = []
for h in np.arange(0.6, 0.7, 0.01):
    SE = []
    for i in range(len(X)):
        X_1 = np.delete(X, i)
        Y_1 = np.delete(Y, i)
        X_pre, Y_pre = main(X_1, Y_1, h)
        Y_p = Y_pre[find_nearest(X_pre, X[i])]
        SE.append((Y_p - Y[i]) ** 2)
    MSE.append(np.mean(SE))

print(MSE.index(min(MSE)))

MSE = []
for h in np.arange(0.68, 0.69, 0.001):
    SE = []
    for i in range(len(X)):
        X_1 = np.delete(X, i)
        Y_1 = np.delete(Y, i)
        X_pre, Y_pre = main(X_1, Y_1, h)
        Y_p = Y_pre[find_nearest(X_pre, X[i])]
        SE.append((Y_p - Y[i]) ** 2)
    MSE.append(np.mean(SE))
print(MSE.index(min(MSE)))
# So the best h is 0.686

X_pre, Y_pre = main(X, Y, 0.686)
plt.figure(figsize=(10,6))
plt.plot(X, Y, 'og')
plt.plot(X_pre, Y_pre, color='blue')
plt.xlabel('Crime Rate')
plt.ylabel('Price')
plt.legend(['Data','Pred'])
plt.show()

# 或者使用python的kernel regression包
model = kernel_regression.KernelReg(endog = Y, exog = [X],var_type = 'c',reg_type = "ll",bw = 'cv_ls',ckertype='gaussian')
pred = model.fit(X)[0]

plt.figure(figsize=(10,6))
plt.scatter(X,Y,color = 'green')
plt.scatter(X,pred,color = 'blue')
plt.xlabel('Crime Rate')
plt.ylabel('Price')
plt.legend(['Data','Pred'])
plt.show()