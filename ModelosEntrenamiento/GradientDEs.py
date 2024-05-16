import numpy as np
import matplotlib.pyplot as plt

# Generar datos
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

X_b = np.c_[np.ones((100, 1)), X]

# Batch Gradient Descent
eta = 0.1 # tasa de aprendizaje
n_iterations = 1000
m = 100
theta = np.random.randn(2,1) # random initialization
theta_values_bgd=[theta]
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients
    theta_values_bgd.append(theta) # Almacenar el nuevo valor de theta

    
print(f"Theta:\t{theta}")

#Stochastic Gradient Descent

n_epochs = 50
t0, t1 = 5, 50 # learning schedule hyperparameters
theta = np.random.randn(2,1) # random initialization
theta_values_sgd = [theta]  # Lista para almacenar los valores de theta en cada paso

def learning_schedule(t):
    return t0 / (t + t1)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
        theta_values_sgd.append(theta)  # Almacenar el nuevo valor de theta


print(f'theta:{theta}')

# Convertir la lista de valores de theta en un array numpy para facilitar la manipulación
theta_values_sgd = np.array(theta_values_sgd)

plt.scatter(X, y)

# Graficar las rectas de regresión calculadas en cada época
for i in range(len(theta_values_sgd)-30):
    y_predict = X_b.dot(theta_values_sgd[i])
    plt.plot(X, y_predict, alpha=0.2, color='r')
    
# Graficar la última recta de regresión calculada con un color diferente
y_predict_last = X_b.dot(theta_values_sgd[-1])
plt.plot(X, y_predict_last, color='g', label=f'{theta}')


plt.xlabel("$X1$")
plt.ylabel("$y$")
plt.title("Los primeros 20 pasos de Descenso de Gradiente Estocástico")
plt.grid(True)
plt.legend()
plt.show()


#Stochastic Gradient Descent usando SGDRegressor

from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())

print(f'sgd_reg.intercept_, sgd_reg.coef_:{sgd_reg.intercept_, sgd_reg.coef_}')

#Mini-batch Gradient Descent

n_epochs = 50
t0, t1 = 5, 50 # learning schedule hyperparameters
theta = np.random.randn(2,1) # random initialization

def learning_schedule(t):
    return t0 / (t + t1)

theta_values_mbgd = [theta]

for epoch in range(n_epochs):
    shuffled_indices = np.random.permutation(m)
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    for i in range(m):
        xi = X_b_shuffled[i:i+1]
        yi = y_shuffled[i:i+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
        theta_values_mbgd.append(theta)# Almacenar el nuevo valor de theta

# Convertir listas de valores de theta en arrays numpy
theta_values_bgd = np.array(theta_values_bgd)
theta_values_sgd = np.array(theta_values_sgd)
theta_values_mbgd = np.array(theta_values_mbgd)

# Graficar los valores de theta obtenidos en cada iteración de cada método
plt.figure(figsize=(12, 8))
plt.plot(theta_values_bgd[:, 0], theta_values_bgd[:, 1], "b-", label="Batch Gradient Descent")
plt.plot(theta_values_sgd[:, 0], theta_values_sgd[:, 1], "r-", label="Stochastic Gradient Descent")
plt.plot(theta_values_mbgd[:, 0], theta_values_mbgd[:, 1], "g-", label="Mini-batch Gradient Descent")
plt.xlabel(r"$\theta_0$", fontsize=14)
plt.ylabel(r"$\theta_1$", fontsize=14)
plt.title("Comparación de los valores de theta entre métodos de optimización", fontsize=16)
plt.legend()
plt.grid(True)
plt.show()