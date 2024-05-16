import numpy as np
import matplotlib.pyplot as plt

# Generar datos
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Graficar los puntos de datos
plt.scatter(X, y, alpha=0.5)

# Ajuste de límites de los ejes X e Y
plt.xlim(0, 2)
plt.ylim(0, 14)

# Etiquetas y título
plt.xlabel('$X_1$') 
plt.ylabel('y')
plt.title('Conjunto de datos lineales generados aleatoriamente')

# Mostrar la gráfica
plt.show()

X_b = np.c_[np.ones((100, 1)), X] # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

print(f'theta_best:{theta_best}')


X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new] # add x0 = 1 to each instance
y_predict = X_new_b.dot(theta_best)
print(f'y_predict:{y_predict}')


plt.plot(X_new, y_predict, "r-",label='Prediccion')
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
plt.xlabel('$X_1$') 
plt.ylabel('y')
plt.title('Predicciones del modelo de Regresión Lineal.')
plt.legend()
plt.show()


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
#lin_reg.intercept_, lin_reg.coef_
print(f'lin_rer.intercept_:{lin_reg.intercept_}, lin_reg.coef_:{lin_reg.coef_}')

lin_reg.predict(X_new)
theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
print(f"theta_best_svd: {theta_best_svd}")

np.linalg.pinv(X_b).dot(y)
print(f"np.linalg.pinv(X_b).dot(y):{np.linalg.pinv(X_b).dot(y)}")