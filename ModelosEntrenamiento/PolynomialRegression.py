import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generar Datos
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

plt.scatter(X, y,color='blue')
plt.xlabel('$X1$')
plt.ylabel('$y$')
plt.title('Conjunto de datos no lineales y ruidosos generados')
plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
print(f'X[0]:{X[0]}')
print(f'X_poly[0]:{X_poly[0]}')


lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
print(f'lin_reg.intercept_, lin_reg.coef_: {lin_reg.intercept_, lin_reg.coef_}')

# Graficar los datos
plt.scatter(X, y, color='blue')
plt.xlabel('X')
plt.ylabel('y')
# Generar valores de x_new dentro del rango de X para predecir
x_new = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
x_new_poly = poly_features.transform(x_new)

# Predecir los valores de y_new
y_new = lin_reg.predict(x_new_poly)

# Graficar la curva de regresión polinomial
plt.plot(x_new, y_new, color='red')

# Mostrar la gráfica
plt.show()
