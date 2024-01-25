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