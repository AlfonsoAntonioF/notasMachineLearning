import matplotlib.pyplot as plt
import numpy as np

# Generar valores de theta
theta = np.linspace(0, 2 * np.pi, 1000)

# Valores de z en el círculo de radio 4
z = 4 * np.exp(1j * theta)

# Mapeo de f(z) = e^z
f_z = np.exp(z)

# Graficar el círculo original y su mapeo
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico del círculo original
ax[0].plot(np.real(z), np.imag(z), label='$|z|=4$', color='blue')
ax[0].set_title('Círculo original: $|z|=4$')
ax[0].set_xlabel('Re')
ax[0].set_ylabel('Im')
ax[0].grid(True)
ax[0].axis('equal')
ax[0].legend()

# Gráfico del mapeo
ax[1].plot(np.real(f_z), np.imag(f_z), label='$f(z) = e^z$', color='red')
ax[1].set_title('Mapeo de $f(z) = e^z$')
ax[1].set_xlabel('Re')
ax[1].set_ylabel('Im')
ax[1].grid(True)
ax[1].axis('equal')
ax[1].legend()

plt.show()



# Generar valores de theta
theta = np.linspace(0, 2 * np.pi, 1000)

# Valores de z en el círculo de radio 4
z = 4 * np.exp(1j * theta)

# Mapeo de f(z) = z^2
f_z = z ** 2

# Graficar el círculo original y su mapeo
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico del círculo original
ax[0].plot(np.real(z), np.imag(z), label='$|z|=4$', color='blue')
ax[0].set_title('Círculo original: $|z|=4$')
ax[0].set_xlabel('Re')
ax[0].set_ylabel('Im')
ax[0].grid(True)
ax[0].axis('equal')
ax[0].legend()

# Gráfico del mapeo
ax[1].plot(np.real(f_z), np.imag(f_z), label='$f(z) = z^2$', color='red')
ax[1].set_title('Mapeo de $f(z) = z^2$')
ax[1].set_xlabel('Re')
ax[1].set_ylabel('Im')
ax[1].grid(True)
ax[1].axis('equal')
ax[1].legend()

plt.show()


# Centros y radios de los círculos
centro_a = (4, -3)
radio_a = 5

centro_b = (-2, -2)
radio_b = 2

# Crear valores theta para los círculos
theta = np.linspace(0, 2 * np.pi, 1000)

# Círculo a)
x_a = centro_a[0] + radio_a * np.cos(theta)
y_a = centro_a[1] + radio_a * np.sin(theta)

# Círculo b)
x_b = centro_b[0] + radio_b * np.cos(theta)
y_b = centro_b[1] + radio_b * np.sin(theta)

# Graficar los círculos
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico del círculo a)
ax[0].plot(x_a, y_a, label='$|z - 4 + j3| = 5$', color='blue')
ax[0].scatter(centro_a[0], centro_a[1], color='red')  # Centro del círculo
ax[0].text(centro_a[0], centro_a[1], ' (4, -3)', fontsize=12, ha='right')
ax[0].set_title('Círculo: $|z - 4 + j3| = 5$')
ax[0].set_xlabel('Re')
ax[0].set_ylabel('Im')
ax[0].grid(True)
ax[0].axis('equal')
ax[0].legend()

# Gráfico del círculo b)
ax[1].plot(x_b, y_b, label='$|z + 2 + j2| = 2$', color='green')
ax[1].scatter(centro_b[0], centro_b[1], color='red')  # Centro del círculo
ax[1].text(centro_b[0], centro_b[1], ' (-2, -2)', fontsize=12, ha='right')
ax[1].set_title('Círculo: $|z + 2 + j2| = 2$')
ax[1].set_xlabel('Re')
ax[1].set_ylabel('Im')
ax[1].grid(True)
ax[1].axis('equal')
ax[1].legend()

plt.show()
