# Final Exercises for Trabajo con Python
# Author: Miguel A. Castellanos (from text) or me? It's the notes.

import numpy as np
import matplotlib.pyplot as plt
import torch

# 1. Create a dictionary with personal info
persona = {
    "edad": 30,
    "sexo": "M",
    "estudios": "Ingeniería",
    "nombre_amigos": ["Ana", "Luis", "Maria"]
}

# 2. Function to print the dictionary

def imprimir_diccionario(dic):
    for clave, valor in dic.items():
        print(f"{clave}: {valor}")

# 3. Class that receives the dictionary and prints it

class Persona:
    def __init__(self, info):
        self.info = info

    def imprimir(self):
        imprimir_diccionario(self.info)

# 4. List of dictionaries with different people
personas = [
    {
        "edad": 25,
        "sexo": "F",
        "estudios": "Medicina",
        "nombre_amigos": ["Juan", "Pedro"]
    },
    {
        "edad": 40,
        "sexo": "M",
        "estudios": "Derecho",
        "nombre_amigos": ["Laura", "Sofia"]
    },
    persona  # reutilizamos la primera persona
]

# 5. Print list using for
for p in personas:
    imprimir_diccionario(p)
    print("-" * 20)

# 6. Generate X from -3 to 3 with step 0.1
x = np.arange(-3, 3.1, 0.1)

# 7. Generate Y as normal distribution (0,1)
y = np.random.normal(0, 1, size=x.shape)

# 8. Plot the distribution
plt.figure()
plt.plot(x, y, label="N(0,1)")
plt.title("Distribución Normal")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()

# 9. Create vector 0..10, reshape to 2x5, multiply by ones, convert to tensor
vector = np.arange(0, 10)
matriz = vector.reshape((2, 5))
resultado = matriz * np.ones((2, 5))
tensor_resultado = torch.from_numpy(resultado)
print("Tensor result:\n", tensor_resultado)

# 10. Compute derivative of x**3 - 2*x and plot
x_tensor = torch.linspace(-5, 5, steps=100, requires_grad=True)
funcion = x_tensor**3 - 2 * x_tensor
funcion.backward(torch.ones_like(x_tensor))

dx = x_tensor.detach().numpy()
dy = funcion.detach().numpy()
derivada = x_tensor.grad.detach().numpy()

plt.figure()
plt.plot(dx, dy, label="x^3 - 2x")
plt.plot(dx, derivada, label="Derivada", linestyle="--")
plt.title("Función y Derivada")
plt.xlabel("x")
plt.grid(True)
plt.legend()
plt.show()
