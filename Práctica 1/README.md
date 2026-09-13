# Práctica 1: Ruteo urbano con tiempos de viaje estocásticos en Manhattan

## Propósito
Modelar y resolver un problema estocástico lineal de dos etapas en el que la primera etapa contiene decisiones binarias de ruteo y la segunda etapa contiene únicamente decisiones continuas de recurso. Se construirá una aproximación por promedio muestral (SAA, del inglés Sample Average Approximation), se validará una formulación extensa como programa lineal entero mixto (MILP, del inglés Mixed-Integer Linear Program), se implementará el algoritmo Integer L-shaped estudiado en clase y se contrastará la solución estocástica con una aproximación determinista basada en el perfil promedio de tiempos.

## Estructura del directorio
```
Práctica 1/
├── Practica 1.pdf: enunciado original
├── p1_code.ipynb: notebook de Python (ejecutado) con la solución
└── README.md: descripción del directorio
```

## Instrucciones para usar el código
El notebook de Python [`p1_code.ipynb`](p1_code.ipynb) fue generado en un ambiente virtual online de Google Colab y copiado a este repositorio.

Para reproducir el código, basta con:
1. previsualizar el notebook desde GitHub al darle click al link del notebook ([`p1_code.ipynb`](p1_code.ipynb)),
2. darle click al botón **Open in Colab** que aparece en la parte superior del notebook,
3. darle click al botón **Run all** en Google Colab y esperar a que se ejecute el código.

**Nota:** los resultados obtenidos son EXACTAMENTE los mismos que se pueden observar en la previsualización del notebook aquí en el repositorio.