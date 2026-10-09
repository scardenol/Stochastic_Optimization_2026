# Práctica 2 — Inversión multietapa con L-shaped anidado

Optimización Estocástica · Maestría en Matemáticas Aplicadas, EAFIT · 2026-2

- Todo el código está en un único notebook pre-ejecutado: [`practica2_inversion_multietapa.ipynb`](practica2_inversion_multietapa.ipynb) (para visualizar el notebook ejecutado basta con abrir el link y GitHub lo renderiza). No genera módulos ni scripts.
- Solver: `scipy.optimize.linprog` (HiGHS), sin licencias.
- Datos: únicamente `yfinance` (Yahoo Finance, campo `Adj Close`, `auto_adjust=False`).

## Propósito
Usar datos hasta 2025 para construir y resolver un modelo de inversión multietapa 
mediante L-shaped anidado. Después, aplicar la política obtenida a los rendimientos observados en
2026 y analizar su desempeño fuera de muestra.

## Cómo ejecutarlo

### Google Colab (recomendado)
1) Abrir el notebook [`practica2_inversion_multietapa.ipynb`](practica2_inversion_multietapa.ipynb),
2) darle click al botón *Open in Colab*,
3) darle click a *Entorno de ejecución → Ejecutar todo*.

La primera celda instala las dependencias (`%pip install`). Al final se descarga
`practica2_salidas.zip` con `datos/`, `resultados/` y `figuras/`.

### **Local**
Descargar el repo y en una terminal ejecutar:
```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace practica2_inversion_multietapa.ipynb
```

La descarga exige que el trimestre julio–septiembre de 2026 haya terminado (fecha actual
≥ 1-oct-2026); de lo contrario el notebook se detiene en lugar de usar un trimestre
incompleto. Para reutilizar los cierres ya guardados sin conexión, poner
`DESCARGAR = False` en la celda de parámetros (lee `datos/*.csv`).

No hay componentes aleatorios, por lo que no se usa semilla. Tiempo aproximado: menos de
un minuto de cómputo más la descarga.

## Estructura del directorio
```
Práctica 2/
├── practica2_inversion_multietapa.ipynb      notebook principal, autocontenido y ejecutable en Colab
├── datos/
│   ├── *.csv                           archivos .csv que contienen los datos de calibración y validación
│   └── metadatos_descarga.json         archivo con la metadata de la consulta de los datos (fuente y fecha)
├── figuras/
│   └── *.png                           imágenes .png con los resultados exportados del notebook
├── informe/
│   ├── enunciado.pdf       enunciado oficial de la práctica 1, en formato pdf
│   └── informe.pdf         informe técnico en LaTeX compilado en pdf (entregable principal)
├── resultados/
│   ├── *.csv               resultados exportados
│   ├── *.tex               resultados extras como cifras y tablas usados para alimentar el informe de latex
│   └── resumen.json        archivo .json extra que compila todas las cifras en un formato fácil de leer
├── requirements.txt        dependencias
└── README.md               (este archivo) descripción, motivación e instalación del directorio
```

## Qué hace cada sección

| Sección | Contenido |
|---|---|
| 0–1 | Parámetros ($b=100$, $M=106$, $\lambda=4$), descarga con `yfinance` y controles de calidad (65 + 4 precios, fechas de cierre) |
| 2–3 | 64 rendimientos, estadísticas, estados conjuntos (`n_u`, `q_u`, `R_u`) |
| 4 | Árbol de 585 nodos; lista exportada; las 512 probabilidades terminales suman 1 |
| 5 | Equivalente determinista (1 243 variables, 585 igualdades) y L-shaped anidado; verificación del signo del dual |
| 6 | Instancia reducida (15 nodos, 37 variables, 15 igualdades) frente al equivalente determinista |
| 7 | Instancia completa: LB/UB por ciclo, cortes, LP terminal y dual |
| 8 | Política (73 nodos), 512 hojas, modelo EV, EEV y VSS |
| 9 | Validación enero–septiembre de 2026 con la extensión por proporciones |
| 10–12 | Diagnósticos para las conclusiones, tablas/macros `.tex` y paquete de salidas |

## Archivos generados

- `datos/cierres_calibracion.csv`, `datos/cierres_validacion.csv`: cierres trimestrales
  ajustados usados; `datos/metadatos_descarga.json`: fuente, versión de `yfinance`,
  ventanas y fecha de consulta.
- `resultados/arbol_nodos.csv`, `politica_73_nodos.csv`, `hojas_512.csv`,
  `convergencia_lshaped.csv`, `validacion_*.csv`, `resumen.json`.
- `resultados/tab_*.tex` y `macros.tex`: tablas y cifras que el informe incluye con `\input`.
- `figuras/*.png`: figuras del informe.

## Informe

`informe/main.tex` incluye todas las cifras desde `resultados/` y las figuras desde
`figuras/`. Para compilar, copiar esas dos carpetas junto a `main.tex` (por ejemplo en
Overleaf) y reemplazar `\RepoURL` y los nombres de los autores.
