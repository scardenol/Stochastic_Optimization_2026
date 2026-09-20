# Práctica 1: Ruteo urbano con tiempos de viaje estocásticos en Manhattan

Universidad EAFIT · Maestría en Matemáticas Aplicadas · Optimización Estocástica 2026-2

## Propósito
Modelar y resolver un problema estocástico lineal de dos etapas en el que la primera
etapa contiene decisiones binarias de ruteo y la segunda etapa contiene únicamente
decisiones continuas de recurso. Se construye una aproximación por promedio muestral
(SAA, *Sample Average Approximation*), se valida una formulación extensa como MILP 
(*Mixed-Integer Lineal Program*), se implementa el algoritmo
*Integer L-shaped* y se contrasta la solución estocástica con una aproximación
determinista basada en el perfil promedio de tiempos.

## Estructura del directorio
```
Práctica 1/
├── p1_practica1.ipynb      notebook principal, autocontenido y ejecutable en Colab
├── p1_datos.py             descarga, filtros y construcción de los pools origen-destino
├── p1_modelo.py            forma extensa MILP, subproblemas de recurso e Integer L-shaped
├── p1_experimentos.py      orquestación de los experimentos y visualizaciones
├── build_notebook.py       genera el notebook embebiendo los tres módulos
├── informe/
    ├── enunciado.pdf       enunciado oficial de la práctica 1, en formato pdf
│   └── informe.pdf         informe técnico en LaTeX compilado en pdf (entregable principal)
├── test_modelo.py          valida el L-shaped contra la forma extensa
├── test_datos.py           valida los cuatro filtros y la construcción de pools
├── test_experimentos.py    prueba de extremo a extremo con datos sintéticos
├── test_informe.py         comprobaciones estructurales del LaTeX
├── requirements.txt        dependencias
└── README.md
```

### Opción A: Google Colab (recomendada)

1. Subir `p1_practica1.ipynb` a Google Colab.
2. Ejecutar **Run all**.

El notebook instala las dependencias, escribe los tres módulos en el entorno de
ejecución, descarga los archivos Parquet oficiales de la TLC y corre todos los
experimentos. No hace falta subir ningún otro archivo.

### Opción B: local

```bash
pip install -r requirements.txt
python build_notebook.py      # regenera el notebook desde los módulos
jupyter notebook p1_practica1.ipynb
```

### Pruebas

```bash
python test_datos.py
python test_modelo.py
python test_experimentos.py
```

Las pruebas usan datos sintéticos con la misma estructura que la instancia real
(11 nodos, 110 arcos), de modo que corren sin descargar los Parquet.

## Semillas y reproducibilidad

| Uso | Semilla |
|---|---|
| Escenarios de entrenamiento (enero, K=50 y K=200) | `29` |
| Escenarios de prueba (febrero, K=1000) | `2027` |

Los escenarios se generan con `numpy.random.default_rng` consumido arco por arco,
lo que garantiza independencia entre las componentes de $\xi$ y reproducibilidad
exacta. Usar la misma semilla en cada arco por separado induciría dependencia entre
arcos, que es justo lo que el enunciado pide evitar.

## Notas de implementación

**Propiedad estructural del recurso.** El escenario entra en el subproblema de
segunda etapa únicamente a través del escalar $\tau_s(x) = \sum_{ij} \xi^{(s)}_{ij} x_{ij}$,
de modo que $Q(x,\xi_s)=\varphi(\tau_s)$ para una única función convexa lineal por
tramos. Esto permite construir los $K$ modelos LP una sola vez y reoptimizar
cambiando solo el lado derecho, y da una verificación independiente de los valores y
los duales (`recurso_analitico`), que coincide con el LP hasta precisión de máquina.

**Recurso completo.** La variable de emergencia $e^{(s)}$ no tiene cota superior, así
que el subproblema es factible para toda ruta y todo escenario. No se generan cortes
de factibilidad y $L=0$ es una cota inferior global válida para el corte de
optimalidad entero.

**Incumbente inicial.** La relajación lineal del ruteo con MTZ es débil, de modo que
el Integer L-shaped arranca con la ruta del perfil promedio como incumbente. Esto no
afecta el óptimo (solo aporta una cota superior) pero reduce de forma apreciable la
exploración del árbol.

**Sin licencias adicionales.** La licencia restringida que viene con ``pip install gurobipy`` limita el
tamaño de los modelos (variables + restricciones). En el proyecto se 
asumía que esto solo afectaba a la forma extensa con $K=200$, pero en la
práctica el LICENSE LIMIT también se alcanza en el maestro del Integer
L-shaped, porque los cortes de Benders y los cortes enteros se van
acumulando como restricciones nuevas sobre el MISMO modelo persistente
durante todo el branch-and-bound (el maestro que revienta con
``GurobiError: Model too large for size-limited license`` incluso con
$K=50$). Conseguir una licencia académica sin restricción de tamaño (WLS)
requiere estar en la red de la universidad, algo que no siempre es posible.

* La forma extensa y el maestro del Integer L-shaped se resuelven con
  ``scipy.optimize.milp`` / ``scipy.optimize.linprog``, que usan ``HiGHS``
  (Apache 2.0, sin límite de tamaño ni de licencia).
* El subproblema de segunda etapa YA NO SE RESUELVE COMO LP. El propio
  informe documenta que, como $alpha_i = 1$ para todo sitio, el recurso
  $Q(x, \xi_s) = \phi(\tau_s)$ tiene forma cerrada (``recurso_analitico``),
  validada contra el LP hasta precisión de máquina. Aquí esa forma cerrada
  deja de ser solo una verificación y pasa a ser el motor de evaluación:
  no hace falta resolver $K$ LPs (ni con ``Gurobi`` ni con ``HiGHS``) para obtener
  $Q_K(x)$, los cortes de Benders (con su dual exacto) ni los detalles de
  asignación ($o, e, r, u$) que necesitan los diagnósticos y las figuras.
  Esto es exacto (no una aproximación) y, de paso, es la parte más costosa
  del algoritmo original, así que el Integer L-shaped queda además más
  rápido que la versión con Gurobi.

La única pieza que de verdad necesita un solver LP/MILP es entonces el
maestro (LP pequeño, resuelto muchas veces) y la forma extensa (un MILP
grande solo con $K=50/200$, usada una vez por experimento). Ambas usan ``HiGHS``
vía ``scipy``, sin restricciones de tamaño.

## Fuente de datos

Yellow Taxi Trip Records de la NYC Taxi & Limousine Commission:

- Entrenamiento: `yellow_tripdata_2015-01.parquet`
- Validación temporal: `yellow_tripdata_2015-02.parquet`

Los archivos Parquet **no** se incluyen en el repositorio; el notebook los descarga.