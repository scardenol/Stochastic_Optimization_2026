"""Genera p1_practica1.ipynb embebiendo los módulos del proyecto.

El notebook resultante es autocontenido: escribe los tres módulos en el entorno
de ejecución y luego corre todos los experimentos del enunciado.
"""

import json
from pathlib import Path

RAIZ = Path(__file__).parent


def md(texto):
    return {"cell_type": "markdown", "metadata": {},
            "source": texto.strip("\n").splitlines(keepends=True)}


def code(texto):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": texto.strip("\n").splitlines(keepends=True)}


def celda_modulo(nombre):
    fuente = (RAIZ / nombre).read_text(encoding="utf-8")
    return code(f"%%writefile {nombre}\n{fuente}")


FORMULACION = r"""
# Formulación matemática

## Conjuntos y datos

- $V=\{0,1,\dots,10\}$ nodos, $N=V\setminus\{0\}$ sitios, $A=\{(i,j)\in V\times V: i\neq j\}$, $|A|=110$.
- $c_{ij}=\kappa\, d^{\text{road}}_{ij}$ con $\kappa=1$ USD/km y $d^{\text{road}}_{ij}=1.60934\times\text{mediana}\{\texttt{trip\_distance}_r\}$.
- $d_i$ demanda de trabajo [min], $c^{\text{out}}_i$ costo unitario de tercerización [USD/min].
- $H=390$, $\bar o=30$, $c^{OT}=1.50$, $c^{EM}=6.00$, $\alpha_i=1$.
- $\xi^{(s)}_{ij}$: tiempo de viaje del arco $(i,j)$ en el escenario $s$, muestreado con reemplazo del pool $P_{ij}$.

## Primera etapa

Variables: $x_{ij}\in\{0,1\}$ indica si la ruta usa el arco $(i,j)$; $v_i\in[1,|N|]$ continuas son las variables de orden de Miller–Tucker–Zemlin.

$$\min_{x,v}\ \sum_{(i,j)\in A} c_{ij}x_{ij} + \mathcal{Q}_K(x)$$

sujeto a

$$\sum_{j\neq i} x_{ij}=1 \quad \forall i\in V \qquad \text{(sale una vez de cada nodo)}$$
$$\sum_{i\neq j} x_{ij}=1 \quad \forall j\in V \qquad \text{(entra una vez a cada nodo)}$$
$$v_i-v_j+|N|\,x_{ij}\le |N|-1 \quad \forall i,j\in N,\ i\neq j \qquad \text{(MTZ: elimina subtours)}$$
$$1\le v_i\le |N| \quad \forall i\in N, \qquad x_{ij}\in\{0,1\}.$$

Las restricciones de grado por sí solas admiten varios ciclos disjuntos. MTZ asigna a cada sitio una posición $v_i$ en el recorrido y obliga a que $v_j\ge v_i+1$ cuando se usa el arco $(i,j)$ entre sitios, lo que hace imposible cerrar un ciclo que no pase por el depósito.

## Segunda etapa (continua)

Para la ruta $x$ fija y el escenario $\xi_s$:

$$Q(x,\xi_s)=\min_{u,r,o,e}\ \sum_{i\in N} c^{\text{out}}_i r^{(s)}_i + c^{OT}o^{(s)} + c^{EM}e^{(s)}$$

sujeto a

$$u^{(s)}_i + r^{(s)}_i = d_i \quad \forall i\in N \qquad [\pi^{(s)}_i] \qquad \text{(toda la demanda se cubre)}$$
$$\sum_{(i,j)\in A}\xi^{(s)}_{ij}x_{ij} + \sum_{i\in N}\alpha_i u^{(s)}_i \le H + o^{(s)} + e^{(s)} \qquad [\lambda_s\le 0] \qquad \text{(presupuesto de tiempo)}$$
$$0\le u^{(s)}_i\le d_i,\quad r^{(s)}_i\ge 0,\quad 0\le o^{(s)}\le\bar o,\quad e^{(s)}\ge 0.$$

La aproximación por promedio muestral es $\mathcal{Q}_K(x)=\frac{1}{K}\sum_{s=1}^K Q(x,\xi_s)$.

Como $e^{(s)}$ no tiene cota superior, el recurso es **completo**: el subproblema es factible para toda ruta y todo escenario. Por lo tanto no se requieren cortes de factibilidad, y como todos los costos de recurso son no negativos, $L=0$ es una cota inferior global válida.

## Forma extensa MILP

Se replican las variables de segunda etapa $K$ veces y se resuelve un único modelo:

$$\min\ \sum_{(i,j)\in A} c_{ij}x_{ij} + \frac{1}{K}\sum_{s=1}^{K}\Big(\sum_{i\in N} c^{\text{out}}_i r^{(s)}_i + c^{OT}o^{(s)} + c^{EM}e^{(s)}\Big)$$

sujeto a las restricciones de ruteo y, para cada $s$, a las de segunda etapa.

## Propiedad estructural

El escenario entra en el subproblema únicamente a través del escalar

$$\tau_s(x)=\sum_{(i,j)\in A}\xi^{(s)}_{ij}x_{ij},$$

de modo que $Q(x,\xi_s)=\varphi(\tau_s(x))$ para una **única** función convexa lineal por tramos $\varphi$, la misma para todos los escenarios. Esto se aprovecha de dos maneras: los $K$ modelos LP se construyen una sola vez y solo se actualiza el lado derecho, y $\varphi$ se usa como verificación independiente de los valores y los duales.

## Cortes del Integer L-shaped

**Corte estándar de Benders**, agregado con probabilidad $1/K$. De la desigualdad de subgradiente $Q_s(\tau)\ge Q_s(\tau^\nu)-\lambda_s(\tau-\tau^\nu)$:

$$\theta \ \ge\ \frac{1}{K}\sum_{s}\big(Q(x^\nu,\xi_s)+\lambda_s\tau_s(x^\nu)\big)\ -\ \frac{1}{K}\sum_{s}\lambda_s\sum_{(i,j)\in A}\xi^{(s)}_{ij}x_{ij}.$$

**Corte de optimalidad entero** (Laporte y Louveaux) con $L=0$ y $S=\{(i,j):x^\nu_{ij}=1\}$:

$$\theta \ \ge\ \big(\mathcal{Q}_K(x^\nu)-L\big)\Big(\sum_{(i,j)\in S}x_{ij}-\sum_{(i,j)\notin S}x_{ij}-|S|+1\Big)+L.$$

Es ajustado en $x^\nu$ y se reduce a $\theta\ge L$ en cualquier otro punto binario. Ambos cortes son **globales**: se acumulan en un único maestro compartido por todos los nodos del árbol.

## Esquema del algoritmo

1. Seleccionar el nodo activo de **mejor cota**.
2. Resolver el maestro **LP** del nodo: ruteo relajado, $\theta\ge L$, cotas locales de ramificación y todos los cortes globales.
3. Podar si el nodo es infactible o si su cota alcanza el incumbente.
4. **Separación de cortes**: mientras $\theta$ subestime $\mathcal{Q}_K(x^\nu)$, agregar el corte de Benders violado y reoptimizar el nodo.
5. Si $x^\nu$ es **binaria**: evaluar el recurso exacto, actualizar el incumbente y agregar el corte de optimalidad entero.
6. Si es **fraccional**: ramificar sobre la variable de arco más cercana a $0.5$.
7. Parar cuando $\text{gap}=(UB-LB)/\max\{1,|UB|\}\le 10^{-3}$.
"""

CELDAS = [
    md(r"""
# Práctica 1 — Optimización Estocástica

**Ruteo urbano con tiempos de viaje estocásticos en Manhattan**

Universidad EAFIT · Maestría en Matemáticas Aplicadas · 2026-2

Programa estocástico lineal de dos etapas con primera etapa binaria (ruteo) y
segunda etapa continua (recurso). Se construye una aproximación por promedio
muestral (SAA), se valida la forma extensa MILP contra una implementación propia
del algoritmo *Integer L-shaped*, y se contrasta la solución estocástica con la
aproximación determinista del perfil promedio de tiempos.
"""),

    md("## 0. Preparación del entorno"),
    code("!pip install -q gurobipy polars pyarrow matplotlib"),

    md("""
### Módulos del proyecto

Las tres celdas siguientes escriben los módulos en el entorno de ejecución, de
modo que el notebook sea autocontenido y reproducible con *Run all*.

- `p1_datos.py`: descarga, filtros y construcción de los pools origen–destino.
- `p1_modelo.py`: forma extensa MILP, subproblemas de recurso e Integer L-shaped.
- `p1_experimentos.py`: orquestación de los experimentos y visualizaciones.
"""),
    celda_modulo("p1_datos.py"),
    celda_modulo("p1_modelo.py"),
    celda_modulo("p1_experimentos.py"),

    code("""
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import p1_datos as datos
import p1_modelo as modelo
import p1_experimentos as exp

SEMILLA_ENTRENAMIENTO = 29
SEMILLA_PRUEBA = 2027
K_CONSISTENCIA = 50
K_PRINCIPAL = 200
K_TEST = 1000
LIMITE_SEGUNDOS = 30 * 60

print("Semillas:", SEMILLA_ENTRENAMIENTO, SEMILLA_PRUEBA)
"""),

    md(FORMULACION),

    md("""
## 1. Datos de entrenamiento: enero de 2015

Se aplican los cuatro filtros del enunciado. El límite superior de la ventana de
fechas es **exclusivo** (`2015-02-01`), de modo que el último día del mes queda
incluido.
"""),
    code("""
archivo_enero = datos.descargar(datos.URL_ENERO)
registros_enero, pools_enero = datos.construir_pools(
    archivo_enero, datetime(2015, 1, 1),
    datetime(2015, 2, 1))

print(f"Registros tras los filtros: {len(registros_enero):,}")
pools_enero.head()
"""),
    code("""
# Control de calidad: deben aparecer 110 pools y el más pequeño 87 registros.
datos.control_calidad(pools_enero, n_arcos_esperado=110, min_registros_esperado=87)
"""),
    code("""
fig = exp.figura_pools(pools_enero)
plt.show()
"""),

    md("""
## 2. Prueba de consistencia con K = 50

La **misma** muestra se resuelve de dos maneras y los valores objetivo deben
coincidir dentro de una tolerancia relativa de $10^{-4}$. No se exige que las
rutas coincidan si existen óptimos alternativos.
"""),
    code("""
consistencia = exp.prueba_consistencia(
    pools_enero, K=K_CONSISTENCIA, semilla=SEMILLA_ENTRENAMIENTO,
    tiempo_limite=LIMITE_SEGUNDOS)
"""),

    md("""
### Verificación independiente del recurso

Como $Q(x,\\xi_s)=\\varphi(\\tau_s)$, se comparan los valores y los duales del LP
contra la forma cerrada de $\\varphi$.
"""),
    code("""
inst50 = consistencia["instancia"]
ev = modelo.EvaluadorRecurso(inst50)
_, _, Q_lp, info = ev.evaluar(consistencia["lshaped"]["x"], detalle=True)
tau = np.array([d["tau"] for d in info])
Q_cerrada, marginal = modelo.recurso_analitico(tau)
lam = np.array([ev.con_tiempo[s].Pi for s in range(inst50.K)])

print(f"max |Q_LP - Q_cerrada|          = {np.max(np.abs(Q_lp - Q_cerrada)):.3e}")
print(f"max |-lambda - costo marginal|  = {np.max(np.abs(-lam - marginal)):.3e}")
"""),

    md("""
## 3. Experimento principal (K = 200) y perfil promedio

Se resuelve primero la aproximación determinista del perfil promedio: además de
ser un entregable, su ruta sirve como incumbente inicial del Integer L-shaped.
Luego se comparan ambas rutas **sobre los mismos 200 escenarios**, sin
reoptimizar ninguna de las dos.
"""),
    code("""
principal = exp.experimento_principal(
    pools_enero, K=K_PRINCIPAL, semilla=SEMILLA_ENTRENAMIENTO,
    tiempo_limite=LIMITE_SEGUNDOS)

x_estrella = principal["x_estrella"]
x_promedio = principal["x_promedio"]
"""),
    code("""
fig = exp.figura_convergencia(principal["lshaped"]["registro"])
plt.show()
"""),
    code("""
ax = exp.figura_mapa_rutas(principal["lshaped"]["ruta"], principal["promedio"]["ruta"])
plt.show()
"""),

    md("""
## 4. Validación temporal fuera de muestra: febrero de 2015

Se repite el procedimiento de construcción de pools con las fechas de febrero,
**sin mezclarlos** con los de enero. Las rutas $x^\\star$ y $x_{\\text{prom}}$, las
distancias y los costos $c_{ij}$ quedan fijos: febrero solo aporta nuevos
escenarios de tiempos de viaje.
"""),
    code("""
archivo_febrero = datos.descargar(datos.URL_FEBRERO)
registros_febrero, pools_febrero = datos.construir_pools(
    archivo_febrero, datetime(2015, 2, 1),
    datetime(2015, 3, 1))

# Control de calidad: 110 pares dirigidos y el pool más pequeño con 90 registros.
datos.control_calidad(pools_febrero, n_arcos_esperado=110, min_registros_esperado=90)
"""),
    code("""
# Se alinea el orden de arcos y se conservan los costos de entrenamiento.
pools_febrero_alineados = datos.alinear_pools(pools_enero, pools_febrero)
_, c_entrenamiento, _ = modelo.pools_a_numpy(pools_enero)

validacion = exp.validacion_febrero(
    pools_febrero_alineados, c_entrenamiento, x_estrella, x_promedio,
    K_test=K_TEST, semilla=SEMILLA_PRUEBA)
"""),
    code("""
est = validacion["comparacion"]["x_estrella"]
prom = validacion["comparacion"]["x_promedio"]
fig = exp.figura_fuera_muestra(est, prom, validacion["comparacion"]["D"])
plt.show()
"""),

    md("## 5. Comparación de enfoques"),
    code("""
filas = exp.tabla_enfoques(consistencia, principal)
"""),
    code("""
fig = exp.figura_comparacion(est, prom, tiempos={
    "extensa K=50": consistencia["extenso"]["tiempo"],
    "L-shaped K=50": consistencia["lshaped"]["tiempo"],
    "L-shaped K=200": principal["lshaped"]["tiempo"],
    "perfil promedio": principal["promedio"]["tiempo"],
})
plt.show()
"""),

    md("""
### Exportar las figuras para el informe

Guarda las cinco visualizaciones con los nombres que espera `informe/informe.tex`
y las comprime para descargarlas.
"""),
    code("""
import os, zipfile

os.makedirs("figuras", exist_ok=True)

exportables = {
    "fig_pools.png": exp.figura_pools(pools_enero),
    "fig_convergencia.png": exp.figura_convergencia(principal["lshaped"]["registro"]),
    "fig_mapa.png": exp.figura_mapa_rutas(principal["lshaped"]["ruta"],
                                          principal["promedio"]["ruta"]).figure,
    "fig_fuera_muestra.png": exp.figura_fuera_muestra(
        est, prom, validacion["comparacion"]["D"]),
    "fig_comparacion.png": exp.figura_comparacion(est, prom, tiempos={
        "extensa K=50": consistencia["extenso"]["tiempo"],
        "L-shaped K=50": consistencia["lshaped"]["tiempo"],
        "L-shaped K=200": principal["lshaped"]["tiempo"],
        "perfil promedio": principal["promedio"]["tiempo"],
    }),
}
for nombre, fig in exportables.items():
    fig.savefig(os.path.join("figuras", nombre), dpi=200, bbox_inches="tight")
    plt.close(fig)

with zipfile.ZipFile("figuras_informe.zip", "w") as z:
    for nombre in exportables:
        z.write(os.path.join("figuras", nombre))

print("Figuras guardadas en figuras/ y comprimidas en figuras_informe.zip")
try:
    from google.colab import files
    files.download("figuras_informe.zip")
except Exception:
    pass
"""),

    md("""
## 6. Preguntas de interpretación

> Las respuestas se desarrollan en el informe técnico en PDF. Las celdas de esta
> sección producen la evidencia numérica que las sustenta.

1. **Compromiso entre distancia determinista y exposición a tiempos altos.**
2. **Escenarios que activan la tercerización y sitios que la concentran.**
3. **Qué aporta la evaluación fuera de muestra frente al valor óptimo SAA.**
4. **Qué se pierde al usar zonas TLC y al muestrear los arcos de forma independiente.**
5. **Por qué $\\bar{\\mathcal{Q}}_K(x)=Q(x,\\bar\\xi_K)$ y $\\mathcal{Q}_K(x)$ difieren** (convexidad y desigualdad de Jensen).
6. **Comparación de $x^\\star$ y $x_{\\text{prom}}$ con $VSS_K$, $\\Delta_{\\text{Feb}}$ y el intervalo pareado.**
"""),
    code("""
# Evidencia para la pregunta 5: desigualdad de Jensen.
inst200 = principal["instancia"]
ev200 = modelo.EvaluadorRecurso(inst200)
Q_saa, _, _, _ = ev200.evaluar(x_estrella)
Q_prom_perfil, _, _, _ = modelo.EvaluadorRecurso(inst200.promedio()).evaluar(x_estrella)

print(f"Q_K(x*)            = {Q_saa:.6f}   (promedio de los K recursos)")
print(f"Q(x*, xi_promedio) = {Q_prom_perfil:.6f}   (recurso en el perfil promedio)")
print(f"Diferencia (Jensen) = {Q_saa - Q_prom_perfil:.6f}  (debe ser >= 0 por convexidad)")
"""),
    code("""
# Evidencia para la pregunta 2: dónde se concentra la tercerización.
orden = np.argsort(-est["tercerizacion_por_sitio"])
print(f"{'sitio':<8}{'min. tercerizados':>20}{'c_out':>10}{'demanda':>10}")
for k in orden:
    i = modelo.SITIOS[k]
    print(f"{i:<8}{est['tercerizacion_por_sitio'][k]:>20.3f}"
          f"{modelo.C_OUT[i]:>10.2f}{modelo.DEMANDA[i]:>10.0f}")
"""),
    code("""
# Evidencia para la pregunta 1: costo de ruta frente a exposición temporal.
for nombre, d in (("x*", est), ("x_prom", prom)):
    tau = d["tau_por_escenario"]
    print(f"{nombre:<8} costo ruta = {d['costo_ruta']:8.3f} km/USD | "
          f"tau medio = {tau.mean():7.2f} min | "
          f"p90 = {np.percentile(tau, 90):7.2f} | Q = {d['Q_prom']:8.3f}")
"""),
]


def main():
    nb = {
        "cells": CELDAS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    destino = RAIZ / "p1_practica1.ipynb"
    destino.write_text(json.dumps(nb, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    print(f"Notebook generado: {destino.name} ({len(CELDAS)} celdas)")


if __name__ == "__main__":
    main()
