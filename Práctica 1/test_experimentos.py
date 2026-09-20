"""Prueba de extremo a extremo del pipeline de experimentos y figuras.

Usa pools sintéticos con la estructura real (11 nodos, 110 arcos) y valores de K
pequeños, para validar la orquestación completa sin descargar los Parquet.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import polars as pl

from p1_datos import ZONAS
from p1_experimentos import (
    experimento_principal, figura_comparacion, figura_convergencia,
    figura_fuera_muestra, figura_mapa_rutas, figura_pools, prueba_consistencia,
    tabla_enfoques, validacion_febrero,
)
from p1_modelo import pools_a_numpy


def pools_sinteticos(seed=11, n_pool=100, escala=1.0):
    rng = np.random.default_rng(seed)
    coords = np.array([[z[2], z[3]] for z in ZONAS])
    filas = []
    for i in range(11):
        for j in range(11):
            if i == j:
                continue
            km = float(np.linalg.norm(coords[i] - coords[j])) * 111.0 + 0.4
            media = np.log(2.2 * km * escala)
            filas.append({
                "i": i, "j": j,
                "d_road_ij": km,
                "c_ij": km,
                "P_ij": list(np.round(rng.lognormal(media, 0.35, size=n_pool), 1)),
                "registros": n_pool,
            })
    return pl.DataFrame(filas)


def main():
    enero = pools_sinteticos(seed=11)
    febrero = pools_sinteticos(seed=77, escala=1.08)  # ligero desplazamiento temporal

    print("=" * 70)
    print("5.1 Prueba de consistencia")
    print("=" * 70)
    cons = prueba_consistencia(enero, K=8, tiempo_limite=300)
    assert cons["rel"] <= 1e-4, "La consistencia MILP vs L-shaped falló"

    print("\n" + "=" * 70)
    print("5.3 Experimento principal")
    print("=" * 70)
    prin = experimento_principal(enero, K=12, tiempo_limite=300)
    assert prin["x_estrella"] is not None, "No se obtuvo x*"
    assert prin["comparacion"]["delta"] >= -1e-6, "VSS negativo con x* certificada"

    print("\n" + "=" * 70)
    print("5.3.4 Validación temporal")
    print("=" * 70)
    _, c_entrenamiento, _ = pools_a_numpy(enero)
    val = validacion_febrero(febrero, c_entrenamiento, prin["x_estrella"],
                             prin["x_promedio"], K_test=200)

    print("\n" + "=" * 70)
    print("Comparación de enfoques")
    print("=" * 70)
    tabla_enfoques(cons, prin)

    print("\nGenerando figuras...")
    est = val["comparacion"]["x_estrella"]
    prom = val["comparacion"]["x_promedio"]

    figs = {
        "fig_mapa.png": figura_mapa_rutas(prin["lshaped"]["ruta"],
                                          prin["promedio"]["ruta"]).figure,
        "fig_pools.png": figura_pools(enero),
        "fig_convergencia.png": figura_convergencia(prin["lshaped"]["registro"]),
        "fig_comparacion.png": figura_comparacion(
            est, prom,
            tiempos={"extensa K=50": cons["extenso"]["tiempo"],
                     "L-shaped K=50": cons["lshaped"]["tiempo"],
                     "L-shaped K=200": prin["lshaped"]["tiempo"],
                     "perfil promedio": prin["promedio"]["tiempo"]}),
        "fig_fuera_muestra.png": figura_fuera_muestra(
            est, prom, val["comparacion"]["D"]),
    }
    for nombre, fig in figs.items():
        fig.savefig(nombre, dpi=110, bbox_inches="tight")
        print(f"  {nombre}")

    print("\nTodas las pruebas de extremo a extremo pasaron.")


if __name__ == "__main__":
    main()
