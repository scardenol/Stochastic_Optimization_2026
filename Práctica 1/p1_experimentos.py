"""Orquestación de los experimentos de la Práctica 1 y visualizaciones.

Cubre la sección 5 del enunciado: prueba de consistencia con K = 50,
experimento principal con K = 200, aproximación del perfil promedio,
VSS dentro de muestra y validación temporal con febrero de 2015.
"""

import numpy as np

from p1_datos import ZONAS
from p1_modelo import (
    C_OUT, SITIOS, EvaluadorRecurso, IntegerLShaped, Instancia, comparar_rutas,
    construir_instancia, evaluar_ruta, muestrear_escenarios, pools_a_numpy,
    resolver_extenso,
)

SEMILLA_ENTRENAMIENTO = 29
SEMILLA_PRUEBA = 2027


# --------------------------------------------------------------------------
# 5.1 Prueba de consistencia
# --------------------------------------------------------------------------
def prueba_consistencia(pools, K=50, semilla=SEMILLA_ENTRENAMIENTO,
                        tiempo_limite=1800, tol_relativa=1e-4):
    """Resuelve la MISMA muestra con la forma extensa y con Integer L-shaped."""
    inst = construir_instancia(pools, K=K, seed=semilla)

    ext = resolver_extenso(inst, tiempo_limite=tiempo_limite)
    calentamiento = resolver_extenso(inst.promedio(), tiempo_limite=tiempo_limite)
    ls = IntegerLShaped(inst, L=0.0, tol_gap=1e-4, tiempo_limite=tiempo_limite,
                        verbose=False, x_inicial=calentamiento["x"]).solve()

    rel = abs(ls["obj"] - ext["obj"]) / max(1.0, abs(ext["obj"]))
    print(f"Forma extensa    : {ext['obj']:.6f}  "
          f"({ext['n_vars']} vars, {ext['n_cons']} restr, {ext['tiempo']:.1f} s)")
    print(f"  ruta: {ext['ruta']}")
    print(f"Integer L-shaped : {ls['obj']:.6f}  "
          f"({ls['nodos']} nodos, {ls['cortes_benders']}+{ls['cortes_enteros']} cortes, "
          f"{ls['tiempo']:.1f} s)")
    print(f"  ruta: {ls['ruta']}")
    print(f"Diferencia relativa: {rel:.3e}  "
          f"({'COINCIDEN' if rel <= tol_relativa else 'NO COINCIDEN'} a {tol_relativa:g})")
    if ext["ruta"] != ls["ruta"]:
        print("  Nota: las rutas difieren; con costos iguales puede haber óptimos alternativos.")
    return {"instancia": inst, "extenso": ext, "lshaped": ls, "rel": rel}


# --------------------------------------------------------------------------
# 5.3 Experimento principal, perfil promedio y VSS
# --------------------------------------------------------------------------
def experimento_principal(pools, K=200, semilla=SEMILLA_ENTRENAMIENTO,
                          tiempo_limite=1800):
    inst = construir_instancia(pools, K=K, seed=semilla)

    # Se resuelve primero el perfil promedio: además de ser un entregable, su
    # ruta es un incumbente factible con el que arrancar el Integer L-shaped.
    print("Aproximación del perfil promedio (MILP determinista)")
    inst_prom = inst.promedio()
    prom = resolver_extenso(inst_prom, tiempo_limite=tiempo_limite)
    print(f"  x_prom: {prom['ruta']}")
    print(f"  objetivo determinista = {prom['obj']:.6f}  "
          f"({prom['n_vars']} vars, {prom['n_cons']} restr, {prom['tiempo']:.1f} s)")

    print(f"\nInteger L-shaped con K = {K} (límite {tiempo_limite / 60:.0f} min)")
    ls = IntegerLShaped(inst, L=0.0, tol_gap=1e-3, tiempo_limite=tiempo_limite,
                        verbose=True, x_inicial=prom["x"]).solve()
    print(f"  x*  : {ls['ruta']}")
    print(f"  LB = {ls['LB']:.6f}  UB = {ls['UB']:.6f}  gap = {ls['gap']:.3e}  "
          f"({ls['motivo']})")

    print("\nComparación justa sobre los MISMOS K escenarios (sin reoptimizar)")
    cmp = comparar_rutas(inst, ls["x"], prom["x"])
    print(f"  C_hat_K(x*)     = {cmp['x_estrella']['costo_total']:.6f}")
    print(f"  C_hat_K(x_prom) = {cmp['x_promedio']['costo_total']:.6f}")
    print(f"  VSS_K           = {cmp['delta']:.6f}")
    if ls["motivo"] != "límite de tiempo" and cmp["delta"] < -1e-6:
        print("  ATENCIÓN: VSS negativo con x* certificada; revisar.")

    return {"instancia": inst, "lshaped": ls, "promedio": prom,
            "comparacion": cmp, "x_estrella": ls["x"], "x_promedio": prom["x"]}


# --------------------------------------------------------------------------
# 5.3.4 Validación temporal fuera de muestra
# --------------------------------------------------------------------------
def validacion_febrero(pools_feb_alineados, c_entrenamiento, x_estrella,
                       x_promedio, K_test=1000, semilla=SEMILLA_PRUEBA):
    """Evalúa las dos rutas ya fijas sobre escenarios de febrero.

    Los costos c_ij y las distancias se mantienen en los valores de
    entrenamiento: febrero solo aporta los pools de tiempos de viaje.
    """
    arcos, _, pools_dict = pools_a_numpy(pools_feb_alineados)
    xi = muestrear_escenarios(arcos, pools_dict, K_test, semilla)
    inst = Instancia(arcos, c_entrenamiento, xi)

    cmp = comparar_rutas(inst, x_estrella, x_promedio)
    est, prom = cmp["x_estrella"], cmp["x_promedio"]

    print(f"Validación temporal con {K_test} escenarios de febrero de 2015")
    print(f"  C_hat_Feb(x*)     = {est['costo_total']:.6f}")
    print(f"  C_hat_Feb(x_prom) = {prom['costo_total']:.6f}")
    print(f"  Delta_Feb         = {cmp['delta']:.6f}")
    print(f"  D promedio        = {cmp['D_media']:.6f}  "
          f"(s.e. {cmp['D_error_estandar']:.6f})")
    print(f"  IC 95%            = ({cmp['ic95'][0]:.6f}, {cmp['ic95'][1]:.6f})")
    significativo = cmp["ic95"][0] > 0 or cmp["ic95"][1] < 0
    print(f"  El intervalo {'NO ' if not significativo else ''}excluye el cero: "
          f"{'hay' if significativo else 'no hay'} evidencia de diferencia.")

    print("\n  Frecuencias de activación del recurso")
    print(f"  {'ruta':<10}{'tiempo adic.':>14}{'terceriz.':>12}{'emergencia':>13}")
    for nombre, d in (("x*", est), ("x_prom", prom)):
        print(f"  {nombre:<10}{d['frec_overtime']:>14.3f}"
              f"{d['frec_tercerizacion']:>12.3f}{d['frec_emergencia']:>13.3f}")

    return {"instancia": inst, "comparacion": cmp, "significativo": significativo}


# --------------------------------------------------------------------------
# Tabla comparativa de enfoques
# --------------------------------------------------------------------------
def tabla_enfoques(consistencia, principal):
    ext, ls = consistencia["extenso"], consistencia["lshaped"]
    ls200, prom = principal["lshaped"], principal["promedio"]
    filas = [
        ("Forma extensa (K=50)", ext["obj"], ext["n_vars"], ext["n_cons"], ext["tiempo"]),
        ("Integer L-shaped (K=50)", ls["obj"], "maestro+K sub", "cortes dinámicos", ls["tiempo"]),
        ("Integer L-shaped (K=200)", ls200["obj"], "maestro+K sub", "cortes dinámicos", ls200["tiempo"]),
        ("Perfil promedio (MILP)", prom["obj"], prom["n_vars"], prom["n_cons"], prom["tiempo"]),
    ]
    ancho = max(len(f[0]) for f in filas) + 2
    print(f"{'Enfoque':<{ancho}}{'Objetivo':>14}{'Variables':>18}{'Restricciones':>18}{'Tiempo [s]':>12}")
    for nombre, obj, nv, nc, t in filas:
        print(f"{nombre:<{ancho}}{obj:>14.4f}{str(nv):>18}{str(nc):>18}{t:>12.2f}")
    print("\nNota: el objetivo del perfil promedio evalúa el recurso una sola vez en "
          "xi promedio\ny no es comparable directamente con los valores SAA.")
    return filas


# ==========================================================================
# Visualizaciones
# ==========================================================================
def _coords():
    lat = {z[0]: z[2] for z in ZONAS}
    lon = {z[0]: z[3] for z in ZONAS}
    nombre = {z[0]: z[1] for z in ZONAS}
    return lat, lon, nombre


def figura_mapa_rutas(ruta_estrella, ruta_promedio, ax=None):
    """Mapa comparativo de x* y x_prom con depósito, sitios y orden de visita."""
    import matplotlib.pyplot as plt

    lat, lon, nombre = _coords()
    iguales = ruta_estrella == ruta_promedio

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 9))

    for ruta, color, etiqueta, estilo, ancho in (
            (ruta_estrella, "#1f77b4", "x* (SAA)", "-", 2.2),
            (ruta_promedio, "#d62728", "x_prom (perfil promedio)", "--", 1.6)):
        if ruta is None:
            continue
        xs = [lon[n] for n in ruta]
        ys = [lat[n] for n in ruta]
        ax.plot(xs, ys, estilo, color=color, linewidth=ancho, label=etiqueta,
                alpha=0.85, zorder=2)

    for n in sorted(lat):
        es_deposito = n == 0
        ax.scatter(lon[n], lat[n], s=190 if es_deposito else 110,
                   marker="s" if es_deposito else "o",
                   color="black" if es_deposito else "white",
                   edgecolor="black", zorder=3)
        ax.annotate(str(n), (lon[n], lat[n]), ha="center", va="center",
                    fontsize=8, color="white" if es_deposito else "black", zorder=4)
        ax.annotate(nombre[n], (lon[n], lat[n]), textcoords="offset points",
                    xytext=(9, 7), fontsize=7, color="#444444", zorder=4)

    if iguales:
        ax.set_title("Rutas óptimas: x* y x_prom COINCIDEN", fontsize=11)
    else:
        ax.set_title("Rutas óptimas: x* frente a x_prom", fontsize=11)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.25)
    return ax


def figura_pools(pools, arcos_destacados=None):
    """Matriz de tamaños de pools y distribuciones de al menos cuatro arcos."""
    import matplotlib.pyplot as plt

    n = 11
    tam = np.full((n, n), np.nan)
    dist = {}
    for fila in pools.iter_rows(named=True):
        i, j = int(fila["i"]), int(fila["j"])
        tam[i, j] = fila["registros"]
        dist[(i, j)] = np.asarray(fila["P_ij"], dtype=float)

    if arcos_destacados is None:
        ordenados = sorted(dist, key=lambda a: -len(dist[a]))
        arcos_destacados = ordenados[:2] + ordenados[-2:]

    fig = plt.figure(figsize=(13, 5.5))
    ax0 = fig.add_subplot(1, 2, 1)
    im = ax0.imshow(tam, cmap="viridis")
    ax0.set_title(f"Tamaño de los {int(np.isfinite(tam).sum())} pools |R_ij|", fontsize=10)
    ax0.set_xlabel("destino j")
    ax0.set_ylabel("origen i")
    ax0.set_xticks(range(n))
    ax0.set_yticks(range(n))
    fig.colorbar(im, ax=ax0, fraction=0.046, label="registros")

    ax1 = fig.add_subplot(1, 2, 2)
    for arco in arcos_destacados:
        ax1.hist(dist[arco], bins=30, histtype="step", linewidth=1.6,
                 density=True, label=f"({arco[0]},{arco[1]})  n={len(dist[arco])}")
    ax1.set_title("Distribución empírica del tiempo de viaje", fontsize=10)
    ax1.set_xlabel("minutos")
    ax1.set_ylabel("densidad")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def figura_convergencia(registro):
    """Evolución de LB, UB, brecha y número acumulado de cortes."""
    import matplotlib.pyplot as plt

    if not registro:
        raise ValueError("El registro del algoritmo está vacío")
    t = np.array([r["tiempo"] for r in registro])
    lb = np.array([r["LB"] for r in registro], dtype=float)
    ub = np.array([r["UB"] for r in registro], dtype=float)
    gap = np.array([r["gap"] for r in registro], dtype=float)
    cortes = np.array([r["cortes_total"] for r in registro])

    finito = np.isfinite(lb)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    axes[0].plot(t[finito], lb[finito], label="LB", color="#1f77b4")
    axes[0].plot(t[np.isfinite(ub)], ub[np.isfinite(ub)], label="UB", color="#d62728")
    axes[0].set_title("Cotas globales", fontsize=10)
    axes[0].set_xlabel("tiempo [s]")
    axes[0].set_ylabel("costo")
    axes[0].legend(fontsize=8)

    valido = np.isfinite(gap) & (gap > 0)
    axes[1].semilogy(t[valido], gap[valido], color="#2ca02c")
    axes[1].axhline(1e-3, color="gray", linestyle=":", label="tolerancia 1e-3")
    axes[1].set_title("Brecha relativa", fontsize=10)
    axes[1].set_xlabel("tiempo [s]")
    axes[1].legend(fontsize=8)

    axes[2].plot(t, cortes, color="#9467bd")
    axes[2].set_title("Cortes acumulados", fontsize=10)
    axes[2].set_xlabel("tiempo [s]")
    axes[2].set_ylabel("cortes")

    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def figura_comparacion(est, prom, tiempos=None):
    """Costos, frecuencias de recurso y tiempos de cómputo de los enfoques."""
    import matplotlib.pyplot as plt

    n_paneles = 3 if tiempos else 2
    fig, axes = plt.subplots(1, n_paneles, figsize=(5.2 * n_paneles, 4.2))

    etiquetas = ["x*", "x_prom"]
    ancho = 0.35
    pos = np.arange(2)
    componentes = [
        ("ruta", [est["costo_ruta"], prom["costo_ruta"]], "#4c72b0"),
        ("tercerización", [est["costo_tercerizacion"], prom["costo_tercerizacion"]], "#dd8452"),
        ("tiempo adicional", [est["costo_overtime"], prom["costo_overtime"]], "#55a868"),
        ("emergencia", [est["costo_emergencia"], prom["costo_emergencia"]], "#c44e52"),
    ]
    base = np.zeros(2)
    for nombre, vals, color in componentes:
        axes[0].bar(pos, vals, ancho * 1.6, bottom=base, label=nombre, color=color)
        base += np.array(vals)
    axes[0].set_xticks(pos)
    axes[0].set_xticklabels(etiquetas)
    axes[0].set_title("Composición del costo esperado", fontsize=10)
    axes[0].set_ylabel("USD")
    axes[0].legend(fontsize=8)

    frec = [("tiempo adic.", "frec_overtime"), ("terceriz.", "frec_tercerizacion"),
            ("emergencia", "frec_emergencia")]
    pos2 = np.arange(len(frec))
    axes[1].bar(pos2 - ancho / 2, [est[k] for _, k in frec], ancho, label="x*")
    axes[1].bar(pos2 + ancho / 2, [prom[k] for _, k in frec], ancho, label="x_prom")
    axes[1].set_xticks(pos2)
    axes[1].set_xticklabels([n for n, _ in frec], fontsize=8)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Frecuencia de activación del recurso", fontsize=10)
    axes[1].legend(fontsize=8)

    if tiempos:
        nombres = list(tiempos)
        axes[2].barh(range(len(nombres)), [tiempos[n] for n in nombres], color="#8172b3")
        axes[2].set_yticks(range(len(nombres)))
        axes[2].set_yticklabels(nombres, fontsize=8)
        axes[2].set_xlabel("segundos")
        axes[2].set_title("Tiempo de cómputo", fontsize=10)

    for ax in axes:
        ax.grid(alpha=0.25, axis="both")
    fig.tight_layout()
    return fig


def figura_fuera_muestra(est, prom, D):
    """Composición del costo fuera de muestra y tercerización por sitio."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    pos = np.arange(len(SITIOS))
    ancho = 0.38
    axes[0].bar(pos - ancho / 2, est["tercerizacion_por_sitio"], ancho, label="x*")
    axes[0].bar(pos + ancho / 2, prom["tercerizacion_por_sitio"], ancho, label="x_prom")
    axes[0].set_xticks(pos)
    axes[0].set_xticklabels(SITIOS)
    axes[0].set_xlabel("sitio i")
    axes[0].set_ylabel("minutos promedio")
    axes[0].set_title("Tercerización promedio por sitio (febrero)", fontsize=10)
    axes[0].legend(fontsize=8)

    ax2 = axes[0].twinx()
    ax2.plot(pos, [C_OUT[i] for i in SITIOS], "k.--", linewidth=1, markersize=7,
             label="c_out")
    ax2.set_ylabel("c_out [USD/min]", fontsize=8)
    ax2.legend(fontsize=7, loc="upper right")

    axes[1].hist(est["costo_por_escenario"], bins=40, alpha=0.6, label="x*")
    axes[1].hist(prom["costo_por_escenario"], bins=40, alpha=0.6, label="x_prom")
    axes[1].set_xlabel("costo por escenario [USD]")
    axes[1].set_title("Distribución del costo fuera de muestra", fontsize=10)
    axes[1].legend(fontsize=8)

    media = float(np.mean(D))
    error = float(np.std(D, ddof=1) / np.sqrt(len(D)))
    axes[2].hist(D, bins=40, color="#937860")
    axes[2].axvline(0, color="black", linestyle=":", linewidth=1)
    axes[2].axvline(media, color="#c44e52", linewidth=1.8,
                    label=f"media {media:.3f}")
    axes[2].axvspan(media - 1.96 * error, media + 1.96 * error, color="#c44e52",
                    alpha=0.2, label="IC 95%")
    axes[2].set_xlabel("D = C(x_prom) - C(x*)")
    axes[2].set_title("Diferencias pareadas", fontsize=10)
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig
