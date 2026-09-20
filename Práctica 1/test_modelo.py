"""Prueba de consistencia del Integer L-shaped contra la forma extensa.

Usa pools sintéticos con la misma estructura que la instancia real (11 nodos,
110 arcos, tiempos en minutos) para validar la lógica del algoritmo sin
depender de la descarga del archivo Parquet.
"""

import numpy as np

from p1_modelo import (
    NODOS, EvaluadorRecurso, Instancia, IntegerLShaped, comparar_rutas,
    evaluar_ruta, extraer_ruta, muestrear_escenarios, recurso_analitico,
    resolver_extenso,
)


def instancia_sintetica(K, seed=7, n_pool=120):
    """Pools sintéticos: tiempos lognormales crecientes con la distancia."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 10, size=(len(NODOS), 2))

    arcos, costos, pools = [], [], {}
    for i in NODOS:
        for j in NODOS:
            if i == j:
                continue
            dist = float(np.linalg.norm(coords[i] - coords[j])) + 0.5
            arcos.append((i, j))
            costos.append(dist)
            media = np.log(2.0 * dist)
            pools[(i, j)] = np.round(rng.lognormal(media, 0.35, size=n_pool), 0)

    xi = muestrear_escenarios(arcos, pools, K, seed + 1)
    return Instancia(arcos, costos, xi), pools


def test_consistencia(K):
    inst, _ = instancia_sintetica(K)
    print(f"\n{'=' * 62}\nK = {K}  |  |A| = {inst.nA}\n{'=' * 62}")

    ext = resolver_extenso(inst, tiempo_limite=600)
    print(f"Forma extensa   : obj = {ext['obj']:.6f}  "
          f"({ext['n_vars']} vars, {ext['n_cons']} restr, {ext['tiempo']:.1f}s)")
    print(f"  ruta = {ext['ruta']}")

    ls = IntegerLShaped(inst, L=0.0, tol_gap=1e-4, tiempo_limite=600, verbose=False)
    res = ls.solve()
    print(f"Integer L-shaped: obj = {res['obj']:.6f}  "
          f"({res['nodos']} nodos, {res['cortes_benders']} cortes Benders, "
          f"{res['cortes_enteros']} enteros, {res['tiempo']:.1f}s)")
    print(f"  ruta = {res['ruta']}  gap = {res['gap']:.2e}  motivo = {res['motivo']}")

    rel = abs(res["obj"] - ext["obj"]) / max(1.0, abs(ext["obj"]))
    print(f"Diferencia relativa: {rel:.3e}")
    assert rel <= 1e-4, f"Los valores objetivo no coinciden (rel = {rel:.3e})"
    assert res["ruta"] is not None, "La ruta del L-shaped no es un ciclo válido"
    assert ext["ruta"] is not None, "La ruta de la forma extensa no es válida"
    print("OK: consistencia dentro de 1e-4")
    return inst, ext, res


def test_ruta_valida(inst, x_vec):
    ruta = extraer_ruta(inst, x_vec)
    assert ruta is not None, "La ruta no es un ciclo hamiltoniano"
    assert ruta[0] == 0 and ruta[-1] == 0, "La ruta no empieza y termina en el depósito"
    assert sorted(ruta[:-1]) == NODOS, "La ruta no visita cada sitio exactamente una vez"
    print(f"OK: ruta hamiltoniana válida {ruta}")


def test_perfil_promedio(inst, x_estrella):
    """El perfil promedio debe dar una ruta evaluable y VSS >= 0 en muestra."""
    prom = resolver_extenso(inst.promedio(), tiempo_limite=300)
    print(f"\nPerfil promedio : obj (determinista) = {prom['obj']:.6f}")
    print(f"  ruta = {prom['ruta']}")

    cmp = comparar_rutas(inst, x_estrella, prom["x"])
    print(f"  C_hat(x*)     = {cmp['x_estrella']['costo_total']:.6f}")
    print(f"  C_hat(x_prom) = {cmp['x_promedio']['costo_total']:.6f}")
    print(f"  VSS_K         = {cmp['delta']:.6f}")
    print(f"  D promedio    = {cmp['D_media']:.6f}  IC95 = "
          f"({cmp['ic95'][0]:.4f}, {cmp['ic95'][1]:.4f})")
    assert cmp["delta"] >= -1e-6, "VSS negativo con x* certificada como óptima"
    assert abs(cmp["delta"] - cmp["D_media"]) < 1e-6, "VSS y D promedio deben coincidir"
    print("OK: VSS_K >= 0 y consistente con la media de diferencias pareadas")


def test_diagnosticos(inst, x_vec):
    d = evaluar_ruta(inst, x_vec)
    suma = d["costo_tercerizacion"] + d["costo_overtime"] + d["costo_emergencia"]
    assert abs(suma - d["Q_prom"]) < 1e-6, "La descomposición del recurso no suma Q"
    for clave in ("frec_overtime", "frec_tercerizacion", "frec_emergencia"):
        assert 0.0 <= d[clave] <= 1.0, f"{clave} fuera de [0,1]"
    print(f"\nDiagnósticos: Q = {d['Q_prom']:.4f} = terc {d['costo_tercerizacion']:.4f}"
          f" + OT {d['costo_overtime']:.4f} + EM {d['costo_emergencia']:.4f}")
    print(f"  frecuencias -> OT {d['frec_overtime']:.2f}, "
          f"terc {d['frec_tercerizacion']:.2f}, EM {d['frec_emergencia']:.2f}")
    print("OK: la descomposición del costo de recurso cuadra")


def test_recurso_analitico(inst, x_vec):
    """El LP y la forma cerrada deben coincidir en valor y en costo marginal."""
    evaluador = EvaluadorRecurso(inst)
    _, _, Q_lp, info = evaluador.evaluar(x_vec, detalle=True)
    tau = np.array([d["tau"] for d in info])
    Q_an, marg = recurso_analitico(tau)

    err_Q = float(np.max(np.abs(Q_lp - Q_an)))
    print(f"\nVerificación analítica: max |Q_LP - Q_cerrada| = {err_Q:.3e}")
    assert err_Q < 1e-6, "El LP y la forma cerrada del recurso no coinciden"

    # -lambda_s es el costo marginal del tiempo de viaje; debe estar en la escalera.
    lam = np.array([evaluador.con_tiempo[s].Pi for s in range(inst.K)])
    err_m = float(np.max(np.abs(-lam - marg)))
    print(f"  max |-lambda - costo marginal| = {err_m:.3e}")
    assert err_m < 1e-6, "Los duales no coinciden con el costo marginal teórico"
    print("OK: valores y duales del recurso verificados en forma cerrada")


if __name__ == "__main__":
    inst, ext, res = test_consistencia(K=10)
    test_ruta_valida(inst, res["x"])
    test_diagnosticos(inst, res["x"])
    test_recurso_analitico(inst, res["x"])
    test_perfil_promedio(inst, res["x"])

    test_consistencia(K=50)
    print("\nTodas las pruebas pasaron.")
