"""Práctica 1 - Optimización Estocástica (EAFIT, 2026-2).

Ruteo urbano con tiempos de viaje estocásticos en Manhattan.

Programa estocástico de dos etapas:
  1ra etapa: ruta binaria sobre V = {0,...,10} (grado + Miller-Tucker-Zemlin).
  2da etapa: recurso continuo por escenario (trabajo directo, tercerización,
             tiempo adicional ordinario y sobretiempo de emergencia).

La variable de emergencia e no tiene cota superior, por lo que el recurso es
completo: no se requieren cortes de factibilidad y L = 0 es una cota inferior
global válida para la función de recurso esperada.
"""

import heapq
import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB

# --------------------------------------------------------------------------
# Parámetros del experimento (sección 3 del enunciado)
# --------------------------------------------------------------------------
H = 390.0          # horizonte operativo regular [min]
O_BAR = 30.0       # tiempo adicional ordinario máximo [min]
C_OT = 1.50        # costo del tiempo adicional ordinario [USD/min]
C_EM = 6.00        # costo del sobretiempo de emergencia [USD/min]
KAPPA = 1.00       # costo de desplazamiento [USD/km]

N_SITIOS = 10
SITIOS = list(range(1, N_SITIOS + 1))
NODOS = list(range(0, N_SITIOS + 1))

DEMANDA = {1: 30.0, 2: 30.0, 3: 35.0, 4: 30.0, 5: 25.0,
           6: 25.0, 7: 35.0, 8: 40.0, 9: 35.0, 10: 25.0}

C_OUT = {1: 2.60, 2: 2.40, 3: 2.80, 4: 2.30, 5: 2.10,
         6: 2.00, 7: 2.70, 8: 3.00, 9: 2.90, 10: 2.20}

ALPHA = {i: 1.0 for i in SITIOS}

NOMBRES = {
    0: "Depósito: Javits Center", 1: "Times Square", 2: "Rockefeller Center",
    3: "Grand Central Terminal", 4: "New York Public Library", 5: "Union Square",
    6: "Washington Square Park", 7: "Madison Square Garden",
    8: "One World Trade Center", 9: "New York Stock Exchange",
    10: "South Street Seaport",
}


# --------------------------------------------------------------------------
# Instancia SAA
# --------------------------------------------------------------------------
class Instancia:
    """Arcos, costos de primera etapa y matriz de escenarios (K x |A|)."""

    def __init__(self, arcos, c, xi):
        self.arcos = [tuple(a) for a in arcos]
        self.nA = len(self.arcos)
        self.c = np.asarray(c, dtype=float)
        self.xi = np.atleast_2d(np.asarray(xi, dtype=float))
        self.K = self.xi.shape[0]
        self.pos = {a: k for k, a in enumerate(self.arcos)}
        if self.xi.shape[1] != self.nA:
            raise ValueError("xi debe tener |A| columnas")

    def promedio(self):
        """Instancia de un solo escenario con el perfil promedio de tiempos."""
        return Instancia(self.arcos, self.c, self.xi.mean(axis=0, keepdims=True))


def pools_a_numpy(pools_df):
    """Convierte el dataframe de pools (polars) en arcos, costos y pools.

    Espera las columnas i, j, c_ij y P_ij tal como las produce el notebook.
    """
    arcos, costos, pools = [], [], {}
    for fila in pools_df.iter_rows(named=True):
        arco = (int(fila["i"]), int(fila["j"]))
        arcos.append(arco)
        costos.append(float(fila["c_ij"]))
        pools[arco] = np.asarray(fila["P_ij"], dtype=float)
    return arcos, np.asarray(costos, dtype=float), pools


def muestrear_escenarios(arcos, pools, K, seed):
    """Muestrea K escenarios con reemplazo, de forma independiente entre arcos.

    Se usa un único generador de NumPy consumido arco por arco: esto garantiza
    reproducibilidad sin reutilizar la misma semilla en cada arco, que es lo que
    induciría dependencia entre componentes de xi.
    """
    rng = np.random.default_rng(seed)
    xi = np.empty((K, len(arcos)), dtype=float)
    for k, arco in enumerate(arcos):
        pool = pools[arco]
        xi[:, k] = pool[rng.integers(0, len(pool), size=K)]
    return xi


def construir_instancia(pools_df, K, seed):
    arcos, costos, pools = pools_a_numpy(pools_df)
    return Instancia(arcos, costos, muestrear_escenarios(arcos, pools, K, seed))


# --------------------------------------------------------------------------
# Restricciones de ruteo (comunes al MILP extenso y al maestro)
# --------------------------------------------------------------------------
def _restricciones_ruteo(modelo, x, v):
    """Grado de entrada/salida y eliminación de subtours por MTZ."""
    for i in NODOS:
        modelo.addConstr(gp.quicksum(x[(i, j)] for j in NODOS if j != i) == 1,
                         name=f"sale_{i}")
        modelo.addConstr(gp.quicksum(x[(j, i)] for j in NODOS if j != i) == 1,
                         name=f"entra_{i}")
    for i in SITIOS:
        for j in SITIOS:
            if i != j:
                modelo.addConstr(
                    v[i] - v[j] + N_SITIOS * x[(i, j)] <= N_SITIOS - 1,
                    name=f"mtz_{i}_{j}")


def extraer_ruta(inst, x_vec):
    """Devuelve la secuencia de nodos 0 -> ... -> 0 de una solución binaria."""
    siguiente = {}
    for k, (i, j) in enumerate(inst.arcos):
        if x_vec[k] > 0.5:
            siguiente[i] = j
    ruta, actual = [0], 0
    for _ in range(len(NODOS)):
        actual = siguiente.get(actual)
        if actual is None:
            return None
        ruta.append(actual)
        if actual == 0:
            return ruta if len(ruta) == len(NODOS) + 1 else None
    return None


# --------------------------------------------------------------------------
# Forma extensa MILP
# --------------------------------------------------------------------------
def resolver_extenso(inst, tiempo_limite=1800, mip_gap=1e-6, verbose=False):
    """Resuelve el equivalente determinista completo de la muestra SAA."""
    t0 = time.time()
    modelo = gp.Model("extenso")
    modelo.Params.OutputFlag = 1 if verbose else 0
    modelo.Params.TimeLimit = tiempo_limite
    modelo.Params.MIPGap = mip_gap

    x = modelo.addVars(inst.arcos, vtype=GRB.BINARY, name="x")
    v = modelo.addVars(SITIOS, lb=1.0, ub=float(N_SITIOS), name="v")
    _restricciones_ruteo(modelo, x, v)

    escenarios = range(inst.K)
    u = modelo.addVars(escenarios, SITIOS, lb=0.0, name="u")
    r = modelo.addVars(escenarios, SITIOS, lb=0.0, name="r")
    o = modelo.addVars(escenarios, lb=0.0, ub=O_BAR, name="o")
    e = modelo.addVars(escenarios, lb=0.0, name="e")

    for s in escenarios:
        for i in SITIOS:
            modelo.addConstr(u[s, i] + r[s, i] == DEMANDA[i], name=f"dem_{s}_{i}")
        modelo.addConstr(
            gp.quicksum(inst.xi[s, k] * x[a] for k, a in enumerate(inst.arcos))
            + gp.quicksum(ALPHA[i] * u[s, i] for i in SITIOS)
            <= H + o[s] + e[s],
            name=f"tiempo_{s}")

    modelo.setObjective(
        gp.quicksum(inst.c[k] * x[a] for k, a in enumerate(inst.arcos))
        + (1.0 / inst.K) * gp.quicksum(
            gp.quicksum(C_OUT[i] * r[s, i] for i in SITIOS)
            + C_OT * o[s] + C_EM * e[s]
            for s in escenarios),
        GRB.MINIMIZE)

    modelo.optimize()
    if modelo.SolCount == 0:
        raise RuntimeError("La forma extensa no encontró solución factible.")

    x_vec = np.array([x[a].X for a in inst.arcos])
    return {
        "x": np.round(x_vec),
        "obj": modelo.ObjVal,
        "cota": modelo.ObjBound,
        "gap": modelo.MIPGap,
        "tiempo": time.time() - t0,
        "n_vars": modelo.NumVars,
        "n_cons": modelo.NumConstrs,
        "n_bin": modelo.NumBinVars,
        "ruta": extraer_ruta(inst, x_vec),
    }


# --------------------------------------------------------------------------
# Segunda etapa: K subproblemas continuos reutilizados
# --------------------------------------------------------------------------
class EvaluadorRecurso:
    """Mantiene los K modelos LP de segunda etapa y solo actualiza su RHS.

    Q(x, xi_s) depende de x únicamente a través del escalar
    tau_s = sum_{(i,j)} xi_ij^(s) x_ij, así que basta reoptimizar cambiando el
    lado derecho de la restricción de tiempo. Esto evita reconstruir K modelos
    en cada separación de cortes.
    """

    def __init__(self, inst):
        self.inst = inst
        self.env = gp.Env(params={"OutputFlag": 0})
        self.modelos, self.con_tiempo = [], []
        self.u, self.r, self.o, self.e = [], [], [], []
        for _ in range(inst.K):
            m = gp.Model("sub", env=self.env)
            u = m.addVars(SITIOS, lb=0.0, name="u")
            r = m.addVars(SITIOS, lb=0.0, name="r")
            o = m.addVar(lb=0.0, ub=O_BAR, name="o")
            e = m.addVar(lb=0.0, name="e")
            for i in SITIOS:
                m.addConstr(u[i] + r[i] == DEMANDA[i], name=f"dem_{i}")
            ct = m.addConstr(
                gp.quicksum(ALPHA[i] * u[i] for i in SITIOS) - o - e <= H,
                name="tiempo")
            m.setObjective(
                gp.quicksum(C_OUT[i] * r[i] for i in SITIOS) + C_OT * o + C_EM * e,
                GRB.MINIMIZE)
            m.update()
            self.modelos.append(m)
            self.con_tiempo.append(ct)
            self.u.append(u)
            self.r.append(r)
            self.o.append(o)
            self.e.append(e)
        self.n_lps = 0

    def evaluar(self, x_vec, detalle=False):
        """Devuelve Q_K(x), el corte agregado y los valores por escenario.

        El corte es la pareja (const, coef) tal que  theta >= const + coef @ x.
        Se obtiene de la desigualdad de subgradiente
            Q_s(tau) >= Q_s(tau^v) - lambda_s (tau - tau^v),
        promediada sobre los K escenarios con probabilidad 1/K.
        """
        x_vec = np.asarray(x_vec, dtype=float)
        tau = self.inst.xi @ x_vec
        Q = np.empty(self.inst.K)
        lam = np.empty(self.inst.K)
        info = [] if detalle else None

        for s in range(self.inst.K):
            m = self.modelos[s]
            self.con_tiempo[s].RHS = H - tau[s]
            m.optimize()
            self.n_lps += 1
            if m.Status != GRB.OPTIMAL:
                raise RuntimeError(f"Subproblema {s} no óptimo (status {m.Status})")
            Q[s] = m.ObjVal
            lam[s] = self.con_tiempo[s].Pi
            if detalle:
                info.append({
                    "Q": m.ObjVal,
                    "tau": float(tau[s]),
                    "o": self.o[s].X,
                    "e": self.e[s].X,
                    "r": np.array([self.r[s][i].X for i in SITIOS]),
                    "u": np.array([self.u[s][i].X for i in SITIOS]),
                })

        const = float(np.mean(Q + lam * tau))
        coef = -(self.inst.xi * lam[:, None]).mean(axis=0)
        return float(Q.mean()), (const, coef), Q, info


# --------------------------------------------------------------------------
# Maestro del Integer L-shaped
# --------------------------------------------------------------------------
class MaestroRuteo:
    """Maestro LP por nodo: ruteo relajado, theta >= L y cortes globales."""

    def __init__(self, inst, theta_lb=0.0):
        self.inst = inst
        self.m = gp.Model("maestro")
        self.m.Params.OutputFlag = 0
        self.x = self.m.addVars(inst.arcos, lb=0.0, ub=1.0, name="x")
        self.v = self.m.addVars(SITIOS, lb=1.0, ub=float(N_SITIOS), name="v")
        self.theta = self.m.addVar(lb=theta_lb, name="theta")
        _restricciones_ruteo(self.m, self.x, self.v)
        self.xvars = [self.x[a] for a in inst.arcos]
        self.m.setObjective(
            gp.quicksum(inst.c[k] * self.xvars[k] for k in range(inst.nA))
            + self.theta,
            GRB.MINIMIZE)
        self.n_benders = 0
        self.n_enteros = 0

    def agregar_corte_benders(self, corte):
        const, coef = corte
        self.m.addConstr(
            self.theta >= const + gp.quicksum(
                float(coef[k]) * self.xvars[k] for k in range(self.inst.nA)))
        self.n_benders += 1

    def agregar_corte_entero(self, x_bin, Qx, L=0.0):
        """Corte de optimalidad entero de Laporte y Louveaux.

        theta >= (Q - L) (sum_{S} x - sum_{no S} x - |S| + 1) + L,
        con S el conjunto de arcos activos en x_bin. Es ajustado en x_bin y se
        reduce a theta >= L en cualquier otro punto binario.
        """
        S = [k for k in range(self.inst.nA) if x_bin[k] > 0.5]
        fuera = [k for k in range(self.inst.nA) if x_bin[k] <= 0.5]
        expr = (gp.quicksum(self.xvars[k] for k in S)
                - gp.quicksum(self.xvars[k] for k in fuera))
        self.m.addConstr(self.theta >= (Qx - L) * (expr - (len(S) - 1)) + L)
        self.n_enteros += 1

    def resolver(self, lb_v, ub_v):
        for k in range(self.inst.nA):
            self.xvars[k].LB = lb_v[k]
            self.xvars[k].UB = ub_v[k]
        self.m.optimize()
        if self.m.Status != GRB.OPTIMAL:
            return None
        x_vec = np.array([xv.X for xv in self.xvars])
        return x_vec, self.theta.X, self.m.ObjVal


# --------------------------------------------------------------------------
# Integer L-shaped
# --------------------------------------------------------------------------
class IntegerLShaped:
    """Branch-and-bound sobre los arcos con descomposición L-shaped por nodo.

    Selección de nodo por mejor cota, ramificación sobre la variable de arco más
    cercana a 0.5 y criterio de parada  (UB - LB) / max(1, |UB|) <= tol_gap.
    """

    def __init__(self, inst, L=0.0, tol_gap=1e-3, tol=1e-6,
                 tiempo_limite=1800.0, max_cortes_nodo=50, verbose=True,
                 x_inicial=None):
        self.inst = inst
        self.L = L
        self.tol_gap = tol_gap
        self.tol = tol
        self.tiempo_limite = tiempo_limite
        self.max_cortes_nodo = max_cortes_nodo
        self.verbose = verbose
        # Ruta factible con la que arrancar el incumbente. La relajación LP del
        # ruteo con MTZ es débil, así que entrar con una cota superior razonable
        # (por ejemplo la ruta del perfil promedio) reduce mucho la exploración.
        self.x_inicial = x_inicial
        self.registro = []

    def _es_binario(self, x_vec):
        return bool(np.all(np.minimum(x_vec, 1.0 - x_vec) <= 1e-6))

    def solve(self):
        t0 = time.time()
        maestro = MaestroRuteo(self.inst, theta_lb=self.L)
        evaluador = EvaluadorRecurso(self.inst)

        raiz = (np.zeros(self.inst.nA), np.ones(self.inst.nA))
        activos = [(-np.inf, 0, raiz)]
        contador, nodos = 1, 0
        UB, incumbente = np.inf, None
        LB = -np.inf
        motivo = "árbol agotado"

        if self.x_inicial is not None:
            x0 = np.round(np.asarray(self.x_inicial, dtype=float))
            Q0, corte0, _, _ = evaluador.evaluar(x0)
            UB = float(self.inst.c @ x0 + Q0)
            incumbente = x0
            maestro.agregar_corte_benders(corte0)
            maestro.agregar_corte_entero(x0, Q0, self.L)
            if self.verbose:
                print(f"  incumbente inicial: z = {UB:.4f}")

        while activos:
            if time.time() - t0 > self.tiempo_limite:
                motivo = "límite de tiempo"
                LB = activos[0][0] if activos else UB
                break

            cota, _, (lb_v, ub_v) = heapq.heappop(activos)
            LB = cota
            if np.isfinite(UB):
                gap = (UB - LB) / max(1.0, abs(UB))
                if gap <= self.tol_gap:
                    motivo = "tolerancia alcanzada"
                    break

            nodos += 1
            res = maestro.resolver(lb_v, ub_v)
            if res is None:
                continue
            x_vec, theta, obj_nodo = res
            if obj_nodo >= UB - self.tol:
                continue

            # Separación de cortes estándar antes de ramificar.
            Qk, corte, vigente = None, None, False
            for _ in range(self.max_cortes_nodo):
                Qk, corte, _, _ = evaluador.evaluar(x_vec)
                vigente = True  # Qk corresponde al x_vec actual
                if theta >= Qk - 1e-6 * max(1.0, abs(Qk)):
                    break
                maestro.agregar_corte_benders(corte)
                res = maestro.resolver(lb_v, ub_v)
                if res is None:
                    break
                x_vec, theta, obj_nodo = res
                vigente = False  # x_vec cambió: Qk quedó obsoleto
                if obj_nodo >= UB - self.tol:
                    break
            if res is None or obj_nodo >= UB - self.tol:
                continue

            if self._es_binario(x_vec):
                x_bin = np.round(x_vec)
                if not vigente:
                    Qk, corte, _, _ = evaluador.evaluar(x_bin)
                z = float(self.inst.c @ x_bin + Qk)
                if z < UB - self.tol:
                    UB, incumbente = z, x_bin.copy()
                    if self.verbose:
                        print(f"  nodo {nodos}: incumbente z = {z:.4f}")
                maestro.agregar_corte_entero(x_bin, Qk, self.L)
                maestro.agregar_corte_benders(corte)
                contador += 1
                heapq.heappush(activos, (obj_nodo, contador, (lb_v, ub_v)))
            else:
                k = int(np.argmin(np.abs(x_vec - 0.5)))
                lb0, ub0 = lb_v.copy(), ub_v.copy()
                ub0[k] = 0.0
                lb1, ub1 = lb_v.copy(), ub_v.copy()
                lb1[k] = 1.0
                contador += 1
                heapq.heappush(activos, (obj_nodo, contador, (lb0, ub0)))
                contador += 1
                heapq.heappush(activos, (obj_nodo, contador, (lb1, ub1)))

            gap_actual = ((UB - LB) / max(1.0, abs(UB))
                          if np.isfinite(UB) else np.inf)
            self.registro.append({
                "nodo": nodos,
                "tiempo": time.time() - t0,
                "LB": LB,
                "UB": UB,
                "gap": gap_actual,
                "cortes_benders": maestro.n_benders,
                "cortes_enteros": maestro.n_enteros,
                "cortes_total": maestro.n_benders + maestro.n_enteros,
            })
        else:
            LB = UB

        gap_final = ((UB - LB) / max(1.0, abs(UB))
                     if np.isfinite(UB) else np.inf)
        if self.verbose:
            print(f"Integer L-shaped: {motivo} | nodos = {nodos} | "
                  f"LB = {LB:.4f} | UB = {UB:.4f} | gap = {gap_final:.2e} | "
                  f"cortes = {maestro.n_benders}+{maestro.n_enteros}")

        return {
            "x": incumbente,
            "obj": UB,
            "LB": LB,
            "UB": UB,
            "gap": gap_final,
            "nodos": nodos,
            "motivo": motivo,
            "tiempo": time.time() - t0,
            "cortes_benders": maestro.n_benders,
            "cortes_enteros": maestro.n_enteros,
            "lps_resueltos": evaluador.n_lps,
            "registro": self.registro,
            "ruta": extraer_ruta(self.inst, incumbente) if incumbente is not None else None,
        }


# --------------------------------------------------------------------------
# Evaluación de una ruta fija sobre una muestra
# --------------------------------------------------------------------------
def evaluar_ruta(inst, x_vec, evaluador=None):
    """Calcula C_hat(x) = sum c_ij x_ij + (1/K) sum_s Q(x, xi_s) y diagnósticos.

    No reoptimiza la primera etapa: la ruta entra fija. Se usa tanto para el
    cálculo del VSS dentro de muestra como para la validación con febrero.
    """
    x_vec = np.asarray(x_vec, dtype=float)
    if evaluador is None:
        evaluador = EvaluadorRecurso(inst)
    Qk, _, Q_s, info = evaluador.evaluar(x_vec, detalle=True)

    costo_ruta = float(inst.c @ x_vec)
    o_s = np.array([d["o"] for d in info])
    e_s = np.array([d["e"] for d in info])
    r_s = np.vstack([d["r"] for d in info])

    return {
        "costo_total": costo_ruta + Qk,
        "costo_ruta": costo_ruta,
        "Q_prom": Qk,
        "costo_por_escenario": costo_ruta + Q_s,
        "Q_por_escenario": Q_s,
        "tau_por_escenario": np.array([d["tau"] for d in info]),
        "frec_overtime": float(np.mean(o_s > 1e-6)),
        "frec_tercerizacion": float(np.mean(r_s.sum(axis=1) > 1e-6)),
        "frec_emergencia": float(np.mean(e_s > 1e-6)),
        "tercerizacion_por_sitio": r_s.mean(axis=0),
        "costo_tercerizacion": float(np.mean(r_s @ np.array([C_OUT[i] for i in SITIOS]))),
        "costo_overtime": float(np.mean(C_OT * o_s)),
        "costo_emergencia": float(np.mean(C_EM * e_s)),
        "ruta": extraer_ruta(inst, x_vec),
    }


# --------------------------------------------------------------------------
# Verificación analítica del recurso
# --------------------------------------------------------------------------
D_TOTAL = sum(DEMANDA.values())

# Escalera de costos marginales: el tiempo adicional ordinario es lo más barato,
# luego la tercerización por sitio en orden creciente de c_out, y al final la
# emergencia, que es más cara que cualquier tercerización y no tiene cota.
_TRAMOS = ([(C_OT, O_BAR)]
           + sorted((C_OUT[i], DEMANDA[i]) for i in SITIOS)
           + [(C_EM, np.inf)])
_CAP_ACUM = np.cumsum([cap for _, cap in _TRAMOS])
_COSTO_TRAMO = np.array([costo for costo, _ in _TRAMOS])


def recurso_analitico(tau):
    """Evalúa Q(x, xi_s) en forma cerrada a partir del escalar tau.

    El subproblema de segunda etapa no depende del escenario salvo por
    tau_s = sum_ij xi_ij^(s) x_ij, de modo que Q(x, xi_s) = phi(tau_s) para una
    única función convexa lineal por tramos phi. Sirve como verificación
    independiente de los valores y los duales obtenidos con el LP.

    Solo es válida con alpha_i = 1 para todo sitio.
    """
    if any(abs(ALPHA[i] - 1.0) > 1e-12 for i in SITIOS):
        raise ValueError("recurso_analitico supone alpha_i = 1")

    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    deficit = np.maximum(tau + D_TOTAL - H, 0.0)

    Q = np.zeros_like(deficit)
    restante = deficit.copy()
    for costo, cap in _TRAMOS:
        usado = np.minimum(restante, cap)
        Q += costo * usado
        restante -= usado

    tramo = np.searchsorted(_CAP_ACUM, deficit, side="left")
    tramo = np.clip(tramo, 0, len(_TRAMOS) - 1)
    marginal = np.where(deficit > 0, _COSTO_TRAMO[tramo], 0.0)
    return Q, marginal


def comparar_rutas(inst, x_estrella, x_promedio):
    """Comparación pareada de dos rutas sobre los mismos escenarios."""
    evaluador = EvaluadorRecurso(inst)
    est = evaluar_ruta(inst, x_estrella, evaluador)
    prom = evaluar_ruta(inst, x_promedio, evaluador)

    D = prom["costo_por_escenario"] - est["costo_por_escenario"]
    n = len(D)
    media = float(D.mean())
    error = float(D.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return {
        "x_estrella": est,
        "x_promedio": prom,
        "delta": prom["costo_total"] - est["costo_total"],
        "D_media": media,
        "D_error_estandar": error,
        "ic95": (media - 1.96 * error, media + 1.96 * error),
        "D": D,
    }
