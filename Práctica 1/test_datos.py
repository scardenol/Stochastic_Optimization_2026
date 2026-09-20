"""Prueba del pipeline de datos con un parquet sintético pequeño.

No valida las cifras reales (110 pools, mínimo 87 registros), que dependen del
archivo oficial de la TLC; valida que los cuatro filtros y la agregación se
comporten como especifica el enunciado.
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from p1_datos import ZONAS, construir_pools, zonas_dataframe


def parquet_sintetico(ruta, n=60000, seed=3):
    rng = np.random.default_rng(seed)
    loc_ids = [z[4] for z in ZONAS]

    base = datetime(2015, 1, 1)
    minutos = rng.integers(0, 60 * 24 * 59, size=n)  # enero y febrero
    pickup = [base + timedelta(minutes=int(m)) for m in minutos]
    duracion = rng.integers(1, 130, size=n)
    dropoff = [p + timedelta(minutes=int(d)) for p, d in zip(pickup, duracion)]

    pl.DataFrame({
        "tpep_pickup_datetime": pickup,
        "tpep_dropoff_datetime": dropoff,
        "PULocationID": rng.choice(loc_ids + [999], size=n),
        "DOLocationID": rng.choice(loc_ids + [999], size=n),
        "trip_distance": rng.uniform(0.0, 40.0, size=n),
    }).write_parquet(ruta)
    return ruta


def test_filtros(ruta):
    datos, pools = construir_pools(ruta, datetime(2015, 1, 1), datetime(2015, 2, 1))
    print(f"Registros tras filtros: {len(datos):,}  |  pools: {pools.shape[0]}")

    pu = datos["tpep_pickup_datetime"]
    assert pu.min() >= datetime(2015, 1, 1), "hay pickups antes del inicio"
    assert pu.max() < datetime(2015, 2, 1), "el límite superior no es exclusivo"
    assert datos["tpep_pickup_datetime"].dt.weekday().max() <= 5, "hay fin de semana"
    horas = datos["tpep_pickup_datetime"].dt.hour()
    assert horas.min() >= 9 and horas.max() <= 16, "franja horaria incorrecta"
    assert datos["t"].min() >= 1 and datos["t"].max() <= 90, "duración fuera de rango"
    assert datos["trip_distance"].min() >= 0.1, "distancia menor a 0.1"
    assert datos["trip_distance"].max() <= 30, "distancia mayor a 30"
    assert (datos["PULocationID"] != datos["DOLocationID"]).all(), "hay viajes i = j"
    assert 999 not in datos["PULocationID"].to_list(), "zona fuera de la tabla"
    print("OK: los cuatro filtros se aplican correctamente")

    assert pools["i"].max() <= 10 and pools["j"].max() <= 10, "índices fuera de V"
    assert (pools["i"] != pools["j"]).all(), "hay arcos i = j"
    assert pools["registros"].min() >= 1
    largos = [len(p) for p in pools["P_ij"].to_list()]
    assert largos == pools["registros"].to_list(), "|P_ij| no coincide con registros"
    assert (pools["c_ij"] > 0).all(), "hay costos no positivos"
    print("OK: pools bien formados y consistentes con |R_ij|")


def test_dia_final_incluido(ruta):
    """El 30 de enero de 2015 fue viernes: debe aparecer en los datos."""
    datos, _ = construir_pools(ruta, datetime(2015, 1, 1), datetime(2015, 2, 1))
    dias = set(datos["tpep_pickup_datetime"].dt.day().to_list())
    assert 30 in dias, "se perdió el último día hábil del mes"
    print("OK: el último día hábil del mes queda incluido")


def test_separacion_meses(ruta):
    _, enero = construir_pools(ruta, datetime(2015, 1, 1), datetime(2015, 2, 1))
    _, febrero = construir_pools(ruta, datetime(2015, 2, 1), datetime(2015, 3, 1))
    tot_e = sum(enero["registros"].to_list())
    tot_f = sum(febrero["registros"].to_list())
    assert tot_e > 0 and tot_f > 0, "algún mes quedó vacío"
    print(f"OK: pools separados por mes (enero {tot_e:,} / febrero {tot_f:,})")


if __name__ == "__main__":
    ruta = parquet_sintetico("_sintetico.parquet")
    test_filtros(ruta)
    test_dia_final_incluido(ruta)
    test_separacion_meses(ruta)
    print(f"\nZonas cargadas: {zonas_dataframe().shape[0]}")
    print("Todas las pruebas de datos pasaron.")
