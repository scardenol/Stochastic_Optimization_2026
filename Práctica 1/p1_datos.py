"""Construcción de pools origen-destino a partir de los registros de la TLC.

El mismo procedimiento se aplica a enero (entrenamiento) y a febrero
(validación temporal fuera de muestra), sin mezclar los pools.
"""

import urllib.request
from datetime import datetime, time

import polars as pl

KAPPA = 1.00        # USD/km
MILLA_A_KM = 1.60934

ZONAS = [
    (0, "Depósito: Javits Center", 40.75750, -74.00250, 246),
    (1, "Times Square", 40.75800, -73.98550, 230),
    (2, "Rockefeller Center", 40.75870, -73.97870, 161),
    (3, "Grand Central Terminal", 40.75278, -73.97722, 162),
    (4, "New York Public Library (Main)", 40.75306, -73.98194, 164),
    (5, "Union Square", 40.73590, -73.99110, 234),
    (6, "Washington Square Park", 40.73083, -73.99750, 114),
    (7, "Madison Square Garden", 40.75056, -73.99361, 186),
    (8, "One World Trade Center", 40.71274, -74.01338, 261),
    (9, "New York Stock Exchange", 40.70693, -74.01125, 87),
    (10, "South Street Seaport (Pier 17)", 40.70600, -74.00270, 209),
]

ESQUEMA_ZONAS = {
    "i": pl.Int64, "Sitio": pl.Utf8, "Latitud": pl.Float64,
    "Longitud": pl.Float64, "LocationID": pl.Int64,
}

URL_ENERO = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2015-01.parquet"
URL_FEBRERO = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2015-02.parquet"


def zonas_dataframe():
    return pl.DataFrame(ZONAS, schema=ESQUEMA_ZONAS, orient="row")


def descargar(url, destino=None):
    destino = destino or url.rsplit("/", 1)[-1]
    urllib.request.urlretrieve(url, destino)
    return destino


def construir_pools(archivo, inicio, fin_exclusivo, zonas_df=None):
    """Aplica los cuatro filtros del enunciado y devuelve los pools por arco.

    El límite superior es EXCLUSIVO: para enero se pasa 2015-02-01 y para
    febrero 2015-03-01, de modo que el último día del mes queda incluido.
    """
    zonas_df = zonas_dataframe() if zonas_df is None else zonas_df
    loc_ids = zonas_df["LocationID"].to_list()

    datos = (
        pl.scan_parquet(archivo)
        .select(["tpep_pickup_datetime", "tpep_dropoff_datetime",
                 "PULocationID", "DOLocationID", "trip_distance"])
        # 1) ventana de fechas, lunes a viernes, franja [09:00, 17:00)
        .filter(
            pl.col("tpep_pickup_datetime") >= inicio,
            pl.col("tpep_pickup_datetime") < fin_exclusivo,
            pl.col("tpep_pickup_datetime").dt.weekday().is_between(1, 5),
            pl.col("tpep_pickup_datetime").dt.time() >= time(9, 0),
            pl.col("tpep_pickup_datetime").dt.time() < time(17, 0),
        )
        # 2) origen y destino en zonas distintas de la tabla
        .filter(
            pl.col("PULocationID").is_in(loc_ids),
            pl.col("DOLocationID").is_in(loc_ids),
            pl.col("PULocationID") != pl.col("DOLocationID"),
        )
        # 3) duración en minutos dentro de [1, 90]
        .with_columns(
            ((pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime"))
             .dt.total_seconds() / 60.0).alias("t")
        )
        .filter(pl.col("t").is_between(1, 90))
        # 4) distancia entre 0.1 y 30 millas
        .filter(pl.col("trip_distance").is_between(0.1, 30))
        .collect()
    )

    pools = (
        datos.group_by(["PULocationID", "DOLocationID"])
        .agg(
            pl.col("t").alias("P_ij"),
            pl.len().alias("registros"),
            (pl.col("trip_distance").median() * MILLA_A_KM).alias("d_road_ij"),
        )
        .with_columns((KAPPA * pl.col("d_road_ij")).alias("c_ij"))
        .join(zonas_df.select([pl.col("LocationID").alias("PULocationID"), pl.col("i")]),
              on="PULocationID")
        .join(zonas_df.select([pl.col("LocationID").alias("DOLocationID"),
                               pl.col("i").alias("j")]),
              on="DOLocationID")
        .select(["i", "j", "d_road_ij", "c_ij", "P_ij", "registros"])
        .sort(["i", "j"])
    )
    return datos, pools


def control_calidad(pools, n_arcos_esperado=110, min_registros_esperado=None):
    n = pools.shape[0]
    minimo = int(pools["registros"].min())
    print(f"Número de pools: |A| = {n} (esperado {n_arcos_esperado})")
    print(f"Pool más pequeño: {minimo} registros", end="")
    if min_registros_esperado is not None:
        print(f" (esperado {min_registros_esperado})")
    else:
        print()
    ok = n == n_arcos_esperado
    if min_registros_esperado is not None:
        ok = ok and minimo == min_registros_esperado
    print("Control de calidad:", "OK" if ok else "REVISAR")
    return ok


def pools_enero(archivo=None):
    archivo = archivo or descargar(URL_ENERO)
    return construir_pools(archivo, datetime(2015, 1, 1), datetime(2015, 2, 1))


def pools_febrero(archivo=None):
    archivo = archivo or descargar(URL_FEBRERO)
    return construir_pools(archivo, datetime(2015, 2, 1), datetime(2015, 3, 1))


def alinear_pools(pools_ref, pools_nuevos):
    """Reordena pools_nuevos según el orden de arcos de pools_ref.

    Las distancias y los costos c_ij se mantienen fijos en los de entrenamiento,
    tal como exige el enunciado para la validación temporal.
    """
    orden = pools_ref.select(["i", "j", "d_road_ij", "c_ij"])
    return (orden.join(pools_nuevos.select(["i", "j", "P_ij", "registros"]),
                       on=["i", "j"], how="left")
            .select(["i", "j", "d_road_ij", "c_ij", "P_ij", "registros"])
            .sort(["i", "j"]))
