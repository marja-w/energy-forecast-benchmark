import polars as pl
from loguru import logger
from sklearn.cluster import AgglomerativeClustering

from src.energy_forecast.config import REPORTS_DIR
from src.energy_forecast.plots import plot_clusters


def hierarchical_clustering_on_meta_data(df: pl.DataFrame, n_clusters: int) -> pl.DataFrame:
    """
    Perform hierarchical clustering on meta data, like average, min, and max values of consumption of a building.
    :param df: DataFrame with consumption information
    :param n_clusters: number of clusters
    :return: DataFrame with added column "labels" represetning the clusters
    """
    logger.info(f"Computing {n_clusters} clusters")
    df = df.group_by("id").agg(pl.col("diff"),
                               pl.col("daily_avg").mode().first().alias("avg"),
                               pl.col("diff").std().alias("std"),
                               pl.col("diff").median().alias("median"),
                               pl.col("diff").min().alias("min"),
                               pl.col("diff").max().alias("max"),
                               pl.col("diff").head(30).alias("month"),
                               pl.len()
                               ).sort("len")
    consumption_info_csv_ = REPORTS_DIR / "buildings_consumption_info.csv"
    df.drop(["diff", "month"]).write_csv(consumption_info_csv_)
    logger.info(f"Writing building consumption info to {consumption_info_csv_}")
    data = df.drop(["id", "diff", "month", "len"]).to_numpy()
    # Create the AgglomerativeClustering model
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    # Fit the model and predict cluster labels
    labels = model.fit_predict(data)
    plot_clusters(df, labels)
    # add label column to dataframe
    df = df.with_columns(pl.Series(labels).alias("label"))
    for c_id in range(n_clusters):
        logger.info(f"Computed Cluster {c_id} with n={len(df.filter(pl.col('label') == c_id))}")
    return df
