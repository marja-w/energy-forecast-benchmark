import polars as pl
from loguru import logger
from sklearn.cluster import AgglomerativeClustering

from src.energy_forecast.config import REPORTS_DIR, PROCESSED_DATA_DIR
from src.energy_forecast.plots import plot_clusters


def hierarchical_clustering_on_meta_data(df: pl.DataFrame, n_clusters: int) -> pl.DataFrame:
    """
    Perform hierarchical clustering on meta data, like average, min, and max values of consumption of a building.
    :param df: DataFrame with consumption information
    :param n_clusters: number of clusters
    :return: DataFrame with added column "labels" represetning the clusters
    """
    logger.info(f"Computing {n_clusters} clusters")
    logger.info(f"Length of df: {len(df)}")
    df = df.group_by("id").agg(pl.col("diff"),
                               pl.col("diff").mean().alias("avg"),
                               pl.col("diff").std().alias("std"),
                               pl.col("diff").median().alias("median"),
                               pl.col("diff").min().alias("min"),
                               pl.col("diff").max().alias("max"),
                               pl.col("diff").head(30).alias("month"),
                               pl.len()
                               ).sort("len")
    consumption_info_csv_ = REPORTS_DIR / "buildings_consumption_info.csv"
    df.drop(["diff", "month"]).write_csv(consumption_info_csv_)

    df = df.filter(pl.col("len") > 14)
    logger.info(f"Number of series after filtering: {len(df)}")
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
        logger.info(f"Average consumption: {df.filter(pl.col('label') == c_id)['avg'].mean()}")
    return df

if __name__ == "__main__":
    attr = ["id", "diff", "datetime"]
    df_daily = pl.read_csv(PROCESSED_DATA_DIR / "building_daily_7_7.csv").select(attr).with_columns(pl.col("datetime").str.to_datetime())
    df_hourly = pl.read_csv(PROCESSED_DATA_DIR / "building_hourly_72_72.csv").with_columns(pl.col("datetime").str.to_datetime()).select(attr)
    df_hourly = df_hourly.group_by_dynamic(index_column="datetime", every="1d", group_by=["id"]).agg([pl.col("diff").sum().alias("diff")]).select(attr)
    df = pl.concat([df_daily, df_hourly])
    df = df.with_columns(
        pl.col("id").str.replace_all("(-\d+)?-\d+$", "").alias("id")
    )
    df = df.unique(["id", "datetime"]).drop(["datetime"])  # remove duplicates
    hierarchical_clustering_on_meta_data(df, 3)