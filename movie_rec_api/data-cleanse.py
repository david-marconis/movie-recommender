import os
import shutil
import sys
from enum import Enum, auto
from glob import glob
from typing import Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, expr, regexp_extract


class OutputFormat(Enum):
    CSV = auto()
    PARQUET = auto()

    def __str__(self):
        return self.name.lower()


def write_output(data_frame, out_format, output_dir):
    writer = data_frame.write.mode("overwrite").option(
        "mapreduce.fileoutputcommitter.marksuccessfuljobs", "false"
    )
    if out_format == OutputFormat.CSV:
        writer.csv(output_dir, header=True)
    elif out_format == OutputFormat.PARQUET:
        writer.parquet(output_dir)
    else:
        exit_with_error(f"Output format not implemented: {out_format}")


def exit_with_error(error):
    print(error, file=sys.stderr)
    print(f"Usage: {sys.argv[0]} <dataset-dir> <output-dir> <out-format>")
    print("Output formats: csv, parquet. Default is parquet")
    sys.exit(1)


def load_ml100k_dataset(
    input_dir: str, spark: SparkSession
) -> Tuple[DataFrame, DataFrame]:
    metadata_schema = "movieId int, title string"
    metadata = spark.read.csv(f"{input_dir}/u.item", sep="|", schema=metadata_schema)

    # Extract the year from the title
    metadata = (
        metadata.filter(col("movieId").cast("int").isNotNull())
        .select(
            col("movieId"),
            regexp_extract(col("title"), r"\((\d{4})\)$", 1).alias("year"),
            expr("substring(title, 1, length(title) - 7)").alias("title"),
        )
        .coalesce(1)
    )

    ratings_schema = "userId int, movieId int, rating float"
    ratings = spark.read.csv(
        f"{input_dir}/u.data", sep="\t", schema=ratings_schema
    ).coalesce(1)
    return metadata, ratings


def load_the_movies_dataset(
    input_dir: str, spark: SparkSession
) -> Tuple[DataFrame, DataFrame]:
    metadata = spark.read.csv(
        f"{input_dir}/movies_metadata.csv", header=True, inferSchema=True
    )
    ratings = spark.read.csv(f"{input_dir}/ratings.csv", header=True, inferSchema=True)
    popular_movies = (
        ratings.groupBy("movieId").count().filter("count > 10").select("movieId")
    )
    cleansed_metadata = (
        metadata.withColumn("year", col("release_date").substr(1, 4).cast("int"))
        .withColumn("vote_count", col("vote_count").cast("int"))
        .withColumn("movieId", col("id").cast("long"))
        .select(["movieId", "year", "title"])
        .filter(col("year").isNotNull())
        .filter("adult = false AND vote_count > 10")
        .join(popular_movies, "movieId", "inner")
        .coalesce(1)
    )
    cleansed_ratings = (
        cleansed_metadata.select("movieId")
        .join(ratings, "movieId", "inner")
        .select(["userId", "movieId", "rating"])
        .filter("rating >= 1 AND rating <= 5")
        .coalesce(1)
    )
    return cleansed_metadata, cleansed_ratings


def main():
    if len(sys.argv) < 3:
        exit_with_error("Invalid number of parameters")
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    output_format = len(sys.argv) > 3 and sys.argv[3] or "parquet"
    try:
        output_format = OutputFormat[output_format.upper()]
    except KeyError:
        exit_with_error(f"Unsupported output format: {output_format}")

    spark = (
        SparkSession.builder.config(
            "spark.driver.extraJavaOptions",
            "-Dlog4j.configuration=file:log4j.properties",
        )
        .config(
            "spark.executor.extraJavaOptions",
            "-Dlog4j.configuration=file:log4j.properties",
        )
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    if os.path.exists(f"{input_dir}/u.data"):
        metadata, ratings = load_ml100k_dataset(input_dir, spark)
    elif os.path.exists(f"{input_dir}/ratings.csv"):
        metadata, ratings = load_the_movies_dataset(input_dir, spark)
    else:
        print(f"Unable to locate ratings in dir: {input_dir}", file=sys.stderr)
        exit(1)
    metadata_dir = f"{output_dir}/metadata"
    ratings_dir = f"{output_dir}/ratings"

    write_output(metadata, output_format, metadata_dir)
    write_output(ratings, output_format, ratings_dir)

    shutil.move(glob(f"{ratings_dir}/*")[0], f"{output_dir}/ratings.{output_format}")
    shutil.move(
        glob(f"{metadata_dir}/*")[0], f"{output_dir}/movies_metadata.{output_format}"
    )
    shutil.rmtree(metadata_dir)
    shutil.rmtree(ratings_dir)


if __name__ == "__main__":
    main()
