import shutil
import sys
from enum import Enum, auto
from glob import glob

from pyspark.sql import SparkSession
from pyspark.sql.functions import col


class OutputFormat(Enum):
    CSV = auto()
    PARQUET = auto()

    def __str__(self):
        return self.name.lower()


def write_output(data_frame, out_format, output_dir):
    writer = data_frame.write\
        .mode("overwrite")\
        .option("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
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

    spark = SparkSession.builder\
        .config(
            "spark.driver.extraJavaOptions",
            "-Dlog4j.configuration=file:log4j.properties")\
        .config(
            "spark.executor.extraJavaOptions",
            "-Dlog4j.configuration=file:log4j.properties")\
        .config("spark.ui.showConsoleProgress", "false")\
        .getOrCreate()

    metadata = spark.read.csv(
        f"{input_dir}/movies_metadata.csv", header=True, inferSchema=True)
    ratings = spark.read.csv(
        f"{input_dir}/ratings.csv", header=True, inferSchema=True)
    popular_movies = ratings\
        .groupBy("movieId")\
        .count()\
        .filter("count > 10")\
        .withColumnRenamed("movieId", "id")\
        .select("id")
    cleansed_metadata = metadata\
        .withColumn("year", col("release_date").substr(1, 4).cast("int"))\
        .withColumn("vote_count", col("vote_count").cast("int"))\
        .withColumn("id", col("id").cast("long"))\
        .select(["id", "year", "title"])\
        .filter(col("year").isNotNull())\
        .filter("adult = false AND vote_count > 10")\
        .join(popular_movies, "id", "inner")\
        .coalesce(1)
    cleansed_ratings = cleansed_metadata\
        .select("id")\
        .join(ratings, cleansed_metadata.id == ratings.movieId, "inner")\
        .select(["userId", "movieId", "rating"])\
        .filter("rating >= 1 AND rating <= 5")\
        .coalesce(1)

    metadata_dir = f"{output_dir}/metadata"
    ratings_dir = f"{output_dir}/ratings"

    write_output(cleansed_metadata, output_format, metadata_dir)
    write_output(cleansed_ratings, output_format, ratings_dir)

    shutil.move(glob(f"{ratings_dir}/*")[0],
                f"{output_dir}/ratings.{output_format}")
    shutil.move(glob(f"{metadata_dir}/*")[0],
                f"{output_dir}/movies_metadata.{output_format}")
    shutil.rmtree(metadata_dir)
    shutil.rmtree(ratings_dir)


if __name__ == "__main__":
    main()
