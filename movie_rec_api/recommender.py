import sys
from typing import List

from pydantic import BaseModel
from pyspark.ml.recommendation import ALS
from pyspark.sql import DataFrame, Row, SparkSession
from pyspark.sql.functions import lit


class Rating(BaseModel):
    movieId: int
    rating: float


class Recommendation(BaseModel):
    title: str
    prediction: float


class MovieRecommender():
    def __init__(self, dataset_dir: str):
        self.spark = SparkSession.builder\
            .config(
                "spark.driver.extraJavaOptions",
                "-Dlog4j.configuration=file:log4j.properties")\
            .config(
                "spark.executor.extraJavaOptions",
                "-Dlog4j.configuration=file:log4j.properties")\
            .config("spark.ui.showConsoleProgress", "false")\
            .getOrCreate()
        self.spark.sparkContext.setLogLevel("ERROR")
        self.movie_metadata = self.spark.read.parquet(
            f"{dataset_dir}/movies_metadata.parquet").cache()
        self.ratings = self.spark.read\
            .parquet(f"{dataset_dir}/ratings.parquet")
        self.als = ALS(
            maxIter=5,
            regParam=0.01,
            userCol="userId",
            itemCol="movieId",
            ratingCol="rating")

    def recommend(self, new_ratings: List[Rating], top_k: int) -> List[Recommendation]:
        ratings = [Row(userId=0, movieId=r.movieId, rating=r.rating)
                   for r in new_ratings]
        ratings_df = self.spark.createDataFrame(ratings)
        print("Ratings:")
        ratings_df.show()
        model = self.als.fit(self.ratings.union(ratings_df))
        movies_to_predict = self.movie_metadata\
            .select(["id", "title", "year"])\
            .withColumnRenamed("id", "movieId")\
            .join(ratings_df, "movieId", "left")\
            .withColumn("userId", lit(0))\
            .orderBy("movieId")
        print("To be predicted:")
        movies_to_predict.show()
        recommendations = model.transform(movies_to_predict)
        top_recommendations = recommendations\
            .sort(recommendations.prediction.desc())
        print("Predictions:")
        top_recommendations.show()
        return [
            {
                "id": r.movieId,
                "title": r.title,
                "year": r.year,
                "prediction": r.prediction,
            }
            for r in top_recommendations.take(top_k)
        ]

    def stop(self):
        self.spark.stop()


def get_user_ratings(movie_metadata: DataFrame):
    new_ratings = []
    to_rate = movie_metadata.sample(withReplacement=False, fraction=0.1)
    print("Enter your rating of the following movies on a scale from 1-5. "
          "Enter no rating to stop or rating 0 to skip the movie")
    for row in to_rate.toLocalIterator():
        movie_id = row["id"]
        entered = input(f"How would you rate '{row.title} ({row.year})': ")
        if not entered:
            break
        try:
            rating = float(entered)
            if rating < 1 or rating > 5:
                if rating != 0:
                    print(f"Ignoring invalid rating: {rating}")
                continue
            new_ratings.append(Rating(movieId=movie_id, rating=rating))
        except ValueError:
            print(f"Invalid input: '{entered}'")
    return new_ratings


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path-to-dataset>")
        sys.exit(1)
    dataset_dir = sys.argv[1]
    recommender = MovieRecommender(dataset_dir)
    new_ratings = get_user_ratings(recommender.movie_metadata)
    if len(new_ratings) == 0:
        print("You didn't enter any ratings :(")
        sys.exit(1)

    print("\nRecommending 20 new movies to watch...")
    recommendations = recommender.recommend(new_ratings, 20)

    for recommendation in recommendations:
        print(recommendation.title, recommendation.prediction)
    recommender.stop()


if __name__ == "__main__":
    main()
