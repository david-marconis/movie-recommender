from recommender import MovieRecommender, Rating

r = MovieRecommender("ml-100k")
s = r.recommend(
    [
        Rating(movieId=50, rating=5.0),
        Rating(movieId=172, rating=5.0),
        Rating(movieId=133, rating=1.0),
    ],
    40,
)
[print(x) for x in s]
