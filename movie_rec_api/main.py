from typing import Annotated, Dict
from uuid import uuid4

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jinja2 import Template
from pydantic import BaseModel
from pyspark import Row
from pyspark.sql.functions import col

from .recommender import MovieRecommender, Rating


class SessionData(BaseModel):
    movie_ids: list[int]
    ratings: Dict[int, float]


sessions: Dict[str, SessionData] = {}

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="templates")


movie_recommender = MovieRecommender(dataset_dir="out")


def get_recommender():
    return movie_recommender


movies = [{"title": "Movie 1", "year": 1997, "id": 1}]


@app.get("/")
async def root(
        request: Request,
        recommender: Annotated[MovieRecommender, Depends(get_recommender)]):
    metadata = recommender.movie_metadata
    movie_sample = metadata\
        .sample(fraction=30/metadata.count())\
        .limit(20)\
        .rdd.map(lambda r: r.asDict())\
        .collect()
    return templates.TemplateResponse("movies.html", {
        "request": request,
        "movies": movie_sample
    })


@app.get("/loadMovies/")
async def load_movies(
        request: Request,
        recommender: Annotated[MovieRecommender, Depends(get_recommender)],
):
    session_id = get_or_create_session(request.cookies.get("movieIds"))
    session_data = sessions[session_id]

    metadata = recommender.movie_metadata\
        .filter(~(col("id").isin(session_data.movie_ids)))
    movie_sample = metadata\
        .sample(fraction=30/metadata.count())\
        .limit(20)\
        .rdd.map(lambda r: r.asDict())\
        .collect()
    session_data.movie_ids.extend([movie["id"] for movie in movie_sample])
    rendered = Template("""\
{% for movie in movies %}
<div class="movie">
    <span class="movie-title">{{ movie.title }} ({{ movie.year }})</span>\
    {% for rating in range(1, 6) %}
    <input type="radio" name="m_{{ movie.id }}" value="{{ rating }}">{{ rating }}
    {% endfor %}
</div>
{% endfor %}\
<button hx-post="/submitRatings/" hx-target="#submittedRatings">Submit ratings</button>
""").render(movies=movie_sample)
    response = HTMLResponse(content=rendered)
    response.set_cookie("movieIds", str(session_id), samesite="strict")
    return response


def get_or_create_session(session_id: str | None) -> (str, list[int]):
    if session_id is None or session_id not in sessions:
        session_id = str(uuid4())
        session_data = SessionData(movie_ids=[], ratings={})
        sessions[session_id] = session_data
    return session_id


@app.post("/submitRatings/")
async def submit(request: Request):
    session_id = request.cookies.get("movieIds")
    if session_id is None or session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session")
    session_data = sessions[session_id]
    form_data = await request.form()
    ratings: dict[int, float] = {}
    for m_id, rating in form_data.items():
        if not m_id.startswith("m_") or not isinstance(rating, str):
            continue
        ratings[int(m_id[2:])] = float(rating)
    session_data.ratings.update(ratings)
    ratings_df = movie_recommender.spark.createDataFrame(
        [Row(id=k, rating=v) for k, v in session_data.ratings.items()])
    print(session_data)
    ratings_with_metadata = movie_recommender.movie_metadata\
        .join(ratings_df, "id", "inner")\
        .collect()
    print(ratings_with_metadata)
    rendered = Template("""\
{% for movie in movies %}
<span class="movie-title">{{ movie.title }} ({{ movie.year }}): {{ movie.rating }}</span>
{% endfor %}\
""").render(movies=ratings_with_metadata)
    response = HTMLResponse(content=rendered)
    response.headers.append("HX-Trigger", "ratingsSubmitted")
    return response


@app.post("/recommend/")
async def recommend(
        request: Request,
        recommender: Annotated[MovieRecommender, Depends(get_recommender)]):
    session_id = request.cookies.get("movieIds")
    if session_id is None or session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session")
    ratings = sessions[session_id].ratings
    if len(ratings) == 0:
        raise HTTPException(
            status_code=400, detail="User has no submitted ratings")
    recommendations = recommender.recommend(ratings, 20)
    rendered = Template("""\
<h2>Movie Recommendations</h2>
{% for r in recommendations %}\
<span class="movie-title">{{ r.title }} ({{ r.year }}): {{ r.prediction }}</span>
{% endfor %}\
""").render(recommendations=recommendations)
    return HTMLResponse(content=rendered)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
