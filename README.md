# Movie recommender
## Built using
[![HTMX](https://img.shields.io/badge/HTMX-36C?logo=htmx&logoColor=fff)](#)
[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](#)
## How to run
1. Download the [movies dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset/) or the [ml-100k dataset](https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset) and extract it. The movies dataset is a bit bigger and slower, but contains more movies and ratings and newer movies.
2. Setup, install and activate virtual environment
    ```sh 
    python3 -m venv .venv
    pip install -r requirements.txt
    source .venv/bin/activate # Linux only, Windows is different
    ```
3. Run the data cleanse script
    ```sh
    python movie_rec_api/data-cleanse.py /path/to/the-extracted-dataset out
    ```
4. Start the server
    ```sh
    uvicorn movie_rec_api.main:app
    ```
