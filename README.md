# Movie recommender

## How to run
1. Download movies dataset from here: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset/
2. Setup, install and activate virtual environment
    ```sh 
    python3 -m venv .venv
    pip install -r requirements.txt
    source .venv/bin/activate # Linux only, Windows is different
    ```
3. Run the data cleanse script
    ```sh
    python movie_rec_api/data-cleanse.py /path/to/the-movies-dataset out
    ```
4. Start the server
    ```sh
    uvicorn movie_rec_api.main:app
    ```
