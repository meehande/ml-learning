import numpy as np
import pathlib
import tensorflow as tf
from tensorflow import keras
import requests
import tempfile
import typer
from zipfile import ZipFile
app = typer.Typer()


def ensure_dir(folder: str) -> pathlib.Path:
    dir = pathlib.Path(folder).resolve()
    if not dir.exists():
        dir.mkdir(parents=True, exist_ok=True)
    return dir


@app.command("download-data")
def download_movielens_dataset(output_folder: str = "data/movielens"):
    url = "http://files.grouplens.org/datasets/movielens/ml-latest.zip"

    output_folder = ensure_dir(output_folder)

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size_in_bytes= int(r.headers.get('content-length', 0))
        with tempfile.NamedTemporaryFile(mode='rb+') as temp_f:
            downloaded = 0
            dl_iteration = 0
            chunk_size = 8192
            total_chunks = total_size_in_bytes / chunk_size if total_size_in_bytes else 100
            for chunk in r.iter_content(chunk_size=chunk_size):
                downloaded += chunk_size
                dl_iteration += 1
                percent = (100 * dl_iteration * 1.0/total_chunks)
                if dl_iteration % 10 == 0 and percent < 100:
                    print(f'Completed {percent:2f}%')
                elif percent >= 99.9:
                    print(f'Download completed. Now unzipping...')
                temp_f.write(chunk)
            with ZipFile(temp_f, 'r') as zipf:
                zipf.extractall(output_folder)
                print(f"\n\nUnzipped.\n\nFiles downloaded and unziped to:\n\n{output_folder}")

@app.command("train")
def train():
    pass

if __name__ == "__main__":
    app()