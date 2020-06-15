"""
Download util from Google Drive.

Taken from: https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
"""
import requests

import tensorflow as tf


def download_file_from_google_drive(id, destination, target_size=None):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination, target_size)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination, target_size=None):
    CHUNK_SIZE = 32768

    progbar = None
    if target_size:
        progbar = tf.keras.utils.Progbar(target_size)

    with open(destination, "wb") as f:
        for chunk_index, chunk in enumerate(response.iter_content(CHUNK_SIZE)):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                if progbar:
                    progbar.update(chunk_index * CHUNK_SIZE)
