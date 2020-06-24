from pathlib import Path

import pytest

from tf2_yolov4.tools import weights


@pytest.fixture
def tf2_yolov4_default_directory(mocker):
    mock_weights_dir = Path(__file__).parent / ".tf2-yolov4/"
    mocker.patch.object(weights, "TF2_YOLOV4_DEFAULT_PATH", mock_weights_dir)

    yield mock_weights_dir

    if mock_weights_dir.is_dir():
        mock_weights_dir.rmdir()


def test_should_create_weights_dir_if_not_exist(tf2_yolov4_default_directory):
    assert not tf2_yolov4_default_directory.is_dir()

    weights.is_darknet_weights_available()

    assert tf2_yolov4_default_directory.is_dir()


def test_should_download_original_weights_from_google_drive_if_not_available(
    tf2_yolov4_default_directory, mocker
):
    darknet_original_weights_path = tf2_yolov4_default_directory / "yolov4.weights"
    mocker.patch.object(
        weights, "DARKNET_ORIGINAL_WEIGHTS_PATH", darknet_original_weights_path
    )
    mock_download_file_from_google_drive = mocker.patch.object(
        weights, "download_file_from_google_drive"
    )
    mock_load_darknet_weights_in_yolo = mocker.patch.object(
        weights, "load_darknet_weights_in_yolo"
    )
    mock_yolo_model = mocker.MagicMock(
        save_weights=mocker.MagicMock(), load_weights=mocker.MagicMock(),
    )

    weights.download_darknet_weights(mock_yolo_model)

    mock_download_file_from_google_drive.assert_called_once_with(
        weights.YOLOV4_DARKNET_FILE_ID,
        darknet_original_weights_path,
        target_size=weights.YOLOV4_DARKNET_FILE_SIZE,
    )
    mock_load_darknet_weights_in_yolo.assert_called_once_with(
        mock_yolo_model, str(darknet_original_weights_path),
    )
