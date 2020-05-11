install-dev:
	virtualenv .venv -p python3 && . .venv/bin/activate && pip install -r requirements-dev.txt && pre-commit install

install:
	virtualenv .venv -p python3 && . .venv/bin/activate && pip install -r requirements.txt

lint:
	pylint tf2_yolov4 && black tf2_yolov4 --check

test:
	pytest -vv --cov-report term-missing --no-cov-on-fail --cov=tf2_yolov4/ .
