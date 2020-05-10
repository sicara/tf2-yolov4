install-dev:
	virtualenv .venv -p python3 && . .venv/bin/activate && pip install -r requirements-dev.txt


install:
	virtualenv .venv -p python3 && . .venv/bin/activate && pip install -r requirements.txt

test:
	. .venv/bin/activate && pytest -vv --cov-report term-missing --no-cov-on-fail --cov=tf2_yolov4/ .
