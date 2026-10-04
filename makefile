.PHONY: install install-api prepare train eval api test docker

install:
	pip install -r requirements.txt

install-api:
	pip install -r requirements-api.txt

prepare:
	python scripts/prepare_dataset.py

train:
	python -m src.train --config configs/config.yaml

eval:
	python -m src.evaluate --config configs/config.yaml

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest -q

docker:
	docker compose up --build