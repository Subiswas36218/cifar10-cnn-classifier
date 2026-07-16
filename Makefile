install:
	pip install -e .
	pip install -r requirements.txt

format:
	black .

lint:
	ruff check .

typecheck:
	mypy src

test:
	pytest -v

check: format lint typecheck test

train:
	python -m src.train

evaluate:
	python -m src.evaluate

predict:
	python -m src.predict

api:
	uvicorn src.api:app --reload

streamlit:
	streamlit run src/app.py