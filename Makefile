.PHONY: all install reproduce test clean

all: install reproduce test

install:
	pip install -r requirements.txt

reproduce:
	python reproduce.py --all

test:
	pytest -q

clean:
	rm -rf results/*.png __pycache__ .pytest_cache
