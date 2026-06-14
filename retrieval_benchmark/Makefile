.PHONY: run install build evaluate clean

run: install build evaluate

install:
	pip install -r requirements.txt --break-system-packages

build:
	python build_corpus.py

evaluate:
	python evaluate.py

clean:
	rm -f data/corpus.jsonl results/results.json
