.PHONY: all clean
all: clean
	pip install -r requirements.txt
	python3 main.py

clean:
	rm -rf ./logs
	rm -rf ./output/
	rm -rf __pycache__/