all: build

build:
	docker build -t PROJECT .
run:
	docker run --rm PROJECT PROJECT
test:
	docker run -it --rm PROJECT python -m unittest
test_server:
	docker run --rm PROJECT python -m unittest
quality:
	docker run --rm PROJECT flake8 PROJECT
	docker run --rm PROJECT pylint PROJECT

