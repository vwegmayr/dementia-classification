all: build

build:
	docker build -t tracking .
run:
	docker run --rm tracking fiber-tracking
test:
	docker run -it --rm tracking python -m unittest
test_server:
	docker run --rm tracking python -m unittest
quality:
	docker run --rm tracking flake8 globaltracking
	docker run --rm tracking pylint globaltracking

