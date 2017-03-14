all: build

build:
	docker build -t subtyping .
run:
	docker run --rm subtyping subtyping
test:
	docker run -it --rm subtyping python -m unittest
test_server:
	docker run --rm subtyping python -m unittest
quality:
	docker run --rm subtyping flake8 subtyping
	docker run --rm subtyping pylint subtyping

