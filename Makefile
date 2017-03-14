all: build

build:
	docker build -t project_name .
run:
	docker run --rm project_name project_name
test:
	docker run -it --rm project_name python -m unittest
test_server:
	docker run --rm project_name python -m unittest
quality:
	docker run --rm project_name flake8 project_name
	docker run --rm project_name pylint project_name

