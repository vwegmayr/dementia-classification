export PROJECT=$(shell grep -o 'PROJECT.*' settings.py | sed -e 's/PROJECT="\(.*\)"/\1/')
export AUTHOR=$(shell grep -o 'AUTHOR.*' settings.py | sed -e 's/AUTHOR="\(.*\)"/\1/')
MAKEFLAGS="B"

all:
	mv project $(PROJECT)
	conda env create -f environment.yml -n $(PROJECT)

test:
	nosetests --with-doctest --with-coverage --cover-package=$(PROJECT) -v
	rm .coverage

quality:
	flake8 $(PROJECT)
	pylint $(PROJECT)

doc:
	cp doc/example_templates/* doc/source/examples
	sed -i 's/PROJECT/$(PROJECT)/g' doc/source/examples/*.rst && \
	$(MAKE) -C doc html

view:
	see ./doc/build/html/index.html

export:
	conda-env export -f environment_export.yml

remove:
	conda remove -n $(PROJECT) --all

build_server:
	docker build -t $(PROJECT) .

run_server:
	docker run --rm $(PROJECT) $(PROJECT)

test_server:
	docker run --rm $(PROJECT) python -m unittest

quality_server:
	docker run --rm $(PROJECT) flake8 $(PROJECT)
	docker run --rm $(PROJECT) pylint $(PROJECT)
