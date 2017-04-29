export PROJECT=$(shell grep -o 'PROJECT.*' settings.py | sed -e 's/PROJECT="\(.*\)"/\1/')
export AUTHOR=$(shell grep -o 'AUTHOR.*' settings.py | sed -e 's/AUTHOR="\(.*\)"/\1/')
MAKEFLAGS="B"

folder:
	mv project $(PROJECT)

env:
	conda env create -f environment.yml -n $(PROJECT)

test:
	nosetests -v --with-doctest --doctest-tests \
		--with-coverage --cover-package=$(PROJECT)
	rm .coverage
smt:
	@read -r -p "Enter sumatra project: " PROJ; \
	read -r -p "Enter sumatra username: " USER; \
	read -r -p "Enter sumatra password: " PASS; \
	read -r -p "Enter archive path: " ARCH; \
        read -r -p "Enter output path: " OUT;\
	echo "\nAdditionally, using the following sumatra settings:"; \
	echo "--executable python"; \
	echo "--on-changed diff"; \
	echo "--store http://$$USER:***@192.33.91.83:8080/records"; \
	smt init \
	--datapath $$ARCH \
	--archive $$ARCH \
	--executable python \
	--on-changed store-diff \
	--store http://$$USER:$$PASS@192.33.91.83:8080/records \
	$$PROJ
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
data:
	wget -O Data.tar.gz "https://www.dropbox.com/s/juacxvix0rdqo09/Data.tar.gz?dl=0" 
	tar -xzvf Data.tar.gz -C ./
	rm Data.tar.gz
	echo "MRI Data downloaded successfully to 'Data' folder"
