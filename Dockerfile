FROM python:3.5

# install needed dependencies for code quality
RUN pip install flake8 pylint numpy

# import code from host to container
COPY . /code
WORKDIR /code

# install package
RUN pip install .

