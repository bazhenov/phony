test:
	docker run --rm -it -v `pwd`:/wd -w /wd jupyter/tensorflow-notebook python test.py

build:
	docker build . -t bazhenov.me/phony
