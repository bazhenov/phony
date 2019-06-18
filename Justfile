container := "jupyter/tensorflow-notebook"
docker		:= "docker run --rm -it -v `pwd`:/wd -w /wd " + container

test:
	{{docker}} python test.py

ipython:
	{{docker}} ipython

build:
	docker build . -t bazhenov.me/phony
