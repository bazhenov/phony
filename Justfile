jupyter-container := "jupyter/tensorflow-notebook"
phony-container		:= "bazhenov.me/phony"
docker						:= "docker run --rm -it -v `pwd`:/wd -w /wd"

test:
	{{docker}} {{jupyter-container}} python test.py

ipython:
	{{docker}} {{jupyter-container}} ipython

learn:
	{{docker}} {{phony-container}} -f private/input.jsonld -o model.h5

build:
	docker build . -t {{phony-container}}
