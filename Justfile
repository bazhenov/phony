jupyter-container := "jupyter/tensorflow-notebook"
phony-container		:= "bazhenov.me/phony"
docker						:= "docker run --rm -v `pwd`/private:/wd -w /wd"

# Test phone prepare augmentator
test:
	{{docker}} {{jupyter-container}} python test.py

# Runs phone prepare augmentator
prepare:
	pv private/sample.txt | {{docker}} -i {{phony-container}} /phony.py > private/input.jsonld

# Runs ipython instance in a tensorflow notebook for experiments
ipython:
	{{docker}} -it {{jupyter-container}} ipython

# Learns the model
learn:
	{{docker}} -t {{phony-container}} /learn.py -f input.jsonld -o model.h5 -e 10

# Builds a docker container with phony
build:
	docker build . -t {{phony-container}}
