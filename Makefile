jupyter-container			= jupyter/tensorflow-notebook
phony-container				= bazhenov.me/phony
docker-run						= docker run --rm 
mount-private					= -v `pwd`/private:/wd -w /wd
mount-sources					= -v `pwd`:/wd -w /wd

# GPU_IDX is the number of active GPU in nvidia-smi (0 if single GPU is present in the system)
ifdef GPU_IDX
	gpu-flags = --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$(GPU_IDX)
endif

# Test phone prepare augmentator
test:
	$(docker-run) $(mount-sources) $(jupyter-container) python test.py

# Runs phone prepare augmentator
private/input.jsonld:
	pv private/sample.txt | $(docker-run) -i $(phony-container) /phony.py > private/input.jsonld

# Runs ipython instance in a tensorflow notebook for experiments
ipython:
	$(docker-run) -it $(jupyter-container) ipython

# Learns the model
learn: private/input.jsonld
	$(docker-run) -t $(mount-private) $(gpu-flags) $(phony-container) /learn.py -f input.jsonld -o model.h5 -e 10

# Builds a docker container with phony
build:
	docker build . -t $(phony-container)

.PHONY: build learn test ipython