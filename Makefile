jupyter-container			= jupyter/tensorflow-notebook
phony-container				= bazhenov.me/phony
docker-run						= docker run --rm 
mount-private					= -v `pwd`/private:/wd -w /wd
mount-sources					= -v `pwd`:/wd -w /wd

epochs								= 10

# GPU_IDX is the number of active GPU in nvidia-smi (0 if single GPU is present in the system)
ifdef GPU_IDX
	gpu-flags = --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$(GPU_IDX)
endif

ifdef EPOCHS
	epochs = $(EPOCHS)
endif

# Test phone prepare augmentator
test:
	$(docker-run) $(mount-sources) $(jupyter-container) python test.py

# Runs phone prepare augmentator
private/input.jsonld:
	pv private/sample.txt | serve/target/release/augment -s -c 100 > private/input.jsonld

clean:
	rm -f private/input.jsonld

# Runs ipython instance in a tensorflow notebook for experiments
ipython:
	$(docker-run) -it $(jupyter-container) ipython

# Learns the model
learn: private/input.jsonld
	$(docker-run) -t $(mount-private) $(gpu-flags) $(phony-container) /learn.py -f input.jsonld -o ./model -e $(epochs) \
		-v 0.05

# Builds a docker container with phony
build:
	docker build -t $(phony-container) --build-arg TF_VERSION=1.13.1 learn

build-gpu:
	docker build -t $(phony-container) --build-arg TF_VERSION=1.13.1-gpu learn

.PHONY: build learn test ipython
