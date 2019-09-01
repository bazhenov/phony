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

clean:
	rm -f private/input.ndjson private/sample.txt

# Runs ipython instance in a tensorflow notebook for experiments
ipython:
	$(docker-run) -it $(jupyter-container) ipython

# Learns the model
learn: private/input.hdf5
	$(docker-run) -t $(mount-private) $(gpu-flags) $(phony-container) /learn.py -f input.hdf5 -o ./model -e $(epochs) \
		-v 0.05

private/inference.ndjson: private/input.ndjson
	serve inference-file -i private/input.ndjson -o $@ --model $(MODEL)

# Builds a docker container with phony
docker:
	docker build -t $(phony-container) --build-arg TF_VERSION=1.13.1 learn

docker-gpu:
	docker build -t $(phony-container) --build-arg TF_VERSION=1.13.1-gpu learn

private/input.ndjson: private/sq.ndjson
	cat private/sq.ndjson | jq -c '{sample: .text, label: (.spans | map([.start, .end]))}' > $@

private/input.hdf5: private/input.ndjson
	cat private/input.ndjson | head -100000 | serve export -o $@

.PHONY: build learn eval ipython
