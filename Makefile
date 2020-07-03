jupyter-container			= jupyter/tensorflow-notebook
phony-container				= bazhenov.me/phony
docker-run						= docker run --rm 
mount-private					= -v `pwd`/private:/wd -w /wd
mount-sources					= -v `pwd`:/wd -w /wd

epochs								= 10
heldout_examples			= 50000

# GPU_IDX is the number of active GPU in nvidia-smi (0 if single GPU is present in the system)
ifdef GPU_IDX
	gpu-flags = --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$(GPU_IDX)
endif

ifdef EPOCHS
	epochs = $(EPOCHS)
endif

define jq_eval_proc
	.sample as $$text | { \
		text: $$text, \
		label: .label | map($$text[.[0]:.[1]]) | select(. | length > 0), \
		prediction: .prediction | map($$text[.[0]:.[1]]) | select(. | length > 0) \
	} | select(.label != .prediction)
endef

clean:
	rm -f private/input.ndjson private/sample.txt

# Runs ipython instance in a tensorflow notebook for experiments
ipython:
	$(docker-run) -it $(jupyter-container) ipython

# Learns the model
learn: private/input.hdf5
	$(docker-run) -t $(mount-private) $(gpu-flags) $(phony-container) /learn.py -f input.hdf5 -o ./model -e $(epochs) \
		-v 0.05

# Builds a docker container with phony
docker:
	docker build -t $(phony-container) --build-arg TF_VERSION=1.13.1 learn

docker-gpu:
	docker build -t $(phony-container) --build-arg TF_VERSION=1.13.1-gpu learn

private/input.ndjson: private/sq.ndjson
	cat private/sq.ndjson \
	  | jq -c '{sample: .text, label: (.spans | map([.start, .end]))}' \
		| phone-augment -j -p 0.1 | grep -v '""' > $@

private/verify.ndjson: private/input.ndjson
	head -$(heldout_examples) private/input.ndjson > $@

private/learn.ndjson: private/input.ndjson
	tail +$(heldout_examples) private/input.ndjson > $@

private/input.hdf5: private/learn.ndjson
	cat private/learn.ndjson | phony export -o $@

private/eval.ndjson: private/verify.ndjson
	phony inference-file --model=$(MODEL) -i private/verify.ndjson -o private/eval.ndjson

eval: private/eval.ndjson
	phony eval private/eval.ndjson

eval-check: private/eval.ndjson
	cat private/eval.ndjson | jq '$(jq_eval_proc)' | less

.PHONY: build learn eval ipython eval
