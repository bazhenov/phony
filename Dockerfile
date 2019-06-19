FROM tensorflow/tensorflow:latest-gpu-py3

COPY phony.py /
COPY learn.py /
WORKDIR /
ENTRYPOINT [ "python" ]