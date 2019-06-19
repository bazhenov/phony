FROM tensorflow/tensorflow:1.11.0-gpu-py3

COPY phony.py /
COPY learn.py /
WORKDIR /
ENTRYPOINT [ "python" ]
