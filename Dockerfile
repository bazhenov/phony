ARG TF_VERSION
FROM tensorflow/tensorflow:$TF_VERSION-py3

COPY phony.py /
COPY learn.py /
WORKDIR /
ENTRYPOINT [ "python" ]
