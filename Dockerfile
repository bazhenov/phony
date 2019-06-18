FROM jupyter/tensorflow-notebook

COPY phony.py /
COPY learn.py /
WORKDIR /
ENTRYPOINT [ "python" ]