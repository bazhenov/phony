FROM jupyter/tensorflow-notebook

COPY phony.py /app/
COPY learn.py /app/
WORKDIR /app
ENTRYPOINT [ "python", "/app/learn.py" ]