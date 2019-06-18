FROM jupyter/tensorflow-notebook

COPY phony.py /app/
ENTRYPOINT [ "python", "/app/phony.py" ]