ARG TF_VERSION
FROM tensorflow/tensorflow:$TF_VERSION-py3

RUN apt-get update && apt-get -y install git && apt-get clean
RUN pip install git+https://www.github.com/keras-team/keras-contrib.git
COPY phony.py /
COPY learn.py /
WORKDIR /
ENTRYPOINT [ "python" ]
