ARG TF_VERSION
FROM tensorflow/tensorflow:$TF_VERSION-py3

RUN apt-get update && apt-get -y install git && apt-get clean
WORKDIR /root
RUN git clone https://www.github.com/keras-team/keras-contrib.git
WORKDIR /root/keras-contrib
RUN python convert_to_tf_keras.py
ENV USE_TF_KERAS 1
RUN python setup.py install
COPY phony.py /
COPY learn.py /
WORKDIR /
ENTRYPOINT [ "python" ]
