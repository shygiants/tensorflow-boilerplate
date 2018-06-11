FROM tensorflow/tensorflow:1.8.0-gpu AS base

MAINTAINER Sanghoon Yoon <shygiants@gmail.com>

# Install libraries
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get -y install libsm6 libice6 libxrender1 libxext6 libstdc++6

# Tensorflow project
FROM tensorflow/tensorflow:1.7.0-gpu

# Copy libraries
COPY --from=base /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu
COPY --from=base /usr/bin /usr/bin

# Set working directory
WORKDIR "/tf"

RUN mkdir /dataset && mkdir /job-dir

# Copy all source codes
COPY . .

RUN pip install -r /tf/src/requirements.txt

ENTRYPOINT ["bash", "run.sh"]