FROM nvidia/cuda:11.3.1-base-ubuntu20.04
RUN apt update
RUN apt-get install -y python3 python3-pip

RUN pip install torch==1.4.0 torchvision==0.5.0

WORKDIR /DRIVE

COPY . /DRIVE
# RUN pip install -r requirements.txt

RUN echo "test"
