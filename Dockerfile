FROM ubuntu:18.04
LABEL org.opencontainer.image.authors="Yatharth Kapadia" \
      org.opencontainer.image.source="https://github.com/yatharthk2/mnist_pytorch" \
      org.opencontainer.image.description="A simple mnist Neural Network based project with all the functionalities that of a real repo " 
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y htop python3-dev wget && rm -rf /var/lib/apt/lists/*
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda create -y -n pytorch python=3.7
COPY . src/
RUN /bin/bash -c "cd src \
    &&source activate pytorch \
    && pip install -r requirements.txt"