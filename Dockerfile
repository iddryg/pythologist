# An Image development environment
#     docker build -t cio_image_lab:20181129 --build-arg user=USERNAME --build-arg group=GROUPNAME --build-arg user_id=USERID --build-arg group_id=GROUPID .
FROM ubuntu:20.04
RUN apt-get update \
    && apt-get upgrade -y \
    && DEBIAN_FRONTEND='noninteractive' apt-get install -y \
               python3-pip \
               python3-dev \
               nano \
               wget \
               git \
               r-base \
    && apt-get autoremove \
    && apt-get clean

RUN apt-get update \
 && DEBIAN_FRONTEND='noninteractive' \
 && apt-get install -y --no-install-recommends apt-utils \
                                               build-essential \
                                               sudo \
                                               git \
                                               libhdf5-serial-dev \
 && apt-get autoremove \
 && apt-get clean


RUN pip3 install --upgrade pip \
    &&  pip3 install --upgrade setuptools

RUN pip3 install pandas \
    && pip3 install numpy \
    && pip3 install scipy \
    && pip3 install h5py \
    && pip3 install scikit-learn \
    && pip3 install openpyxl \
    && pip3 install umap-learn \
    && pip3 install tables \
    && pip3 install imageio \
    && pip3 install xmltodict \
    && pip3 install scikit-image \
    && pip3 install imagecodecs \
    && pip3 install jsonschema \
    && pip3 install opencv-python-headless

ARG user=jupyter_user
ARG user_id=999
ARG group=jupyter_group
ARG group_id=999

RUN useradd -l -u $user_id -ms /bin/bash $user \
    && groupadd -g $group_id $group \
    && usermod -a -G $group $user

RUN pip3 install jupyterlab \
    && pip3 install matplotlib \
    && pip3 install plotnine[all] \
    && pip3 install seaborn 

RUN mkdir /source \
    && cd /source \
    && git clone https://github.com/jason-weirather/good-neighbors.git \
    && cd good-neighbors \
    && pip3 install -e .

ADD . /source/pythologist
RUN cd /source/pythologist \
    && pip3 install .

RUN mkdir .local \
    && chmod -R 777 .local
RUN mkdir .jupyter \
    && chmod -R 777 .jupyter
RUN mkdir /work
WORKDIR /work

CMD ["jupyter","lab","--ip=0.0.0.0","--port=8888","--allow-root"]
