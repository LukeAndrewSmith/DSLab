FROM ubuntu:latest
# set up the dir and get all the files in there:
#WORKDIR /app
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    git
RUN apt-get install ffmpeg libsm6 libxext6  -y

#mgdal errors
RUN apt-get install -y software-properties-common && apt-get update
#RUN  add-apt-repository ppa:ubuntugis/ppa &&  apt-get update
RUN apt-get install -y gdal-bin libgdal-dev
ARG CPLUS_INCLUDE_PATH=/usr/include/gdal
ARG C_INCLUDE_PATH=/usr/include/gdal
#RUN add-apt-repository ppa:ubuntugis/ppa && apt-get update
#RUN apt-get install python-numpy gdal-bin libgdal-dev


RUN mkdir /dslabtreering
WORKDIR /dslabtreering/src/ringdetector/ringdetector
# clone and get all the necessary files
COPY src /dslabtreering/src
COPY models /dslabtreering/models
#RUN add-apt-repository ppa:ubuntugis/ppa && apt-get update && apt-get install -y gdal-bin python-gdal python3-gdal
# install the dependencies
#RUN cd src
RUN pip install -e /dslabtreering/src/ringdetector/
RUN pip install numpy --upgrade
RUN apt-get install -y rasterio
# install torch: no gpu version:
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

# install detectron2:
RUN python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# please make sure that you set the Paths in Paths.py to the correct ones before building the image




