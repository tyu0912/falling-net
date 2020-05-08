FROM python:3.6-buster

RUN pip install torch==1.2.0 torchvision==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html
#RUN pip install torch==1.0.0 torchvision==0.2.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install tensorboardX numpy scikit-learn matplotlib pandas tqdm Pillow==6.1
RUN pip install jupyterlab

RUN apt-get update
RUN apt-get install -y ffmpeg vim rename

RUN pip install opencv-python tabulate

WORKDIR /workdir
RUN git clone https://github.com/kevinhanna/temporal-shift-module 

WORKDIR /workdir/temporal-shift-module
