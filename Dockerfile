FROM w251/pytorch:dev-tx2-4.3_b132

RUN apt-get update

RUN apt-get install -y git-all vim iputils-ping
RUN apt-get install -y python3-sklearn python3-sklearn-lib 

RUN pip3 install  numpy==1.17.0
RUN pip3 uninstall -y  Pillow
RUN pip3 install Pillow==6.1.0
RUN pip3 install urllib3
RUN pip3 install twilio
RUN pip3 install matplotlib

RUN git clone https://github.com/mit-han-lab/temporal-shift-module.git

WORKDIR /temporal-shift-module

RUN mkdir dev_files
RUN mkdir pretrained

COPY pretrained /temporal-shift-module/pretrained

RUN apt-get install -y libopencv-dev python3-opencv

