# Creates a container from the x86 image. 
xhost +
docker run -ti --runtime=nvidia --ipc=host -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all -p 8888:8888 -p 8889:8889 -v /data:/data -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix  temporal-shift-module bash
