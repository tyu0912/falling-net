## To run the model

docker built -t falling_net .

docker run --privileged -v $(pwd):/falling -v /dev/bus/usb:/dev/bus/usb -v /tmp:/tmp -e QT_X11_NO_MITSHM=1 -e DISPLAY=$DISPLAY -it falling_net bash
