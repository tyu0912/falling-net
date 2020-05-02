# Falling-Net: A Fall Detector on the Edge
Tennison Yu, Stephanie Mather, Apik Zorian, and Kevin Hanna

Falls can be a significant source of injury for many people in many different circumstances. For example, a worker tripping in a manufacturing plant or a senior citizen losing their balance at a retirement home. Many a times, these types of injuries can lead to expensive medical procedures either for the injured, the company they work for, or any other responsible body. Prevention is therefore best. Here, we're developing a system to detect falls by leveraging the temporal shift model developed by https://github.com/mit-han-lab/temporal-shift-module.

Read our medium post here: 

## Steps to run:

**Please feel free to try it out. Note though that the authors are not responsible for any injuries that may occur. **

1. Clone the repo wherever you'd like. 

2. Build the image: `docker build -t <image-name> .`

3. Run the container: 

`sudo docker run --privileged -v $(pwd)/dev_files:/temporal-shift-module/dev_files -v /dev/bus/usb:/dev/bus/usb -v /tmp:/tmp -e QT_X11_NO_MITSHM=1 -e DISPLAY=$DISPLAY -it <image-name> bash`
