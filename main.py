import sys

sys.path.insert(1, './models')
from mobilenet_v2_tsm_test import MobileNetV2
#from arch_mobilenetv2 import MobileNetV2

from PIL import Image
import urllib.request
import os
import torch
import torchvision
import numpy as np
import cv2
import time
import torch.nn as nn
import argparse

from matplotlib import pyplot as plt
from twilio.rest import Client



class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Scale(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]



class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]



class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.concatenate(img_group, axis=2)



class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()



class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
        rep_std = self.std * (tensor.size()[0] // len(self.std))

        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor


def process_output(idx_, history, num_classes):
    # idx_: the output of current frame
    # history: a list containing the history of predictions
    if not REFINE_OUTPUT:
        return idx_, history

    max_hist_len = int((20/27)*num_classes) # max history buffer

    
    if num_classes == 27:
        # mask out illegal action
        if idx_ in [7, 8, 21, 22, 1, 3]:
            idx_ = history[-1]

        # use only one no action class
        if idx_ == 0:
            idx_ = 2

    elif num_classes == 3: 
        if idx_ in [2]:
            idx_ = history[-1]
        
        if idx_ == 0:
            idx_ = 0
    
    # history smoothing

    if idx_ != history[-1] and len(history) != 1:
        if not (history[-1] == history[-2]): #  and history[-2] == history[-3]):
            idx_ = history[-1]
    

    history.append(idx_)
    history = history[-max_hist_len:]

    return history[-1], history

def main(num_classes):

    print("Initializing model...")


    # print settings
    print("Model = MobileNet")
    print("SOFTMAX_THRESHOLD = " + str(SOFTMAX_THRES))
    print("HISTORY_LOGIT = " + str(HISTORY_LOGIT))
    print("CAMERA_FEED = " + str(CAMERA_FEED))
    print("TWILIO_ALERTS = " + str(SEND_ALERTS))


    # Print params for alert
    font                   = cv2.FONT_HERSHEY_COMPLEX
    bottomLeftCornerOfText = (450,950)
    fontScale              = 4
    fontColor              = (0,0,250)
    lineType               = 8

    # Print params for softmaxes
    font2                   = cv2.FONT_HERSHEY_SIMPLEX 
    topRightCornerOfText = (10,50)
    topRightCornerOfText2 = (10,100)
    fontScale2              = 2
    fontColor2 = (250,0,0)
    lineType2 = 2


    if CAMERA_FEED:
        topRightCornerOfText = (5,50)
        topRightCornerOfText2 = (5,70)
        fontScale2              = 0.5
        bottomLeftCornerOfText = (10,150)
        fontScale = 1


    categories = ['Test', "Fall", "Not Fall"]

    cropping = torchvision.transforms.Compose([
        GroupScale(256),
        GroupCenterCrop(224),
    ])


    transform = torchvision.transforms.Compose([
        cropping,
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    torch_module = MobileNetV2(n_class=num_classes)
    #print(torch_module.state_dict().keys())
    model_new = torch.load("./models/weights/ckpt.best.pth.tar")

    # Fixing new model parameter mis-match
    state_dict = model_new['state_dict']
    #print(state_dict.keys())

    from collections import OrderedDict
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        #name = k[7:] # remove `module.`

        if "module.base_model." in k:
            name = k.replace("module.base_model.", "")

            if ".net" in name:
                name = name.replace(".net", "")


        elif "module." in k:
            name = k.replace("module.new_fc.", "classifier.")
    

        new_state_dict[name] = v

    # load params
    torch_module.load_state_dict(new_state_dict)

    # Use GPU if CUDA found
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch_module.to(device)

    # Set system in parallel mode
    torch_module = nn.DataParallel(torch_module)
    torch_module.eval()

    cap = None

    if CAMERA_FEED:
        cap = cv2.VideoCapture(1)
        print("CAMERA")
    else:
        cap = cv2.VideoCapture(VIDEO_PATH) 


    # set a lower resolution for speed up
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    
    full_screen = False
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 480, 480)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, WINDOW_NAME)


    shift_buffer = [torch.zeros([1, 3, 56, 56]),
                    torch.zeros([1, 4, 28, 28]),
                    torch.zeros([1, 4, 28, 28]),
                    torch.zeros([1, 8, 14, 14]),
                    torch.zeros([1, 8, 14, 14]),
                    torch.zeros([1, 8, 14, 14]),
                    torch.zeros([1, 12, 14, 14]),
                    torch.zeros([1, 12, 14, 14]),
                    torch.zeros([1, 20, 7, 7]),
                    torch.zeros([1, 20, 7, 7])]


    t = None    
    index = 0
    idx = 2 # initialize to NotFall
    history = [0]
    history_logit = []
    history_timing = []
    i_frame = -1
    history_for_alerts = []
    frame_counter = {c:0 for c in categories}

    fall_frame_count = 0
    running_preds = []
    idx_ = 2 # initialize to NotFall
    state = "normal"
    softmax = []
    while True:
        
        i_frame += 1
        _, img = cap.read()  # (480, 640, 3) 0 ~ 255

        t1 = time.time()
        img_tran = transform([Image.fromarray(img).convert('RGB')])
        input_var = torch.autograd.Variable(img_tran.view(1, 3, img_tran.size(1), img_tran.size(2)))

        # Send tensor to GPU
        input_var = input_var.to(device)
        #shift_buffer = shift_buffer.to(device)

        prediction = torch_module(input_var, *shift_buffer) #remove *shift_buffer if using arch mobilenet

        feat, shift_buffer = prediction[0], prediction[1:]

        coefs = feat.cpu().detach().numpy() # Move tensor back to CPU to process numpy arrays
        coefs2 = coefs.copy()            

        # Check 
        if SOFTMAX_THRES > 0:
        
            feat_np = coefs2.reshape(-1)

            #print(feat_np)

            feat_np -= feat_np.max()
            softmax = np.exp(feat_np) / np.sum(np.exp(feat_np))
            
            #print(np.round(softmax,2))

            if max(softmax) > SOFTMAX_THRES:
                
                idx_ = np.argmax(feat.cpu().detach().numpy())

                #print("GOT SOFTMAX > 0.7")
    
            else:
                idx_ = idx

        else:                
            idx_ = np.argmax(feat.cpu().detach().numpy()[0]) # For demo mobilenet



        if HISTORY_LOGIT:
            history_logit.append(feat.cpu().detach().numpy())
            history_logit = history_logit[-int(12/27*num_classes):]
            avg_logit = sum(history_logit)
            idx_ = np.argmax(avg_logit, axis=1)[0] # For demo mobilenet
            #idx_ = np.argmax(avg_logit)  # For archnet mobilenet

        idx, history = process_output(idx_, history, num_classes)

        t2 = time.time()
        
        current_status = categories[idx]
        print(f"Prediction @ Frame {index} : {categories[np.argmax(feat.cpu().detach().numpy())]}")
        print("Status: " + str(current_status))
        

        running_preds.append(idx)
        current_time = t2 - t1

        fall_prob = round(softmax[1],2)
        notfall_prob = round(softmax[2],2)
        
        # Display fall/not fall softmax probabilites
        cv2.putText(img, 'fall: ' + str(fall_prob), topRightCornerOfText, font2, fontScale2, fontColor2, lineType2)
        cv2.putText(img, 'not fall: ' + str(notfall_prob), topRightCornerOfText2, font2, fontScale2, fontColor2, lineType2)


        # If fall is detected, display warning
        if idx == 1:
            cv2.putText(img,'FALL DETECTED', bottomLeftCornerOfText, font, fontScale, fontColor, lineType)


        # ALERT Logic 1: If more than 5 Falls captured in last 7 values
        if len(running_preds) > 7:
            running_preds.pop(0)
            fall_counts = running_preds.count(1)

            if fall_counts < 3 and state == "warning":
                print("RETURNED TO NORMAL STATE")
                state = "normal"

            elif fall_counts >=3 and fall_counts <= 5:
                
                if state == "normal":
                    print("WARNING: POTENTIAL FALL DETECTED")
                
                state = "warning"
            
            elif fall_counts > 5:
                print("5 of last 7 frames were falls")
                print("ALERT! FALL HAS HAPPENED!!")
                
                return True

        
        print("")


        # This is to show the camera image and prediction
        img = cv2.resize(img, (480, 480))
        #img = img[:, ::-1]

        # This is to also how the graph to track the labels
        if TRACK_LABELS:
            #tracker[categories[idx]] += 1
            #tracker = {c:0 for c in categories}
            #print(tracker)            
            tracker = None
            tracker = {}
            for i in range(num_classes):                 
                tracker[categories[i]] = softmax[i]

            # Count frames
            frame_counter[categories[idx]] += 1

            # Set figures
            fig, (ax1, ax2) = plt.subplots(1,2)
            fig.tight_layout(pad=5.0)

            # Plot graphs
            ax1.bar(tracker.keys(), tracker.values(), color="red")
            ax1.set_title("Frame Probability Distribution")
            ax1.set_ylabel("# of Frames", fontsize=10)

            ax2.bar(frame_counter.keys(), frame_counter.values(), color="blue")
            ax2.set_title("Total Frame Counts")
            ax2.set_ylabel("Probability", fontsize=10)

            plt.savefig('plot_fig.png')

            img_plot = cv2.imread('plot_fig.png')
            img_plot = cv2.resize(img_plot, (img.shape[1], img.shape[0]))
            img = np.vstack((img, img_plot))


        #cv2.imwrite("./frames/esh_" + str(i_frame) +".jpg", img)
        cv2.imshow(WINDOW_NAME, img)

        key = cv2.waitKey(1)

        if key & 0xFF == ord('q') or key == 27:  # exit
            break
        
        elif key == ord('F') or key == ord('f'):  # full screen
            print('Changing full screen option!')
            
            full_screen = not full_screen
            
            if full_screen:
                print('Setting FS!!!')
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            
            else:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)


        if t is None:
            t = time.time()
        
        else:
            nt = time.time()
            index += 1
            t = nt


    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    print("Starting... \n")

    parser = argparse.ArgumentParser(description="TSM testing")
    parser.add_argument('--video', type=str, default=None)

    args = parser.parse_args()

    

    SOFTMAX_THRES = 0.8
    HISTORY_LOGIT = False
    REFINE_OUTPUT = False
    WINDOW_NAME = "GESTURE CAPTURE"
    TRACK_LABELS = False
    CAMERA_FEED = True
    SEND_ALERTS = False
    VIDEO_PATH = ""
    
    if args.video is not None:
        VIDEO_PATH = args.video
        CAMERA_FEED = False

    else:
        CAMERA_FEED = True



    print("VIDEO_PATH = " + VIDEO_PATH)
    
    


    #Modify number of classes here
    alert = main(3)

    # Your Account Sid and Auth Token from twilio.com/console
    # DANGER! This is insecure. See http://twil.io/secure
    if alert and SEND_ALERTS:
        account_sid = 'ACdbfaa05b13c92b8c951ab45088604ad3'
        auth_token = '54093a9c451d4237cfcbf9f86deeb34a'
        client = Client(account_sid, auth_token)

        message = client.messages.create(
            body='It seems you have fallen. Emergency professionals are on their way',
            from_='',
            to=''
        )

        print(message.sid)


    print("Done")
