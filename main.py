import sys
sys.path.insert(1, '/temporal-shift-module/online_demo')

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


def get_categories(num_classes):

    if num_classes == 27:
        catigories = [
        "Doing other things",  # 0
        "Drumming Fingers",  # 1
        "No gesture",  # 2
        "Pulling Hand In",  # 3
        "Pulling Two Fingers In",  # 4
        "Pushing Hand Away",  # 5
        "Pushing Two Fingers Away",  # 6
        "Rolling Hand Backward",  # 7
        "Rolling Hand Forward",  # 8
        "Shaking Hand",  # 9
        "Sliding Two Fingers Down",  # 10
        "Sliding Two Fingers Left",  # 11
        "Sliding Two Fingers Right",  # 12
        "Sliding Two Fingers Up",  # 13
        "Stop Sign",  # 14
        "Swiping Down",  # 15
        "Swiping Left",  # 16
        "Swiping Right",  # 17
        "Swiping Up",  # 18
        "Thumb Down",  # 19
        "Thumb Up",  # 20
        "Turning Hand Clockwise",  # 21
        "Turning Hand Counterclockwise",  # 22
        "Zooming In With Full Hand",  # 23
        "Zooming In With Two Fingers",  # 24
        "Zooming Out With Full Hand",  # 25
        "Zooming Out With Two Fingers"  # 26
    ]

    elif num_classes == 9: 

        catigories = ["Fall", "SalsaSpin", "Taichi", "WallPushups", "WritingOnBoard", "Archery", "Hulahoop", "Nunchucks", "WalkingWithDog"]
    
    elif num_classes == 10:

        catigories = ["Test", "Fall", "SalsaSpin", "Taichi", "WallPushups", "WritingOnBoard", "Archery", "Hulahoop", "Nunchucks", "WalkingWithDog"]

    elif num_classes == 3 :

        catigories = ['Test', "Fall", "Not Fall"]

    elif num_classes == 2:

        catigories = ["Fall", "Not Fall"]


    return catigories


def main(num_classes):

    print("Initializing model...")


    # print settings
    print("Model = MobileNet")
    print("SOFTMAX_THRESHOLD = " + str(SOFTMAX_THRES))
    print("HISTORY_LOGIT = " + str(HISTORY_LOGIT))
    print("CAMERA_FEED = " + str(CAMERA_FEED))
    print("TWILIO_ALERTS = " + str(SEND_ALERTS))



    if num_classes not in [2, 3, 9, 10, 27]:
        return "Can only handle 2, 3, 9, 10 (Fall) and 27 classes (Gesture)"

    else:
        catigories = get_categories(num_classes)

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

    if num_classes == 27:
        if not os.path.exists("mobilenetv2_jester_online.pth.tar"):  # checkpoint not downloaded
            print('Downloading PyTorch checkpoint...')
            url = 'https://hanlab.mit.edu/projects/tsm/models/mobilenetv2_jester_online.pth.tar'
            urllib.request.urlretrieve(url, './mobilenetv2_jester_online.pth.tar')
    

        torch_module.load_state_dict(torch.load("mobilenetv2_jester_online.pth.tar"))


    else:
        
        if num_classes == 9 or num_classes == 10:
            model_new = torch.load("../../pretrained/9cat/ckpt.best.pth.tar")
    
        elif num_classes == 2 or num_classes == 3:
            model_new = torch.load("../../pretrained/2cat/5_TSM_w251fall_RGB_mobilenetv2_shift8_blockres_avg_segment8_e25/ckpt.best.pth.tar")
            #model_new = torch.load("../../pretrained/2cat/ckpt.best.pth.tar")

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
    else:
        cap = cv2.VideoCapture('./zorian_0965.train.avi') 


    # set a lower resolution for speed up
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    
    full_screen = False
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 640, 480)
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
    tracker = {c:0 for c in catigories}
    history_for_alerts = []

    fall_frame_count = 0
    running_preds = []
    idx_ = 2 # initialize to NotFall
    state = "normal"

    while True:
        
        i_frame += 1
        _, img = cap.read()  # (480, 640, 3) 0 ~ 255

        if i_frame % 2 == 0:
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

                print(feat_np)

                feat_np -= feat_np.max()
                softmax = np.exp(feat_np) / np.sum(np.exp(feat_np))
                
                print(softmax)

                if max(softmax) > SOFTMAX_THRES:
                    
                    idx_ = np.argmax(feat.cpu().detach().numpy())

                    #print("GOT SOFTMAX > 0.7")
        
                else:
                    idx_ = idx
    
            else:                
                idx_ = np.argmax(feat.cpu().detach().numpy()[0]) # For demo mobilenet
                #idx_ = np.argmax(feat.cpu().detach().numpy()) # For archnet mobilenet

                #coefs2 = coefs.copy()
                #print("coefs = " + str(coefs))

                #feat_np = coefs2.reshape(-1)
                #feat_np -= feat_np.max()

                #softmax = np.exp(feat_np) / np.sum(np.exp(feat_np))


            #print("The softmax = " + str(np.round(softmax,2)))


            if HISTORY_LOGIT:
                history_logit.append(feat.cpu().detach().numpy())
                history_logit = history_logit[-int(12/27*num_classes):]
                avg_logit = sum(history_logit)
                idx_ = np.argmax(avg_logit, axis=1)[0] # For demo mobilenet
                #idx_ = np.argmax(avg_logit)  # For archnet mobilenet

            idx, history = process_output(idx_, history, num_classes)

            t2 = time.time()
            
            print(f"Prediction @ Frame {index} : {catigories[np.argmax(feat.cpu().detach().numpy())]}")
            print("Status: " + str(catigories[idx]))
            

            running_preds.append(idx)
            current_time = t2 - t1

            # ALERT Logic 1: If more than 5 Falls captured in last 7 values
            if len(running_preds) > 7:
                #print("last 7: " + str(running_preds[-7::]))
                #print(running_preds[-7::])

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

        # ALERT Logic 2 (Inactive): This is to send alerts if number of falls > 0.95*27 (num gesture classes)
        # if len(history_for_alerts) > 25:
        #    history_for_alerts.pop(0)

        # history_for_alerts.append(catigories[idx])

        # if history_for_alerts.count("Fall") > int(0.95*27):
        #    return True


        # This is to show the camera image and prediction
        img = cv2.resize(img, (640, 480))
        img = img[:, ::-1]
        height, width, _ = img.shape
        label = np.zeros([height // 10, width, 3]).astype('uint8') + 255

        cv2.putText(label, 'Prediction: ' + catigories[idx], (0, int(height / 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(label, 'FR: {:.1f} f/s'.format(1 / current_time), (int((width-170)/2), int(height / 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(label, 'F#: {:.1f} '.format(i_frame), (width - 170, int(height / 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        img = np.concatenate((img, label), axis=0)

        # This is to also how the graph to track the labels
        if track_labels:
            tracker[catigories[idx]] += 1

            fig, ax = plt.subplots()
            plt.bar(tracker.keys(), tracker.values())
            plt.savefig('plot_fig.png')

            img_plot = cv2.imread('plot_fig.png')
            img_plot = cv2.resize(img_plot, (img.shape[1], img.shape[0]))
            img = np.hstack((img, img_plot))

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

    SOFTMAX_THRES = 0.7
    HISTORY_LOGIT = False
    REFINE_OUTPUT = False
    WINDOW_NAME = "GESTURE CAPTURE"
    track_labels = True
    CAMERA_FEED = True
    SEND_ALERTS = False


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
            from_='+14154803682',
            to='+12483858969'
        )

        print(message.sid)


    print("Done")
