#!/usr/bin/env python
import os
from erf_settings import *
import numpy as np
# from scipy.special import softmax
from tools import prob_to_lines as ptl
import cv2
import models
import torch
import torch.nn.functional as F
from options.options import parser
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
# from PIL import Image


import rospy,rospkg

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
import yaml
import argparse
cap_name = './data/test.mp4'
image = './data/10.jpg'


class Img_Sub():
    def __init__(self,cfg):
        self.bridge = CvBridge()
        self.image_raw_sub= rospy.Subscriber(cfg['image_src'], Image, self.callback)
        self.image_ok = False
    def callback(self, msg):
        
        self.image_raw = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.image_ok = True



def main(cfg):
    device = torch.device("cuda:0")
    
    img_sub = Img_Sub(cfg)
    # lane_left_pub = rospy.Publisher("Lane/mask_left", Image)
    # lane_right_pub = rospy.Publisher("Lane/mask_right", Image)
    mask_pub = rospy.Publisher("Lane/mask", Image)
    lane_color_pub = rospy.Publisher("Lane/pred", Image,queue_size=10)
    lane_exist_pub = rospy.Publisher("Lane/exist",Float32MultiArray,queue_size=10)
    
    
    # model
    model = models.ERFNet(5)
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()

    if cfg['resume']:
        if os.path.isfile(cfg['resume']):
            print(("=> loading checkpoint '{}'".format(cfg['resume'])))
            checkpoint = torch.load(cfg['resume'])
            
            torch.nn.Module.load_state_dict(model, checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})".format(cfg['evaluate'], checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(cfg['resume'])))

    # cudnn.benchmark = True
    # cudnn.fastest = True

    rate = rospy.Rate(cfg['publish_rate'])
    bridge = CvBridge()

    # check all ros node ready
    while(1):
        if(img_sub.image_ok):
            break
    
    input_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ]
    )
    with torch.no_grad():
        
        while not rospy.is_shutdown():
            
            image_src = img_sub.image_raw.copy()
            in_frame = cv2.resize(image_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
            croppedImage = in_frame[VERTICAL_CROP_SIZE:, :, :]  # FIX IT
            croppedImageTrain = cv2.resize(croppedImage, (TRAIN_IMG_W, TRAIN_IMG_H), interpolation=cv2.INTER_LINEAR)
            # croppedImageTrain = cv2.resize(image_src, (TRAIN_IMG_W, TRAIN_IMG_H), interpolation=cv2.INTER_LINEAR)
            image = input_transform(croppedImageTrain)
            image = image.unsqueeze(0)
            input_var = image.to(device)
            
            # Comput
            output, output_exist = model(input_var)
            output = F.softmax(output, dim=1)
            pred = np.clip(output.data.cpu().numpy(),0,None) # BxCxHxW
            pred_color = np.zeros((TRAIN_IMG_H,TRAIN_IMG_W,3))
            
            # pred_color[...,0] = pred[0][0] 
            pred_color[...,1] =  pred[0][2] # 
            pred_color[...,2] =  pred[0][3]  #  4 left
            pred_color = (pred_color*255.0).astype(np.uint8)
            

            
            pred_color = cv2.resize(pred_color.copy(), (IN_IMAGE_W, IN_IMAGE_H-VERTICAL_CROP_SIZE), interpolation=cv2.INTER_LINEAR)
            pred_color_expand = np.zeros((IN_IMAGE_H,IN_IMAGE_W,3),dtype= np.uint8)
            pred_color_expand[VERTICAL_CROP_SIZE:,...] = pred_color

            # if output_exist[0][1] > 0.9:
            #     cv2.circle(pred_color_expand,(50, 50), 15, (0, 255, 0), -1)
            # if output_exist[0][2] > 0.9:
            #     cv2.circle(pred_color_expand,(IN_IMAGE_W-50, 50), 15, (0, 0, 255), -1)
            
            lane_color_pub.publish(bridge.cv2_to_imgmsg(pred_color_expand,"bgr8"))
            
            prob_map_left = pred[0][2]
            prob_map_left_left = pred[0][1]
            
            prob_map_right= pred[0][3]
            prob_map_right_right= pred[0][4]

            prob = ((prob_map_left  +prob_map_right )*255).astype(np.uint8)
            prob = cv2.GaussianBlur(prob,(7,7),0)
            # prob = cv2.adaptiveThreshold(prob,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
            ret2,prob =  cv2.threshold(prob,80,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            # prob = prob > 0.3
            # mask = (np.logical_or(prob_map_left,prob_map_right)*255).astype(np.uint8)
            mask =  prob.copy()
            mask = cv2.resize(mask.copy(), (IN_IMAGE_W, IN_IMAGE_H-VERTICAL_CROP_SIZE), interpolation=cv2.INTER_LINEAR)
            
            mask_expand = np.zeros((IN_IMAGE_H,IN_IMAGE_W),dtype= np.uint8)
            mask_expand[VERTICAL_CROP_SIZE:,:] = mask
            mask_pub.publish(bridge.cv2_to_imgmsg(mask_expand,"mono8"))
            
            output_exist = output_exist[0].cpu().float().numpy().tolist()
            exist_msg = Float32MultiArray()
            exist_msg.data = output_exist
            lane_exist_pub.publish(exist_msg)
            
            rate.sleep()
        
    print('exit')

def test(model, image_src):

    in_frame_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)

    # Input
    in_frame = cv2.resize(in_frame_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    croppedImage = in_frame[VERTICAL_CROP_SIZE:, :, :]  # FIX IT
    croppedImageTrain = cv2.resize(croppedImage, (TRAIN_IMG_W, TRAIN_IMG_H), interpolation=cv2.INTER_LINEAR)

    input_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ]
    )

    image = input_transform(croppedImageTrain)
    image = image.unsqueeze(0)
    input_var = torch.autograd.Variable(image)

    # Comput
    output, output_exist = model(input_var)
    output = F.softmax(output, dim=1)
    pred = output.data.cpu().numpy()  # BxCxHxW
    pred_exist = output_exist.data.cpu().numpy()

    maps = []
    mapsResized = []
    exists = []
    img = Image.fromarray(cv2.cvtColor(croppedImageTrain, cv2.COLOR_BGR2RGB))

    for l in range(LANES_COUNT):
        prob_map = (pred[0][l + 1] * 255).astype(int)
        prob_map = cv2.blur(prob_map, (9, 9))
        prob_map = prob_map.astype(np.uint8)
        maps.append(prob_map)
        mapsResized.append(cv2.resize(prob_map, (IN_IMAGE_W, IN_IMAGE_H_AFTER_CROP), interpolation=cv2.INTER_LINEAR))
        img = ptl.AddMask(img, prob_map, COLORS[l],0.4)  # Image with probability map

        exists.append(pred_exist[0][l] > 0.5)
        lines = ptl.GetLines(exists, maps)

    print(exists)
    res_img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
    cv2.imshow("result_pb", res_img)

    for l in range(LANES_COUNT):
        points = lines[l]  # Points for the lane
        for point in points:
            cv2.circle(image_src, point, 5, POINT_COLORS[l], -1)

    cv2.imshow("result_points", image_src)
    cv2.waitKey(100)




if __name__ == '__main__':
    rospy.init_node('Lane_light_condiction_style_transfer', anonymous=True)
    rospack = rospkg.RosPack()
    
    pkg_root = os.path.join(rospack.get_path('lane_detect'),'src','Light-Condition-Style-Transfer')


    # Loading setting from config file
    with open(os.path.join(pkg_root,"configs/demo.yml")) as fp:
        cfg = yaml.load(fp)
    poblished_rate = rospy.get_param("~det_rate")
    image_src = rospy.get_param('~image_src')
    
    cfg['resume'] = os.path.join(pkg_root,cfg['resume'])
    cfg['publish_rate'] = poblished_rate
    cfg['image_src'] = image_src
    # Loading ros param


    main(cfg)