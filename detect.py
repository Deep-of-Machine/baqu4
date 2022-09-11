import argparse
from cmath import isnan
import shutil
import ssl
import time
from pathlib import Path
from sys import platform
#from steering_weight import matching, steering
from models import *
from utils.datasets import *
from utils.utils import *
from sklearn.cluster import KMeans
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

px = (1920, 1080)

outer = 90

# 두번째 이전꺼로 저장, 현재는 그냥 저장하고 이와 다르면 모두 다 이상치

def outer_control(x):
    global outer
    if 45 < x < 135 :
        print('######### 절대 이상치 처리 ############')
    else:
        if outer - 20 < x < outer + 20:
            outer = x
            return x
        else:
            outer = x
            print('######### 상대 이상치 처리 ############')
        
        
virtual_right_line = []
virtual_left_line = []

import math
import numpy as np

def matching(x,input_min,input_max,output_min,output_max):
    return (x-input_min)*(output_max-output_min)/(input_max-input_min)+output_min #map()함수 정의.

def steering_theta(w1, w2):
    if np.abs(w1) > np.abs(w2):  # 우회전
        if w1 * w2 < 0:  #정방향 or 약간 틀어진 방향
            w1 = -w1
            angle = np.arctan(np.abs(math.tan(w1) - math.tan(w2)) / (1 + math.tan(w1)*math.tan(w2)))
            theta = matching(angle, 0, np.pi/2, 90, 0)
        elif w1 * w2 > 0:  #극한으로 틀어진 방향
            if w1 > w2:
                theta = 90
            else:
                theta = 90
        else:
            theta = 0
    elif np.abs(w1) < np.abs(w2) :  # 좌회전
        if w1 * w2 < 0:  #정방향 or 약간 틀어진 방향
            w1 = -w1
            angle = np.arctan(np.abs(math.tan(w1) - math.tan(w2)) / (1 + math.tan(w1)*math.tan(w2)))
            theta = matching(angle, 0, np.pi/2, 90, 180)
        elif w1 * w2 > 0:  #극한으로 틀어진 방향
            if w1 > w2:
                theta = 90
            else:
                theta = 90
        else:
            theta = 0
    else:
        theta = 90

    return theta

def steering_vanishing_point(x):
    standard_x = int(1920/2)
    diff = standard_x - x 
    if diff > 0:   #좌회전
        theta = matching(diff, 0, 1920/2, 90, 45)
    elif diff < 0:
        theta = matching(diff, 0, -1920/2, 90, 135)

    return theta


def detect(
        cfg,
        weights,
        images,
        output='output',  # output folder
        img_size=416,
        conf_thres=0.3,
        nms_thres=0.45,
        save_txt=False,
        save_images=True,
        webcam=True
):
    
    device = torch_utils.select_device('mps:0')
    # device = torch.device('mps:0')
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder

    # Initialize model
    model = Darknet(cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        if weights.endswith('yolov3.pt') and not os.path.exists(weights):
            if (platform == 'darwin') or (platform == 'linux'):
                os.system('wget https://storage.googleapis.com/ultralytics/yolov3.pt -O ' + weights)
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)
    
    model.to(device).eval()

    # Set Dataloader
    if webcam:
        save_images = False
        dataloader = LoadWebcam(img_size=img_size)
    else:
        dataloader = LoadImages(images, img_size=img_size)

    # Get classes and colors
    classes = load_classes(parse_data_cfg('cfg/coco.data')['names'])
    for i, (path, img, im0) in enumerate(dataloader):
        t = time.time()
        if webcam:
            print('webcam frame %g: ' % (i + 1), end='')
        else:
            print('image %g/%g %s: ' % (i + 1, len(dataloader), path), end='')
        save_path = str(Path(output) / Path(path).name)

        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        if ONNX_EXPORT:
            torch.onnx.export(model, img, 'weights/model.onnx', verbose=True)
            # return
        pred = model(img)
        pred = pred[pred[:, :, 4] > conf_thres]  # remove boxes < threshold

        if len(pred) > 0:
            print('----------감지----------')
            
            # Run NMS on predictions
            detections = non_max_suppression(pred.unsqueeze(0), conf_thres, nms_thres)[0]

            # Rescale boxes from 416 to true image size
            scale_coords(img_size, detections[:, :4], im0.shape).round()

            right_line = []
            left_line = []
            stop_line = []
            # Draw bounding boxes and labels of detections
            for x1, y1, x2, y2, conf, cls_conf, cls in detections:
                if save_txt:  # Write to file
                    with open(save_path + '.txt', 'a') as file:
                        file.write('%g %g %g %g %g %g\n' %
                                   (x1, y1, x2, y2, cls, cls_conf * conf))

                # Add bbox to the image
                label = plot_one_box([x1, y1, x2, y2], im0)
                # print(label,end=', ')
  
                x = int((x1 + x2)/2)
                y = int(y2)

                if label == 'blue':
                    left_line.append([x, y])
                elif label == 'yellow':
                    right_line.append([x, y])
                elif label == 'red':
                    stop_line.append([x, y])
                else:
                    pass

            left_x = []
            left_y = []
            right_x = []
            right_y = []

            stop_y = []

            
            
            ########### 차선 좌우 구분 코드 ##############

            

            # print(right_line, left_line, stop_line)

            
            im0 = cv2.line(im0,(int(px[0]/2), int(px[1])),(int(px[0]/2),int(0)),(255,0,0),3)

            im0 = cv2.line(im0,(0, int(px[1]*(2/3))),(px[0],int(px[1]*(2/3))),(0,0,255),3)  # 정지선
            try:
                ############ 한쪽 차선만 인식 ################
                if len(right_line) > 1 and len(left_line) > 1:          ## 양쪽 차선이 다 있을때
                    for i in range(len(right_line)):
                        right_x.append(right_line[i][0])
                        right_y.append(right_line[i][1])
                    for i in range(len(left_line)):
                        left_x.append(left_line[i][0])
                        left_y.append(left_line[i][1])

                    l_x_mean = np.mean(left_x) 
                    l_y_mean = np.mean(left_y)
                    r_x_mean = np.mean(right_x) 
                    r_y_mean = np.mean(right_y)

                    left_calculated_weight = ((left_x - l_x_mean) * (left_y - l_y_mean)).sum() / ((left_x - l_x_mean)**2).sum()
                    left_calculated_bias = l_y_mean - left_calculated_weight * l_x_mean
                    l_target = left_calculated_weight * px[0] + left_calculated_bias
                    print(f"왼: y = {left_calculated_weight} * X + {left_calculated_bias}")

                    right_calculated_weight = ((right_x - r_x_mean) * (right_y - r_y_mean)).sum() / ((right_x - r_x_mean)**2).sum()
                    right_calculated_bias = r_y_mean - right_calculated_weight * r_x_mean
                    r_target = right_calculated_weight * px[0] + right_calculated_bias
                    print(f"오: y = {right_calculated_weight} * X + {right_calculated_bias}")

                    cross_x = (right_calculated_bias - left_calculated_bias) / (left_calculated_weight - right_calculated_weight)
                    cross_y = left_calculated_weight*((right_calculated_bias - left_calculated_bias)/(left_calculated_weight - right_calculated_weight)) + left_calculated_bias

                    # print(left_calculated_bias, l_target)
                    im0 = cv2.line(im0,(0,int(left_calculated_bias)),(int(px[0]),int(l_target)),(0,0,0),10)
                    im0 = cv2.line(im0,(int(0),int(right_calculated_bias)),(px[0],int(r_target)),(0,0,0),10)
                    cv2.circle(im0, (int(cross_x), int(cross_y)), 10, (0, 0, 255), -1, cv2.LINE_AA)

                    if 80 < steering_theta(left_calculated_weight, right_calculated_weight) < 100:
                        print('소실점 조향 서보모터 각도: ', outer_control(steering_vanishing_point(cross_x)))
                        
                    else:
                        print("기울기 조향 서보모터 각도: ", outer_control(steering_theta(left_calculated_weight, right_calculated_weight)))


                    ## 경험적 가상 차선 생성
                    if 83 < steering_theta(left_calculated_weight, right_calculated_weight) < 93:
                        #global virtual_weight
                        virtual_right_line = []
                        virtual_left_line = []
                        virtual_left_line.append(left_calculated_weight)  # 기울기 저장
                        virtual_left_line.append(left_calculated_bias)    # 바이어스 저장
                        virtual_left_line.append(l_target)


                        virtual_right_line.append(right_calculated_weight)   
                        virtual_right_line.append(right_calculated_bias)
                        virtual_right_line.append(r_target)







                elif len(right_line) > 1 and len(left_line) < 2:           ### 오른쪽 차선만 있을때
                    for i in range(len(right_line)):
                        right_x.append(right_line[i][0])
                        right_y.append(right_line[i][1])

                    r_x_mean = np.mean(right_x) 
                    r_y_mean = np.mean(right_y)

                    right_calculated_weight = ((right_x - r_x_mean) * (right_y - r_y_mean)).sum() / ((right_x - r_x_mean)**2).sum()
                    right_calculated_bias = r_y_mean - right_calculated_weight * r_x_mean
                    r_target = right_calculated_weight * px[0] + right_calculated_bias
                    print(f"오: y = {right_calculated_weight} * X + {right_calculated_bias}")

                    

                    if len(virtual_left_line) > 0:
                        # print('가상 차선', virtual_left_line)
                        cross_x = (right_calculated_bias - virtual_left_line[1]) / (virtual_left_line[0] - right_calculated_weight)
                        cross_y = virtual_left_line[0]*((right_calculated_bias - virtual_left_line[1])/(virtual_left_line[0] - right_calculated_weight)) + virtual_left_line[1]
                        
                        im0 = cv2.line(im0,(0,int(virtual_left_line[1])),(int(px[0]),int(virtual_left_line[2])),(0,0,0),10)
                        im0 = cv2.line(im0,(int(0),int(right_calculated_bias)),(px[0],int(r_target)),(0,0,0),10)
                        cv2.circle(im0, (int(cross_x), int(cross_y)), 10, (0, 0, 255), -1, cv2.LINE_AA)

                        if 80 < steering_theta(virtual_left_line[0], right_calculated_weight) < 100:
                            print('오른쪽만; 소실점 조향 서보모터 각도: ', outer_control(steering_vanishing_point(cross_x)))
                        else:
                            print("오른쪽만; 기울기 조향 서보모터 각도: ", outer_control(steering_theta(virtual_left_line[0], right_calculated_weight)))
                    

                elif len(left_line) > 1 and len(right_line) < 2:             ### 왼쪽 차선만 있을때
                    for i in range(len(left_line)):
                        left_x.append(left_line[i][0])
                        left_y.append(left_line[i][1])

                    l_x_mean = np.mean(left_x) 
                    l_y_mean = np.mean(left_y)

                    left_calculated_weight = ((left_x - l_x_mean) * (left_y - l_y_mean)).sum() / ((left_x - l_x_mean)**2).sum()
                    left_calculated_bias = l_y_mean - left_calculated_weight * l_x_mean
                    l_target = left_calculated_weight * px[0] + left_calculated_bias
                    print(f"왼: y = {left_calculated_weight} * X + {left_calculated_bias}")


                    if len(virtual_right_line) > 0:
                        cross_x = (virtual_right_line[1] - left_calculated_bias) / (left_calculated_weight - virtual_right_line[0])
                        cross_y = left_calculated_weight*((virtual_right_line[1] - left_calculated_bias)/(left_calculated_weight - virtual_right_line[0])) + left_calculated_bias

                        # print(left_calculated_bias, l_target)
                        im0 = cv2.line(im0,(0,int(left_calculated_bias)),(int(px[0]),int(l_target)),(0,0,0),10)
                        im0 = cv2.line(im0,(int(0),int(virtual_right_line[1])),(px[0],int(virtual_right_line[2])),(0,0,0),10)
                        cv2.circle(im0, (int(cross_x), int(cross_y)), 10, (0, 0, 255), -1, cv2.LINE_AA)

                        if 80 < steering_theta(left_calculated_weight, virtual_right_line[0]) < 100:
                            print('왼쪽만; 소실점 조향 서보모터 각도: ', outer_control(steering_vanishing_point(cross_x)))
                        else:
                            print("왼쪽만; 기울기 조향 서보모터 각도: ", outer_control(steering_theta(left_calculated_weight, virtual_right_line[0])))

                    

                else:            ## 둘다 없을 떄
                    if len(virtual_right_line) > 0:
                        cross_x = (virtual_right_line[1] - left_calculated_bias) / (left_calculated_weight - virtual_right_line[0])
                        cross_y = left_calculated_weight*((virtual_right_line[1] - left_calculated_bias)/(left_calculated_weight - virtual_right_line[0])) + left_calculated_bias

                        # print(left_calculated_bias, l_target)
                        im0 = cv2.line(im0,(0,int(left_calculated_bias)),(int(px[0]),int(l_target)),(0,0,0),10)
                        im0 = cv2.line(im0,(int(0),int(virtual_right_line[1])),(px[0],int(virtual_right_line[2])),(0,0,0),10)
                        cv2.circle(im0, (int(cross_x), int(cross_y)), 10, (0, 0, 255), -1, cv2.LINE_AA)

                        if 80 < steering_theta(left_calculated_weight, virtual_right_line[0]) < 100:
                            print('둘다 없음; 소실점 조향 서보모터 각도: ', outer_control(steering_vanishing_point(cross_x)))
                        else:
                            print("둘다 없음; 기울기 조향 서보모터 각도: ", outer_control(steering_theta(left_calculated_weight, virtual_right_line[0])))

                
                ################# 정지 인식 #########################
                if len(stop_line) > 0:
                    for i in range(len(stop_line)):
                            stop_y.append(stop_line[i][1])

                    if len(stop_y) > 1:
                        if int(px[1]*(2/3)) < sorted(stop_y, reverse=True)[1]:  # 두번째로 큰 수, max(stop_y)
                            
                            print('############################# stop #################################')
                        print(sorted(stop_y, reverse=True)[1])

            except ValueError:
                pass


        dt = time.time() - t
        print('Done. (%.3fs)' % dt)
        

        # sorted_x = list(sorted_x)
        # y_sorted_by_x = list(y_sorted_by_x)
        # print(sorted_x)
        # print(y_sorted_by_x)

            
        

        if save_images:  # Save generated image with detections
            cv2.imwrite(save_path, im0)

        if webcam:  # Show live webcam
            cv2.imshow(weights, im0)
    
        print('----------')

    if save_images and (platform == 'darwin'):  # linux/macos
        os.system('open ' + output + ' ' + save_path)
        
    # import matplotlib.pyplot as plt
    # im = plt.imread('output/KakaoTalk_20220712_174745887.jpg')
    # plt.imshow(im)
    # plt.plot(leftx,lefty, c = 'black', linewidth = 2)
    # plt.plot(rightx, righty, c ='black', linewidth = 2)
    # plt.plot([px[0]/2, px[0]/2], [px[1]-100, px[1]/2], c = 'blue', linewidth = 2)
    # # plt.show()
    # plt.savefig(fname='result.png', bbox_inches='tight', pad_inches=0)
    # return result




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='path to weights file')
    parser.add_argument('--images', type=str, default='data/samples', help='path to images')
    parser.add_argument('--img-size', type=int, default=32*20, help='size of each image dimension') # best 32* 20                 #32*13
    parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold') #best!!!! 0.01                       #0.5
    parser.add_argument('--nms-thres', type=float, default=0.01, help='iou threshold for non-maximum suppression') # 0.01         #0.45
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(
            opt.cfg,
            opt.weights,
            opt.images,
            img_size=opt.img_size,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres
        )
