import math

def tanh(x):
    y = math.tanh(x)
    return y

def matching(x,input_min,input_max,output_min,output_max):
    return (x-input_min)*(output_max-output_min)/(input_max-input_min)+output_min #map()함수 정의.

def w(x):
    min = -4
    max = 1.6
    return matching(tanh(matching(x,0,90,min,max)),tanh(min),tanh(max),0,90)
    
import time
for i in range(1, 91):
    print(i, w(i))
# time.sleep(1)

import numpy as np
def angle_steering(x):
    angle = np.abs(std - x)
    return angle
    

def initial(x):
    global std
    std = x

def imu_steering(x):
    return matching(angle_steering(x), 0, 90, 90,0)

def speed2angle(a):
    s = 60
    if s == 0:
        return a
    else:
        if a < 90:
            angle = 90 - ((90 - a) * ((100-s)/100))
            return angle
        elif a > 90:
            angle = 90 + ((a - 90) * ((100-s)/100))
            return angle
        else:
            print("speed2angle error")
    
initial(180)
while True:
    x = 165
    if std > x: #우
        print(speed2angle(180 - w(imu_steering(x))))
    elif x > std:
        print(speed2angle(w(imu_steering(x))))
    else:
        print(90)

