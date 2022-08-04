import math
import numpy as np

def matching(x,input_min,input_max,output_min,output_max):
    return (x-input_min)*(output_max-output_min)/(input_max-input_min)+output_min #map()함수 정의.

def steering(w1, w2):
    if np.abs(w1) > np.abs(w2):  # 우회전
        if w1 * w2 < 0:  #정방향 or 약간 틀어진 방향
            w1 = -w1
            angle = np.arctan(np.abs(math.tan(w1) - math.tan(w2)) / (1 + math.tan(w1)*math.tan(w2)))
            theta = matching(angle, np.pi/2, np.pi/2, 90, 180)
        elif w1 * w2 > 0:  #극한으로 틀어진 방향
            if w1 > w2:
                theta = 0
            else:
                theta = 0
        else:
            theta = 0
    elif np.abs(w1) < np.abs(w2) :  # 좌회전
        if w1 * w2 < 0:  #정방향 or 약간 틀어진 방향
            w1 = -w1
            angle = np.arctan(np.abs(math.tan(w1) - math.tan(w2)) / (1 + math.tan(w1)*math.tan(w2)))
            theta = matching(angle, np.pi/2, np.pi2, 0, 90)
        elif w1 * w2 > 0:  #극한으로 틀어진 방향
            if w1 > w2:
                theta = 0
            else:
                theta = 0
        else:
            theta = 0

    return theta



a, b = map(float, input().split())


print(steering(a,b))