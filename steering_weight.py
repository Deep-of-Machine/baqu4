import math
import numpy as np

def matching(x,input_min,input_max,output_min,output_max):
    return (x-input_min)*(output_max-output_min)/(input_max-input_min)+output_min #map()함수 정의.

def steering(w1, w2):
    if w1 > w2:
        if w1 * w2 < 0:  #정방향 or 약간 틀어진 방향
            w1 = -w1
            angle = np.arctan(np.abs(math.tan(w1) - math.tan(w2)) / (1 + math.tan(w1)*math.tan(w2)))
            theta = matching(angle, -1.6, 1.6, 0, 90)
        elif w1 * w2 > 0:  #극한으로 틀어진 방향
            if w1 > w2:
                angle = w2
            else:
                angle = w1
        else:
            angle = 0
    elif w1 < w2 :
        if w1 * w2 < 0:  #정방향 or 약간 틀어진 방향
            w1 = -w1
            angle = np.arctan(np.abs(math.tan(w1) - math.tan(w2)) / (1 + math.tan(w1)*math.tan(w2)))
            theta = matching(angle, -1.6, 1.6, 90, 180)
        elif w1 * w2 > 0:  #극한으로 틀어진 방향
            if w1 > w2:
                theta = w2
            else:
                theta = w1
        else:
            angle = 0

    return theta



a, b = map(float, input().split())


print(steering(a,b))