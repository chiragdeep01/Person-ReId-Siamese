import numpy as np
import cv2



def draw_box(image, box, color, thickness=2):
    
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)


def draw_caption(image, box, caption, recBg_color):
    
    (w, h), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1] - h - 20), (b[0] + w, b[1]), recBg_color, -1)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2) 
    
