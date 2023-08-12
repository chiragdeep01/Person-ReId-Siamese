from ultralytics import YOLO
import cv2
import numpy as np
from ReId import EffcientNetReId
import utils
import json
import torch

class Cam:
    
    def __init__(self, id, camname, ip):
        self.id = id
        self.cam_name = camname
        self.ip = ip
        self.cap = cv2.VideoCapture(self.ip)
    
    def get_frame(self):
        try:
            ret, self.frame = self.cap.read()
            return ret
        except:
            self.cap.release()
            self.cap = cv2.VideoCapture(self.ip)
            return False

def load_cams(json_path):
    
    cams = []
    
    f = open(json_path)
    data = json.load(f)
    i = 0
    for cam in data['cameras']:
        i+=1
        cams.append(Cam(i, cam["camname"], cam["ip"]))
        
    return cams
        
if __name__ == "__main__":
    
    cams_jsonPath = "cams.json"
    yolo_path = "models/yolov8n.pt"
    efficientNet_path = 'models/Eff.pt'
    
    yoloModel = YOLO(yolo_path)
    
    if torch.cuda.is_available():
        yoloModel.to('cuda')
        device = 0  # GPU
    else:
        device = 'cpu'
    
    ReId = EffcientNetReId(efficientNet_path)
    
    cams = load_cams(cams_jsonPath)
    
    while True:
        for c in cams:
            ret = c.get_frame()
            if ret:
                results = yoloModel(c.frame, device = device)
                boxes = results[0].boxes.data
                
                for det in boxes:
                    if det[5]==0 and det[4]>0.8:
                        found, id, color = ReId.compare(c.frame[int(det[1]):int(det[3]),
                                                            int(det[0]):int(det[2])])
                        
                        utils.draw_box(c.frame,[int(det[0]),int(det[1]),int(det[2]),
                                        int(det[3])],color)
                        
                        utils.draw_caption(c.frame,[int(det[0]),int(det[1]),int(det[2]),
                                            int(det[3])],str(id),color)
                            
                cv2.imshow(c.cam_name, c.frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()


