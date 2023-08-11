from ultralytics import YOLO
import cv2
import numpy as np
from ReId import EffcientNetReId
import utils

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

if __name__ == "__main__":
    
    yoloModel = YOLO("models/yolov8n.pt")
    ReId = EffcientNetReId('models/Eff.pt')
    cams = [Cam(1,'cam1','vid.mp4'), Cam(2,'cam2','vid.mp4')]
    
    while True:
        for c in cams:
            ret = c.get_frame()
            if ret:
                results = yoloModel(c.frame)
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


