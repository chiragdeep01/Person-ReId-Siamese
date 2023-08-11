import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2
import collections
import timm


# EFFICIENTNET_B0 FOR FEATURE EXTRACTIOn 
class CustomEfficientNet(nn.Module):
    
    def __init__(self, emb_size = 512):
        super(CustomEfficientNet, self).__init__()
        
        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=True)
        
        # REPLACING OF CLASSIFICATION FOR FEATURE EXTRACTION
        self.efficientnet.classifier = nn.Linear(in_features = self.efficientnet.classifier.in_features,
                                                out_features = emb_size)
        
    
    def forward(self, images):
        features = self.efficientnet(images)
        return features
    
class EffcientNetReId:
    
    def __init__(self, weights_path):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = CustomEfficientNet() # LOADING EFFICIENTNET MODEL
        self.model.load_state_dict(torch.load(weights_path))
        self.model.to(self.device)
        
        self.track = collections.OrderedDict()
        self.track_capacity = 10 
        self.threshold = 0.75 # THRESHOLD FOR COSINE SIMLARITY
        
        self.colors = {
        0: (0, 0, 255),    # Red
        1: (0, 255, 0),    # Green
        2: (255, 0, 0),    # Blue
        3: (255, 255, 0),  # Yellow
        4: (255, 0, 255),  # Magenta
        5: (0, 255, 255),  # Cyan
        6: (128, 0, 0),    # Maroon
        7: (0, 128, 0),    # Olive
        8: (0, 0, 128),    # Navy
        9: (128, 128, 128) # Gray
    }
        self.ID_available = [i for i in range((self.track_capacity-1),-1,-1)]
        
    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2,0,1)/255.0 # HWC TO CHW FOR PYTORCH
        return img
    
    def feature(self, image):
        
        with torch.no_grad():
            image_tensor = self.preprocess(image)
            image_tensor = image_tensor.to(self.device)
            
            feature_vector = self.model(image_tensor.unsqueeze(0))
            
        return feature_vector
    
    # ADDING USING LRU METHOD
    def add2Track(self,feature):
        
        if len(self.track)>=self.track_capacity: 
            removed_feature_id, _ = self.track.popitem(last = False) # POP LEAST RECENTLY MATCHED FEATURE
            self.ID_available.append(removed_feature_id)
            
        new_id = self.ID_available.pop() # ASSIGN NEW ID 
        self.track[new_id] = feature
        
        return False, new_id, self.colors[new_id]
    
    def compare(self, image):
        
        feature1 = self.feature(image)
        max_id = [-1,0]
        
        for id in self.track:
            similarity = F.cosine_similarity(feature1, self.track[id])
            
            if similarity.item()>self.threshold and similarity.item()>max_id[1]: 
                max_id = [id,similarity.item()]
                
        if max_id[0]>-1:
            self.track.move_to_end(max_id[0]) #UPDATING MOST RECENTY MATCHED FEATURE FOR LRU
            return True, max_id[0],self.colors[max_id[0]]   
        
        return self.add2Track(feature1)
