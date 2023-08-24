# Person-ReId-Siamese

Person Re-Identification using Siamese Network with EfficientNet_b0 as architecture and Yolov8 for person Detection.

## Usage
- [main.py](https://github.com/chiragdeep01/Person-ReId-Siamese/blob/main/main.py) - Script for Running the ReId on multiple cameras.
- [ReId.py](https://github.com/chiragdeep01/Person-ReId-Siamese/blob/main/ReId.py) - Contains EfficientNet_b0 changes and ReIdentification class. The ReIdentification involves storing features of persons so I have set-up a max capacity for it and applied Least Recenty Used (LRU) on it. You can change the track_capacity but make sure to add colors to the dictionary. Increasing track_capacity will slow down the program !!!!
- [efficientnetcosine.ipynb](https://github.com/chiragdeep01/Person-ReId-Siamese/blob/main/efficientnetcosine.ipynb) - Training Script
- [cams.json](https://github.com/chiragdeep01/Person-ReId-Siamese/blob/main/cams.json) - For cameras configurations.

## Results
This was done on Validation data.
- Blue - Cosine Similarity between Anchor and Negative samples.  
- Green - Cosine Similarity between Anchor and Positive samples.  
![Before Training Graph](https://github.com/chiragdeep01/Person-ReId-Siamese/blob/main/results/before.jpg?raw=true)
![After Training Graph](https://github.com/chiragdeep01/Person-ReId-Siamese/blob/main/results/after.jpg?raw=true)

## Triplet Loss
The EfficientNet_b0 network has been trained using tripletLoss function with cosine similarity.  
Loss(a,n,p) = max(0, c(a, n) - c(a, p) + margin) where,  
- a = Anchor sample embeddings  
- n = Negative sample embeddings  
- p = Positive sample embeddings  
- c = Cosine Similarity

## Model
I have used EfficientNet_b0 for this task and replaced the final classication layer with a Linear layer with 512 ouputs which would output feature embeddings for a given Image.  

## Training
For training, the dataset is arranged in the form of Anchor Image, Negative Image, Positive Image. All three images are passed through the EfficientNet_b0 network to get their individual embeddings and the passed to the loss function and the Adam Optimizer updates the weights on the basis of Triplet Loss with these embeddings.


