# Person-ReId-Siamese

Person Re-Identification using Siamese Network with EfficientNet_b0 as architecture.

## Triplet Loss
The network has been trained using tripletLoss function with cosine similarity.  
Loss(a,n,p) = max(0, c(a, n) - c(a, p) + margin) where,  
- a = Anchor sample embeddings  
- n = Negative sample embeddings  
- p = Positive sample embeddings  
- c = Cosine Similarity

## Model
I have used EfficientNet_b0 for this task and replaced the final classication layer with a Linear layer with 512 ouputs which would output feature embeddings for a given Image.  

## Training
For training, the dataset is arranged in the form of Anchor Image, Negative Image, Positive Image. All three images are passed through the network to get their individual embeddings and the passed to the loss function and the Adam Optimizer updates the weights on the basis of Triplet Loss with these embeddings.


