# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 15:10:37 2026

@author: s2894562
"""

#working out image augmentation based on Scott's code for image/label loading
#13/2/26
#updated for confirmation report 17.04.26 

#packages 
import torch
torch.manual_seed(0)
from torchvision import tv_tensors
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 
from torchvision.ops import nms
import torchmetrics
from torch.utils.data import ConcatDataset
from torchvision.utils import draw_bounding_boxes


import pandas as pd
import numpy as np
import json
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time



#loading images and labels 
labels = pd.read_csv(r"C:\Users\S2894562\Documents\ML_small\all_labels.csv")
labels_populated = labels[labels['region_count']>0]
labels_populated = labels_populated.reset_index().drop('index', axis = 1)
labels_populated['region_attributes'] = '{"Status": "Live"}'

def get_attribs(row):  
    image = row['filename']
    d = json.loads(row['region_shape_attributes'])
    xmin = d['x']
    ymin =  d['y']
    xmax = xmin + d['width']
    ymax = ymin + d['height'] 
    d1 = row['region_attributes']
    attrib = json.loads(d1) 
    d1_status_dict = attrib['Status']
    label = 'Live'   
    return image, xmin, ymin, xmax, ymax, label

get_attribs(labels_populated.iloc[0,:])

boxes_dict = {'image': [] , 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [] , 'label':[]}  

n_rows = labels_populated.shape[0]
for i in range(n_rows):
    row = labels_populated.iloc[i,:]  
    image, xmin, ymin, xmax, ymax, label = get_attribs(row)
    boxes_dict['image'].append(image)
    boxes_dict['xmin'].append(xmin)
    boxes_dict['ymin'].append(ymin)
    boxes_dict['xmax'].append(xmax)
    boxes_dict['ymax'].append(ymax)
    boxes_dict['label'].append(1)

annotations = pd.DataFrame.from_dict(boxes_dict)

files_dir = r'C:\Users\S2894562\Documents\ML_small\25_labelled\all_labelled'

#creating custom dataset loader
class CustDataset(torch.utils.data.Dataset):
    def __init__(self, df, unique_imgs, indices, transforms1 = None):
        self.df = df
        self.unique_imgs = unique_imgs
        self.indices = indices
        self.transforms1 = transforms1
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
    
        image_name = self.unique_imgs[self.indices[idx]]
        
        records = self.df[self.df.image == image_name]
            
        boxes = records.values[:, 1:5].astype("float32")
        labels = records.values[:, 5].astype("int64")

        img = Image.open(files_dir + r'\\' + image_name).convert('RGB')

        boxes = tv_tensors.BoundingBoxes(
            boxes,
            format="XYXY",
            canvas_size = (img.height, img.width)
        )
        
        target = {
            "boxes": boxes, 
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

        # Apply transforms to image and target
        if self.transforms1 is not None:
            img, target = self.transforms1(img, target)

        return img, target

unique_imgs = annotations['image'].unique()

#creating train and validation datasets
train_inds, val_inds = train_test_split(range(unique_imgs.shape[0]), test_size = 0.1, random_state = 1)

#format of dataset produced
def custom_collate(data):
    return tuple(zip(*data))

#transformations to be applied to original data, and to augmented images
transform_original = v2.Compose([
    v2.ToImage(), #change from PIL to image tensor
    v2.ToDtype(torch.float32, scale=True), #scale image to have pixel values from 0 - 1
])
transform_augmented = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomPhotometricDistort(brightness= (0.8, 1.2), 
                                contrast = (0.5, 1.5),
                                saturation=(0.5, 1.5),
                                hue=(-0.01, 0.01),
                                p = 0.5),
])

#creating original data in format needed for ML, and augmented, doubling size of training data
dataset_original = CustDataset(annotations, unique_imgs, train_inds, transform_original)
dataset_augmented = CustDataset(annotations, unique_imgs, train_inds, transform_augmented)

train_dataset = ConcatDataset([
    dataset_original,
    dataset_augmented
])

#training dataloader
train_dl = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=custom_collate,
    pin_memory=torch.cuda.is_available()
)

#validation dataloader
val_dl = torch.utils.data.DataLoader(
    CustDataset(annotations, unique_imgs, val_inds, transform_original),
    batch_size = 12,
    shuffle = False,
    collate_fn = custom_collate,
    pin_memory = True if torch.cuda.is_available() else False
)


#visualising training data, augmented 
images, targets = next(iter(train_dl))

img = images[0]          # first image in batch
target = targets[0]      # its corresponding target
boxes = target["boxes"]

img_np = img.permute(1, 2, 0).cpu().numpy()

img_uint8 = (img * 255).to(torch.uint8)
img_with_boxes = draw_bounding_boxes(
    img_uint8,
    boxes,
    colors="red",
    width=2
)

plt.imshow(img_with_boxes.permute(1, 2, 0))
plt.axis("off")
plt.show()


# training the faster rcnn ##################################
# fasterrcnn_mobilenet_v3_large_320_fpn can be changed for other models that are available on the torchvision website
model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights='COCO_V1')

# The number of classes is 3 + background = 4. 0 is reserved for the background, so labels are 1, 2 and 3.
num_classes = 2

# Remove the last layer of the network and replace it with a new section to be tuned using our data

in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)



# Load previously trained model

model = torch.load(r"C:\Users\S2894562\Documents\Amy's project code\FasterRCNN_MobileNet_v3_large_15_epochs.pth", 
                   map_location=torch.device('cpu'), 
                   weights_only = False)
#error message suggested adding weights_only = False, unsu re what this does
#something weird happening here- printing 'Live' for every row of the dataset

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9, weight_decay=0.0005)
num_epochs = 30

# This section of code runs the actual model training for the no. of epochs defined above

start_time = time.time()
model.to(device)

loss_e = []
loss_v = []

min_val_loss = 100
early_stopping = 15

es_counter = 0

for epochs in range(num_epochs):
    #training part of the model
    epoch_loss = 0
    model.train()

    for images, targets in train_dl:

        imgs = [img.to(device) for img in images]
    
        new_targets = []
        for t in targets:
            boxes = t["boxes"].to(device).float()
            labels = t["labels"].to(device)
    
            new_targets.append({
                "boxes": boxes,
                "labels": labels
            })
    
        loss_dict = model(imgs, new_targets)
        loss = sum(loss_dict.values())

        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    valid_loss = 0.0
    #creating evaluation part of the model 
    with torch.no_grad():
        for images, targets in val_dl:
            imgs = [img.to(device) for img in images]
            new_targets = []
            
            for t in targets:
                boxes = t["boxes"].to(device).float()
                labels = t["labels"].to(device)
        
                new_targets.append({
                    "boxes": boxes,
                    "labels": labels
                })
                
            loss_dict = model(imgs, new_targets)
            val_loss = sum(v for v in loss_dict.values())
            valid_loss += val_loss.item()
            
        torch.cuda.empty_cache()
 
    # When trying to determine how many epochs to run the training for, this bit of code will break the run when the validation loss appears to stabilise
    if valid_loss < min_val_loss:
        min_val_loss = valid_loss
        es_counter = 0
        best_model_state = model.state_dict()
        print(f"best model was found at epoch {epochs + 1}")
    else:
        es_counter += 1
   

    if es_counter > early_stopping:
        break

 #Print out the training and validation losses for each epoch
        
    loss_e.append(epoch_loss)
    loss_v.append(valid_loss)
    
    print(f"Loss after epoch {epochs + 1}: Training {epoch_loss}, Validation {valid_loss}")

# Timer function to monitor how long the processing takes

end_time = time.time()
print(f'This code took {(end_time - start_time)/60} minutes')


# Save model weights
torch.save(best_model_state, r"C:\Users\S2894562\Documents\ML_small\fasterrnn_small_aug.pth")
#model = torch.load(r"C:\Users\S2894562\Documents\ML_small\fasterrnn_small.pth", weights_only= False)

# Saves training and validation loss values as CSV file so that they can be plotted in Excel.
x = np.arange(0,30)+1
y = loss_e
z = loss_v

loss_df = pd.DataFrame({"x":x, "training_loss":y, "validation_loss":z})


############# model evaluation ##################

model.eval()

images, targets = next(iter(val_dl))

img = images[0].to(device)

with torch.no_grad():
    output = model([img])

out_bbox = output[0]['boxes']
out_labels = output[0]['labels']
out_scores = output[0]['scores']

# This part filters down the number of boxes based on how sure the model is. The lower the number, the fewer boxes are kept

keep = nms(out_bbox, out_scores, 0.2)

# Uses the indices from keep to filter the boxes and labels

boxes = out_bbox[keep]
labels = out_labels[keep]

im = (img.permute(1,2,0).cpu().detach().numpy() * 255).astype('uint8')

vsample = Image.fromarray(im)

# Model generates predictions of live/dead/flying birds starting from an initial count of 0.

model.eval()
data = iter(val_dl).__next__()

#run_num = 0
live_count = 0
gt_count = 0
batch_counter = 1
for images, targets in val_dl:
    val_batch_size = len(images)
    
    for i in range(val_batch_size):
        img = images[i]      
        boxes_gt = targets[i]['boxes']
        labels_gt = targets[i]['labels']

        output = model([img.to(device)])

        out_bbox = output[0]['boxes']
        out_labels = output[0]['labels']
        out_scores = output[0]['scores']

        keep = nms(out_bbox, out_scores, 0.1)

        boxes_pred = out_bbox[keep]
        labels_pred = out_labels[keep]
        
# Draw coloured bounding boxes around model predictions        

        im = (img.permute(1,2,0).cpu().detach().numpy() * 255).astype('uint8')
        vsample = Image.fromarray(im)

        draw = ImageDraw.Draw(vsample)

        n_boxes_pred = len(boxes_pred)

        for k in range(n_boxes_pred):
            if labels_pred[k] == 1:
                live_count += 1
                #draw.rectangle(list(boxes_pred[k]), fill = None, outline =c)
                box = boxes_pred[k].cpu().tolist()
                draw.rectangle(box, outline="green")
# Draw pink bounding boxes around ground truths

        n_boxes_gt = len(boxes_gt)

        for j in range(n_boxes_gt):
            gt_count += 1
            #draw.rectangle(list(boxes_gt[j]), fill = None, outline=c)
            box = boxes_gt[j].cpu().tolist()
            draw.rectangle(box, outline="red")

# Show one image to check if working

        
        plt.imshow(vsample)
        plt.savefig(f'2022_valid_{batch_counter}.jpg',bbox_inches='tight')
        #plt.show()
        batch_counter += 1
            

print(f"live birds: {live_count} \n ground truth: {gt_count}")

#has worked much worse than before (counting 5 birds vs 56 ground truth in validation dataset)
#need to go through and do checks of each section. Something with data types probably gone weird
#still a bit confused about where things are being overwritten too

############### Intersection over union #################
# Function to calculate Intersection over Union (IoU)

def intersection_over_union(boxA, boxB):
    
    # determine the (x, y)-coordinates of the intersection rectangle
    boxA=torch.tensor(boxA)
    boxB=torch.tensor(boxB)
    x1 = torch.max(boxA[0], boxB[0])
    y1 = torch.max(boxA[1], boxB[1])
    x2 = torch.min(boxA[2], boxB[2])
    y2 = torch.min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    intersection_area = torch.max(torch.tensor(0.0), x2 - x1 + 1) * torch.max(torch.tensor(0.0), y2 - y1 + 1)

    # compute the area of both the prediction and ground-truth rectangles
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / (boxA_area + boxB_area - intersection_area)

    # return the intersection over union value
    return iou

# For every ground truth bounding box in an image, calculate the IoU for the predicted box with which it overlaps the most.

val_batch_size = data.__len__()

preds=[]
gt=[]
all_ious = [] #added this

for i in range(val_batch_size):
    img = images[i]
    boxes = targets[i]['boxes']
    labels_gt = targets[i]['labels']
    
    output = model([img.to(device)])
    
    out_bbox = output[0]['boxes']
    out_labels = output[0]['labels']
    out_scores = output[0]['scores']

    keep = nms(out_bbox, out_scores, 0.1)
    
    boxes_pred = out_bbox[keep]
    labels_pred = out_labels[keep]
    scores_pred = out_scores[keep]
    
    boxes_gt = boxes.cpu().detach().numpy()
    boxes_pred = boxes_pred.cpu().detach().numpy()
    
    n_true = boxes_gt.shape[0]
    n_pred = boxes_pred.shape[0]
    
    if n_pred == 0 :
        continue
    else:
    
        iou_matrix = np.zeros((n_true, n_pred))
        for k in range(n_true):
            for j in range(n_pred):
                iou_matrix[k, j] =  intersection_over_union(boxes_gt[k,:], boxes_pred[j,:])

        indx = iou_matrix.argmax(axis=1)
        matched_ious = iou_matrix[np.arange(n_true), indx] #i added this
        matched_ious = matched_ious[matched_ious >= 0.5] #added this

        all_ious.extend(matched_ious.tolist()) #added this

        matched_labels = labels_pred[indx]
        matched_boxes = boxes_pred[indx]
        matched_scores = scores_pred[indx]

        preds_matched_total = {"boxes":torch.tensor(matched_boxes), "scores":torch.tensor(matched_scores),
    "labels":torch.tensor(matched_labels)}

        preds.append(preds_matched_total)

        gt_matched_total = {"boxes":torch.tensor(boxes_gt), "labels":torch.tensor(labels_gt)}

        gt.append(gt_matched_total)
print(f"Mean IoU: {np.mean(all_ious):.4f}")#added this
print(f"Median IoU: {np.median(all_ious):.4f}")#added this


#trying to run IoU in simpler way- something not right with format of preds and targets
torchmetrics.detection.iou()
from torchmetrics.detection import IntersectionOverUnion
IntersectionOverUnion(preds, targets, box_format = 'xyxy')

# Extract only the boxes
pred_boxes = [p['boxes'] for p in preds]
gt_boxes   = [t['boxes'] for t in targets]

iou_metric = IntersectionOverUnion(box_format="xyxy")

iou_value = iou_metric(pred_boxes, gt_boxes)
print("IoU:", iou_value)

############# mAP ######################
# Calculate Mean Average Precision

from torchmetrics.detection.mean_ap import MeanAveragePrecision

mAP = MeanAveragePrecision()
mAP.update(preds, gt) # specifying prediction results and annotation data for this bounding box format
from pprint import pprint
pprint(mAP.cpu().compute())


# all is running but not sure about it. mAP only giving 0 and 1s. 
# target variable has changed from a dictionary to a list of dictionaries.



