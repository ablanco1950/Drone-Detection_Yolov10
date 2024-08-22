# -*- coding: utf-8 -*-
"""
Created on ago 2024

@author: Alfonso Blanco
"""
#######################################################################
# PARAMETERS
######################################################################
# dataset
# https://universe.roboflow.com/drone-detection-pexej/drone-detection-data-set-yolov7/dataset/1

dirname= "Test1"

dirnameYolo1="last28epoch.pt"
dirnameYolo2="last21epoch.pt"
dirnameYolo3="last16epoch.pt"
dirnameYolo4="last20epoch.pt"
#dirnameYolo5="last22epoch.pt"
#dirnameYolo5="last39epoch.pt"
dirnameYolo5="last46epoch.pt"



import cv2
import time
Ini=time.time()

#from ultralytics import YOLOv10
from ultralytics import YOLO

model1 = YOLO(dirnameYolo1)
model2 = YOLO(dirnameYolo2)
model3 = YOLO(dirnameYolo3)
model4 = YOLO(dirnameYolo4)
model5 = YOLO(dirnameYolo5)


class_list = model1.model.names
print(class_list)

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np

import os
import re

import imutils

########################################################################
def loadimages(dirname):
 #########################################################################
 # adapted from:
 #  https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
 # by Alfonso Blanco García
 ########################################################################  
     imgpath = dirname + "\\"
     
     images = []
     TabFileName=[]
   
    
     print("Reading imagenes from ",imgpath)
     NumImage=-2
     
     Cont=0
     for root, dirnames, filenames in os.walk(imgpath):
        
         NumImage=NumImage+1
         
         for filename in filenames:
             
             if re.search("\.(jpg|jpeg|JPEG|png|bmp|tiff)$", filename):
                 
                 
                 filepath = os.path.join(root, filename)
                
                 
                 image = cv2.imread(filepath)
                 image = cv2.resize(image, (640,640), interpolation = cv2.INTER_AREA)
                 
                 #print(filepath)
                 #print(image.shape)                           
                 images.append(image)
                 TabFileName.append(filename)
                 
                 Cont+=1
     
     return images, TabFileName


def unconvert(width, height, x, y, w, h):

    xmax = int((x*width) + (w * width)/2.0)
    xmin = int((x*width) - (w * width)/2.0)
    ymax = int((y*height) + (h * height)/2.0)
    ymin = int((y*height) - (h * height)/2.0)

    return xmin, ymin, xmax, ymax

# ttps://medium.chom/@chanon.krittapholchai/build-object-detection-gui-with-yolov8-and-pysimplegui-76d5f5464d6c
def Detect_drone_detectionWithYolov10 (img):
  
   Tabcrop_drone_detection=[]
   
   y=[]
   yMax=[]
   x=[]
   xMax=[]
   Tabclass_name=[]
   Tabclass_cod=[]
   Tabconfidence=[]

   cont=0
   while cont < 6:
        SwHay=0
        cont=cont+1
        if cont==1:
           results = model1(source=img)
           for j in range(len(results)):
               # may be several plates in a frame
               result=results[j]
       
               xyxy= result.boxes.xyxy.numpy()
               confidence= result.boxes.conf.numpy()
               class_id= result.boxes.cls.numpy().astype(int)
               #print("Class_id" )
               #print(class_id)
               #print("results...."+ str(len(results)))
               if len(class_id)==0 :
                #print("FALLA1")
                continue
               SwHay=1
               model=model1
               break
          
        if SwHay==1 : break 
        if cont==2:
          
           
           results = model2(source=img)
           
           for j in range(len(results)):
               # may be several plates in a frame
               result=results[j]
       
               xyxy= result.boxes.xyxy.numpy()
               confidence= result.boxes.conf.numpy()
               class_id= result.boxes.cls.numpy().astype(int)
               #print("Class_id" )
               #print(class_id)
               #print("results...."+ str(len(results)))
               if len(class_id)==0 :
                        continue
               SwHay=1
               model=model2
               break
        if SwHay==1 : break
        
        if cont==3:
           
           results = model3(source=img)
           for j in range(len(results)):
               # may be several plates in a frame
               result=results[j]
       
               xyxy= result.boxes.xyxy.numpy()
               confidence= result.boxes.conf.numpy()
               class_id= result.boxes.cls.numpy().astype(int)
               #print("Class_id" )
               #print(class_id)
               #print("results...."+ str(len(results)))
               if len(class_id)==0 :
                      continue
               SwHay=1
               model=model3
               break
        if cont==4:
           
           results = model4(source=img)
           for j in range(len(results)):
               # may be several plates in a frame
               result=results[j]
       
               xyxy= result.boxes.xyxy.numpy()
               confidence= result.boxes.conf.numpy()
               class_id= result.boxes.cls.numpy().astype(int)
               #print("Class_id" )
               #print(class_id)
               #print("results...."+ str(len(results)))
               if len(class_id)==0 :
                      continue
               SwHay=1
               model=model4
               break
        if cont==5:
           
           results = model5(source=img)
           for j in range(len(results)):
               # may be several plates in a frame
               result=results[j]
       
               xyxy= result.boxes.xyxy.numpy()
               confidence= result.boxes.conf.numpy()
               class_id= result.boxes.cls.numpy().astype(int)
               #print("Class_id" )
               #print(class_id)
               #print("results...."+ str(len(results)))
               if len(class_id)==0 :
                      continue
               SwHay=1
               model=model5
               break
          
        if SwHay==1 : break 
        continue
   
   # https://blog.roboflow.com/yolov10-how-to-train/
  
   for i in range(len(results)):
       # may be several plates in a frameh
       result=results[i]
       
       xyxy= result.boxes.xyxy.numpy()
       confidence= result.boxes.conf.numpy()
       class_id= result.boxes.cls.numpy().astype(int)
       print(class_id)
      
       out_image = img.copy()
       LabelTotal=""
       for j in range(len(class_id)):
           con=confidence[j]
           Tabconfidence.append(con)
           
           label=class_list[class_id[j]] + " " + str(con)[0:4]
           print(label)
           LabelTotal=LabelTotal+" " + label
           box=xyxy[j]
           
           crop_drone_detection=out_image[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
           
           Tabcrop_drone_detection.append(crop_drone_detection)
           y.append(int(box[1]))
           yMax.append(int(box[3]))
           x.append(int(box[0]))
           xMax.append(int(box[2]))

           # 
           Tabclass_name.append(label)
           Tabclass_cod.append(class_id[j])
          
   
   return Tabconfidence, Tabcrop_drone_detection, y,yMax,x,xMax, Tabclass_name, Tabclass_cod, LabelTotal

def plot_image(image, boxes, imageCV, TabFileName):
    """Plots predicted bounding boxes on the image"""
    cmap = plt.get_cmap("tab20b")
    #class_labels = PASCAL_CLASSES
    class_labels=class_list
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    fig.suptitle(TabFileName)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    Cont=0
    print(boxes)
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        conf=box[1]
        conf=str(conf)
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            upper_left_x * width,
            upper_left_y * height,
            s=class_labels[int(class_pred)] + " conf: " + str(conf[:3]),
            color="red",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )
      
        

        
        Cont+=1
        #if Cont > 1: break # only the most predicted box
        #break
      # rect with true fracture
   
    plt.show()

###########################################################
# MAIN
##########################################################


imagesComplete, TabFileName=loadimages(dirname)

print("Number of images to test: " + str(len(imagesComplete)))

ContError=0
ContHit=0
ContNoDetected=0

for i in range (len(imagesComplete)):
     
            gray=imagesComplete[i]              
            Tabconfidence, TabImgSelect, y, yMax, x, xMax, Tabclass_name, Tabclass_cod, LabelTotal =Detect_drone_detectionWithYolov10(gray)
            Tabnms_boxes=[]
            #print(gray.shape)
            #if TabImgSelect==[]:
            if len(TabImgSelect)==0:     
                print(TabFileName[i] + " NON DETECTED")
                ContNoDetected=ContNoDetected+1 
                continue
            else:
                #ContDetected=ContDetected+1
                print(TabFileName[i] + " DETECTED ")
                
               
            #for z in range(len(TabImgSelect)-1,0, -1):
            for z in range(len(TabImgSelect)):     
                #if TabImgSelect[z] == []: continue
                if len(TabImgSelect[z]) == 0: continue
                gray1=TabImgSelect[z]
                #cv2.waitKey(0)
                # may be several tumors, positives and negatives
                #print(x[z])
                text_color = (255,255,255)
                
                cv2.putText(gray, LabelTotal ,(20,20)
                             , cv2.FONT_HERSHEY_SIMPLEX , 1
                             , text_color, 2 ,cv2.LINE_AA)
                
                start_point=(x[z],y[z]) 
                end_point=(xMax[z], yMax[z])
                color=(255,0,0)
                
                img = cv2.rectangle(gray, start_point, end_point,color, 2)
               
            plot_image(img, Tabnms_boxes, img, TabFileName[i])
                
             
              
print("")           
print("NO detected=" + str(ContNoDetected))


print("")      
print( " Time in seconds "+ str(time.time()-Ini))
