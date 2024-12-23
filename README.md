# Drone-Detection_Yolov10
From dataset https://universe.roboflow.com/drone-detection-pexej/drone-detection-data-set-yolov7/dataset/1  a model is obtained, based on yolov10 to detect drones in images. 

=== Installation:

 Download all project datasets to a folder on disk.

Install yolov10 (if not yet installed) following the instructions given at: https://blog.roboflow.com/yolov10-how-to-train/ 

which may be reduced to !pip install -q git+https://github.com/THU-MIG/yolov10.git

If you already have ultralytics installed, it would be advisable to upgrade ultralytics, unless you have applications based on yolov10 without updating, which could be affected by the update.

You must have an upgraded version of ultralytics and the proper version of lap, for that:

inside conda in the scripts directory of the user environment:

python pip-script.py install --no-cache-dir "lapx>=0.5.2"

upgrade ultralytics:

python pip-script.py install --upgrade ultralytics

And download from https://github.com/THU-MIG/yolov10/releases the yolov10n.pt model. In case this operation causes problems, this file is attached with the rest of the project files.

Unzip the Test1.zip folder

Some zip decompressors duplicate the name of the folder to be decompressed; a folder that contains another folder with the same name, should only contain one. In these cases it will be enough to cut the innermost folder and copy it to the project folder.

=== Test:

Execute:

Test_drone-detection_Yolov10.py
 
that evaluate the 39 first images downloaded from 

https://www.kaggle.com/datasets/sshikamaru/drone-yolo-detection/data

This images are independent of training process and has different sizes.

The images are presented on the screen with a red box , or several red boxes, indicating the predictions, and the confidence of predicted drone detection.

The model has been obtained with a MAP50 of 0.868 and MAP50-95 of 0.501 corresponding to epoch 33 of the training (see log in the attached LOG.txt file, and results.png)

Comparing the results with those obtained in the reference project https://universe.roboflow.com/drone-detection-pexej/drone-detection-data-set-yolov7/dataset/1 , the results are similar.

Since the results are not good, they can be optimized by running the program that uses the predictions of several models in cascade:

Test_drone-detection_SeveralModels_Yolov10.py

Visually checking:

Images 10, 16, 18, 30 and 35 are not detected
 
Images 7 and 33 are incorrectly detected

Images 12,15, 17 and 19  are detected, although with a certain imprecision

The rest of the 39 images are detected with precision

It would be  82-70% precision


=== Training

The project comes with an optimized model: last33epoch.pt

To obtain this model, the following has been executed:

 Download de dataset

 https://universe.roboflow.com/drone-detection-pexej/drone-detection-data-set-yolov7/dataset/1

If you do not have a roboflow user key, you can obtain one at
https://docs.roboflow.com/api-reference/authentication

After downloading the dataset, a folder Drone-Detection-data-set(yolov7)-1 is created which must be moved to the project folder

Execute:

Train_drone-detection_Yolov10.py

This program has been adapted from

 https://medium.com/@huzeyfebicakci/custom-dataset-training-with-yolov10-a-deep-dive-into-the-latest-evolution-in-real-time-object-ab8c62c6af85

It assumes that the project is located in the folder 
“Drone-Detection_Yolov10”, 

otherwise the assignment must be changed by modifying line 22 .

The parameter multi_scale has been changed to true.

also uses the .yaml file:

data.yaml

In data.yaml the absolute addresses of the project appear assuming that it has been installed on disk C:, if it has another location these absolute addresses will have to be changed.

Evaluate the model running

python Evaluate_drone-detection_Yolov10.py

changiing line 17

#dirnameYolo="runs\\train\\exp\\weights\\last.pt"
dirnameYolo="last33epoch.pt"

with the model that appears in the directory runs\\train\\expnn\\weights\\last.pt after the training
where expnn is the las directory en runs\\train

in green appears the labeled object, in red the predicted and, so as not to confuse the image, a text above with the  conf of prediction or predictios

Comparing the results with those obtained in the reference project https://universe.roboflow.com/drone-detection-pexej/drone-detection-data-set-yolov7/dataset/1 , the results are similar.

=== References

https://universe.roboflow.com/drone-detection-pexej/drone-detection-data-set-yolov7/dataset/1

https://www.kaggle.com/datasets/sshikamaru/drone-yolo-detection/data

https://medium.com/@huzeyfebicakci/custom-dataset-training-with-yolov10-a-deep-dive-into-the-latest-evolution-in-real-time-object-ab8c62c6af85 

https://github.com/ablanco1950/Drone_Detection-SVR

https://github.com/ablanco1950/brain-tumors-detection_yolov10

https://github.com/ablanco1950/Fracture.v1i_Reduced_YoloFromScratch

https://github.com/ablanco1950/bone-fracture-7fylg_Yolov10

https://github.com/ablanco1950/BrainTumor_sagittal_t1wce_Yolov10

https://github.com/ablanco1950/PointOutWristPositiveFracture_on_xray

https://github.com/ablanco1950/Kidney_Stone-Yolov10
