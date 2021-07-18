# computer-vision


## Repository summary
### How to use?
- Everything from data preparation to model training is done using Colab Notebooks so that no setup is required locally. 
- All the relevant commentaries have been included inside the Colab Notebooks.


#### 1. References
This section provides more resources on the topic if you are looking to go deeper.

    * [1]https://www.cse.usf.edu/~r1k/MachineVisionBook/MachineVision.files/MachineVision_Chapter15.pdf

    * [2]http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture11.pdf

    * [3]https://core.ac.uk/download/pdf/295558123.pdf

    * [4]https://ai.googleblog.com/2019/08/on-device-real-time-hand-tracking-with.html

    * [5]https://developer.oculus.com/documentation/unity/unity-handtracking/

    * [6]     https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-73003-5_35

    * [7]https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Learning_Human-Object_Interaction_Detection_Using_Interaction_Points_CVPR_2020_paper.pdf

    * [8] https://levelup.gitconnected.com/product-detection-from-grocery-shelf-9db031e0ddc1

#### 2. Object Detection
![image](https://user-images.githubusercontent.com/53899191/126057888-fc998cfa-b2e6-4501-8292-057175555374.png)

##### YOLOv4 on Webcam Videos
```
# start streaming video from webcam
video_stream()
# label for video
label_html = 'Capturing...'
# initialze bounding box to empty
bbox = ''
count = 0 
while True:
    js_reply = video_frame(label_html, bbox)
    if not js_reply:
        break

    # convert JS response to OpenCV Image
    frame = js_to_image(js_reply["img"])

    # create transparent overlay for bounding box
    bbox_array = np.zeros([480,640,4], dtype=np.uint8)

    # call our darknet helper on video frame
    detections, width_ratio, height_ratio = darknet_helper(frame, width, height)

    # loop through detections and draw them on transparent overlay image
    for label, confidence, bbox in detections:
      left, top, right, bottom = bbox2points(bbox)
      left, top, right, bottom = int(left * width_ratio), int(top * height_ratio), int(right * width_ratio), int(bottom * height_ratio)
      bbox_array = cv2.rectangle(bbox_array, (left, top), (right, bottom), class_colors[label], 2)
      bbox_array = cv2.putText(bbox_array, "{} [{:.2f}]".format(label, float(confidence)),
                        (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        class_colors[label], 2)

    bbox_array[:,:,3] = (bbox_array.max(axis = 2) > 0 ).astype(int) * 255
    # convert overlay of bbox into bytes
    bbox_bytes = bbox_to_bytes(bbox_array)
    # update bbox so next frame gets new overlay
    bbox = bbox_bytes
```
#### 3. Object_Tracking
```
mot_tracker = Sort()      # Tracker using SORT Algorithm
for key in odata.keys():   
    arrlist = []
    det_img = cv2.imread(os.path.join(img_path, key))
    overlay = det_img.copy()
    det_result = data[key] 
    
    for info in det_result:
        bbox = info['bbox']
        labels = info['labels']
        scores = info['scores']
        templist = bbox+[scores]
        
        if labels == 1: # label 1 is a person in MS COCO Dataset
            arrlist.append(templist)
            
    track_bbs_ids = mot_tracker.update(np.array(arrlist))
    
    mot_imgid = key.replace('.jpg','')
    newname = save_path + mot_imgid + '_mot.jpg'
    print(mot_imgid)
    
    for j in range(track_bbs_ids.shape[0]):  
        ele = track_bbs_ids[j, :]
        x = int(ele[0])
        y = int(ele[1])
        x2 = int(ele[2])
        y2 = int(ele[3])
        track_label = str(int(ele[4])) 
        cv2.rectangle(det_img, (x, y), (x2, y2), (0, 255, 255), 4)
        cv2.putText(det_img, '#'+track_label, (x+5, y-10), 0,0.6,(0,255,255),thickness=2)
        
    cv2.imwrite(newname,det_img)
```
#### 4. product-detection

![image](https://user-images.githubusercontent.com/53899191/126057914-2837f0f9-af76-420e-81fe-dd3c6f3c6c6c.png)

```

# Install TFOD API (TF 1)
%tensorflow_version 1.x
import tensorflow as tf 
print(tf.__version__)

!git clone https://github.com/tensorflow/models.git

% cd models/research
!pip install --upgrade pip
# Compile protos.
!protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
!cp object_detection/packages/tf1/setup.py .
!python -m pip install --use-feature=2020-resolver .

```


