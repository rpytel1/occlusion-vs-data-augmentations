## Setting up
### Setting up for OpenPose
1. install requirements
```$xslt
pip3 install -r requirements.txt
```
2. download model 
```$xslt
wget https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth
```

## Training
### Training for OpenPose 
Training consist of 3 steps:
- Training from MobileNet weights. (expected AP is 39%)
- Training from weights, obtained from previous step
- Training from weights obtained from previous step and increased number of refinement stages to 3 in network. 
1. Download pre-trained MobileNet v1 weights from [here](https://drive.google.com/file/d/18Ya27IAhILvBHqV_tDp0QjDFvsNNy-hv/view)
2. Convert train annotations in internal fromat. Run
``
python3 scripts/make_val_subset.py --labls <COCO_HOME>/annotations/person_keypoints_val2017.json
``
It will produce ***prepared_train_annotation.pkl*** with converted in internal fromat annotations. 
3. To train from MobileNet weights, run: 
``
python3 train.py --train-images-folder <COCO_HOME>/train2017/ --prepared-train-labels prepared_train_annotation.pkl --val-labels val_subset.json --val-images-folder <COCO_HOME>/val2017/ --checkpoint-path <path_to>/mobilenet_sgd_68.848.pth.tar --from-mobilenet
``
4. Next to train from previous checkpoint step run: 
```$xslt
python3 train.py --train-images-folder <COCO_HOME>/train2017/ --prepared-train-labels prepared_train_annotation.pkl --val-labels val_subset.json --val-images-folder <COCO_HOME>/val2017/ --checkpoint-path <path_to>/checkpoint_iter_280000.pth --weights-only --num-refinement-stages 3
```

## Validating
### Openpose validating
Run: 
```$xslt
python3 val.py --labels <COCO_HOME>/annotations/person_keypoints_val2017.json --images-folder <COCO_HOME>/val2017 --checkpoint-path <CHECKPOINT>
```

## Demo
 To check demo from webcam:
 ```$xslt
python3 demo.py --checkpoint-path <path_to>/checkpoint_iter_370000.pth --video 0
```