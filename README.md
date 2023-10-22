-- Repository still under construction...

# Best of the N-Trackers

This work predicts the best tracker from a list of N available trackers that can best be used to track a given object in a sequence.

Available Trackers: 

1. [ARDiMP](https://github.com/MasterBin-IIAU/AlphaRefine)
2. [KeepTrack](https://github.com/visionml/pytracking)
3. [STMTrack](https://github.com/fzh0917/STMTrack)
4. [TransT](https://github.com/chenxin-dlut/TransT)
5. [ToMP](https://github.com/visionml/pytracking)
6. [RTS](https://github.com/visionml/pytracking)
7. [SparseTT](https://github.com/fzh0917/SparseTT)

## Environment Setup

1. Create the python environment

```bash
conda create -y --name n_trackers python==3.7.16
conda activate n_trackers  
``` 

2. Install pytorch and torchvision
```bash
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

3. Install other packages

```bash
pip install -r requirements.txt
```

## Training the Tracker Predictor

1. The following datasets has been utilized to train the tracker predictor
- TrackingNet Train set
- LaSOT Train set
- Got10k Train set

2. Training Predictor With ResNet Backbone
```bash
python classifier_all_data_1.py
```

3. Training Predictor With Vision Transformer (ViT) Backbone
```bash
python classifier_all_data_2.py
```

## Tracking with the Predicted Tracker
```bash
python main_eval.py
```

## Tracking with the Other Trackers

```bash
python main_eval.py
```
 
