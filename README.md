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
- Got10k Train set
- LaSOT Train set
- TrackingNet Train set

2. Download the datasets above and put them in the testing_datasets folder.For ease of training, some of the video folders have been renamed, especially for the TrackingNet dataset (see the excel file below).

3. The excel file [all_train_LASOT_GOT10k_TrackingNet_new.xlsx](./all_train_LASOT_GOT10k_TrackingNet_new.xlsx) contains the tracking success rate of the trackers on the videos in the dataset. This result is used to generate the output of the predictor where the tracker with the highest success rate is 1 and the others, 0 for each video. 

4. Training Predictor With ResNet Backbone
```bash
python classifier_all_data_1.py
```

5. Training Predictor With Vision Transformer (ViT) Backbone
```bash
python classifier_all_data_2.py
```

## Testing/Tracking with the Predicted Tracker

1. To track, first download the pre-trained models of the 7 base trackers from the below links and put them in the trained trackers folder.
    1. [ARDiMP](https://kuacae-my.sharepoint.com/:f:/g/personal/100060517_ku_ac_ae/Er8rDSdhy31Nr9Nf076gqV4Bj6-RU8dO4aRqUbYmVihdhg?e=Jp5V4G)
    2. [KeepTrack](https://kuacae-my.sharepoint.com/:f:/g/personal/100060517_ku_ac_ae/EkhVxPTBgAtNpL-TRlmojgEB6OzayO_E2K0EpbOIzd2kEg?e=EAC8A7)
    3. [STMTrack](https://kuacae-my.sharepoint.com/:f:/g/personal/100060517_ku_ac_ae/Ei7KPct5H4xPjeCUhx-Zi9ABSosQLkxpyKHZKVA7QTzvog?e=3XZL8M)
    4. [TransT](https://kuacae-my.sharepoint.com/:f:/g/personal/100060517_ku_ac_ae/EqJkM3jV2YxEnbns-ADjOVMB7qkQ3K1nNUAf4rab3thHTg?e=ntuPTP)
    5. [ToMP](https://kuacae-my.sharepoint.com/:f:/g/personal/100060517_ku_ac_ae/EkhVxPTBgAtNpL-TRlmojgEB6OzayO_E2K0EpbOIzd2kEg?e=EAC8A7)
    6. [RTS](https://kuacae-my.sharepoint.com/:f:/g/personal/100060517_ku_ac_ae/EkhVxPTBgAtNpL-TRlmojgEB6OzayO_E2K0EpbOIzd2kEg?e=EAC8A7)
    7. [SparseTT](https://kuacae-my.sharepoint.com/:f:/g/personal/100060517_ku_ac_ae/Ev5MSxYfr1dKmVjytHiCofYBQl2TUg635A0KnrGXsbcmyA?e=zjY199)

2. To track, simply run the main_eval.py file. Tracking results will be found in the tracker_results folder.

```bash
python main_eval.py
```

NOTE: This will run the main trackers and also run the best of them on the videos using both ResNet and ViT backbones.
 
