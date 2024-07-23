## ...Repository still under construction...

# Predicting the Best of the *N* Visual Trackers

 - This repository provides all codes and resources on our work on **Predicting the best of _N_ Trackers (*BofN*)**.

 <p align="center">
<a href="https://arxiv.org/abs/2407.15707"><img src="https://img.shields.io/badge/arXiv-Paper_Link-blue"></a>
</p>

![Structure of the Proposed BofN](/images/model.png)


## Abstract
We observe that the performance of SOTA visual trackers surprisingly strongly varies across different video attributes and
datasets. No single tracker remains the best performer across all tracking attributes and datasets. To bridge this gap, for a given video
sequence, we predict the “Best of the N Trackers”, called the BofN meta-tracker. At its core, a Tracking Performance Prediction
Network (TP2N) selects a predicted best performing visual tracker for the given video sequence using only a few initial frames. We also
introduce a frame-level BofN meta-tracker which keeps predicting best performer after regular temporal intervals. The TP2N is based
on self-supervised learning architectures MocoV2, SwAv, BT, and DINO; experiments show that the DINO with ViT-S as a backbone
performs the best. The video-level BofN meta-tracker outperforms, by a large margin, existing SOTA trackers on nine standard
benchmarks – LaSOT, TrackingNet, GOT-10K, VOT2019, VOT2021, VOT2022, UAV123, OTB100, and WebUAV-3M. Further
improvement is achieved by the frame-level BofN meta-tracker effectively handling variations in the tracking scenarios within long
sequences. For instance, on GOT-10k, BofN meta-tracker average overlap is 88.7% and 91.1% with video and frame-level settings
respectively. The best performing tracker, RTS, achieves 85.20% AO. On VOT2022, BofN expected average overlap is 67.88% and
70.98% with video and frame level settings, compared to the best performing ARTrack, 64.12%. This work also presents an extensive
evaluation of competitive tracking methods on all commonly used benchmarks, following their protocols.

## Methodology

Please find details in our paper which can be accessed [here](https://arxiv.org/abs/2407.15707).

### This work utilized the following trackers and others: 

[ARDiMP](https://github.com/MasterBin-IIAU/AlphaRefine) | [KeepTrack](https://github.com/visionml/pytracking) | [STMTrack](https://github.com/fzh0917/STMTrack) | [TransT](https://github.com/chenxin-dlut/TransT) | [ToMP](https://github.com/visionml/pytracking)
| [RTS](https://github.com/visionml/pytracking) | [SparseTT](https://github.com/fzh0917/SparseTT)

## Results

![Results](/images/plots.png)


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

## Citation
 
 If you find our work useful for your research, please consider citing:

```bibtex
@article{Alawode2024,
    archivePrefix = {arXiv},
    arxivId = {2407.15707},
    author = {Alawode, Basit and Javed, Sajid and Mahmood, Arif and Matas, Jiri},
    eprint = {2407.15707},
    number = {8},
    pages = {1--12},
    title = {{Predicting the Best of N Visual Trackers}},
    url = {http://arxiv.org/abs/2407.15707},
    volume = {14},
    year = {2024}
}
```
