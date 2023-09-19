import atexit
import pandas as pd
import numpy as np
import random
import os
import cv2

import torch
import timm
import torch.nn as nn
import torch.functional as F
from torchvision import transforms
from torchsummary import summary
from torchview import draw_graph
from torchmetrics import Accuracy as perf_metric

import graphviz
graphviz.set_jupyter_format('png')

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from natsort import natsorted

from basit_codes.utils import seed_everything, set_requires_grad

seed_everything(42)
os.environ["CUDA_VISIBLE_DEVICES"]="1"

#DATASET_NAME = "LASOT_GOT10k_train"

LASOT_DATASET_PATH = f"testing_datasets/LASOT_train"
GOT_DATASET_PATH = f"testing_datasets/GOT10k_train"
TRACKINGNET_DATASET_PATH = f"testing_datasets/TrackingNet_train"
#SAVED_LASOT_FEAT_PATH = f"testing_datasets_features/LASOT"
#SAVED_GOT_FEAT_PATH = f"testing_datasets_features/GOT10k_train"
EXCEL_PATH = "all_train_LASOT_GOT10k_TrackingNet_new.xlsx"

got10k_lasot_names = ["GOT-10k", "airplane", "basketball", "bear", "bicycle",
                      "bird", "boat", "book", "bottle", "bus", "car", "cat",
                      "cattle", "chameleon", "coin", "crab", "crocodile",
                      "cup", "deer", "dog", "drone", "elephant", "electricfan",
                      "flag", "fox", "frog", "gametarget", "gecko", "giraffe",
                      "goldfish", "gorilla", "guitar", "hand", "hat", "helmet",
                      "hippo", "horse", "kangaroo", "kite", "leopard", "licenseplate",
                      "lion", "lizard", "microphone", "monkey", "motorcycle", "mouse",
                      "person", "pig", "pool", "rabbit", "racing", "robot", "rubicCube",
                      "sepia", "shark", "sheep", "skateboard", "spider", "squirrel",
                      "surfboard", "swing", "tank", "tiger", "train", "truck", "turtle",
                      "umbrella", "volleyball", "yoyo", "zebra"]

trackers_list = ["TransT", "ToMP", "RTS", "STMTrack", 
                     "ARDiMP", "SparseTT", "KeepTrack"]

train_test_ratio = 0.8
train_model = True    # Train model?
use_roi = True
use_class_weight = True
dont_test, aborted = False, False
backbone = 'vit'
opt_name = f"{backbone}_Opt_Cross_diff_reg_full_model"  # "Opt_Cross", "Opt_Cross_MAE", "Opt_MAE"
filepath = f"{os.getcwd()}/tracker_classifier/model_{opt_name}.pth"

# Model Parameters
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 300
stop_after = 10   # Stop if no improvement
print_every = 10

MAX_SEQ_LENGTH = 1
NUM_FEATURES = 2048
n_frames = MAX_SEQ_LENGTH
resize = (IMG_SIZE, IMG_SIZE)

# load Trackers results for the video sequences
trackers_results_read = pd.read_excel(EXCEL_PATH, usecols=["Video Name"].append(trackers_list))
video_names_all = list(trackers_results_read["Video Name"])
n_videos = len(video_names_all)
n_trackers = len(trackers_list)

tracker_results_pd = trackers_results_read[trackers_list]
trackers_results = tracker_results_pd.to_numpy(dtype=np.float32)

y_true_all = np.zeros((trackers_results.shape))
for i in range(y_true_all.shape[0]):
    vid_track_res = trackers_results[i,:]
    y_true_all[i, np.argmax(vid_track_res==np.max(vid_track_res))] = 1


# Obtain balanced video instances for training nd testing
#random.seed(0)

inds_list = [i for i in range(n_videos)]
train_inds = random.sample(inds_list, int(train_test_ratio*n_videos)) 
test_inds = list(np.setdiff1d(inds_list, train_inds))

class_weight = np.sum(y_true_all, axis=0)
class_weight /= np.sum(class_weight)
class_weight = 1/class_weight
class_weight /=np.sum(class_weight)
class_weight = np.around(10*class_weight, decimals=2)


# Get training and testing indices
random.shuffle(train_inds)
random.shuffle(test_inds)


# Get video frames
def get_frames(video_path):
    # Read all video frames
    all_frames = natsorted(os.listdir(video_path)) 
    frames_names = all_frames[:n_frames]

    # Read and return video frames
    frames = []
    for frame_name in frames_names:
        frame = cv2.imread(os.path.join(video_path, frame_name))
        shape = (frame.shape[0], frame.shape[1])
        frame = cv2.resize(frame, resize)
        frame = frame[:, :, [2, 1, 0]]
        frames.append(frame)

    return np.array(frames, dtype='float32'), shape

# Get rois
def get_roi_s(gt_path, roi_shape, delimiter=','):
    # Read all gt_bboxes
    with open(gt_path) as f:   # Read all
        gt_bboxes = f.readlines()
                
    # Read and return video frames
    roi_s = []
    for i in range(n_frames):
        bbox = (gt_bboxes[i].split('\n')[0]).split(delimiter)
        bbox = [int(x) for x in list(map(float, bbox))]
        roi = np.zeros((roi_shape))
        roi[bbox[0]:bbox[0]+bbox[2], bbox[1]:bbox[1]+bbox[3]] = 1.0
        roi = cv2.resize(roi, resize)
        roi_s.append(roi)

    return np.expand_dims(np.array(roi_s, dtype='float32'), axis=-1)

# Prepare Videos data
def get_data(indices, use_roi=False):
    n_batch = len(indices)

    # For each video.
    frames = np.zeros((n_batch, n_frames, IMG_SIZE, IMG_SIZE, 3), dtype='float32')
    roi_s = np.zeros((n_batch, n_frames, IMG_SIZE, IMG_SIZE, 1), dtype='float32')
    labels = []
    for idx in range(n_batch):
        video_name = video_names_all[indices[idx]]
        
        # Gather all its frames and add a batch dimension.
        if video_name.startswith(tuple(got10k_lasot_names)) and "-" in video_name:
            if "got" in video_name.lower():
                video_path = os.path.join(GOT_DATASET_PATH, video_name)  
                gt_path = f"{video_path}/groundtruth.txt"     
            else:
                video_path = os.path.join(LASOT_DATASET_PATH, video_name, "img")
                gt_path = f"{LASOT_DATASET_PATH}/{video_name}/groundtruth.txt"
        else:
            video_path = os.path.join(TRACKINGNET_DATASET_PATH, "zips", video_name)
            gt_path = f"{TRACKINGNET_DATASET_PATH}/anno/{video_name}.txt"
            
        curr_frames, shape = get_frames(video_path)
        curr_roi_s = get_roi_s(gt_path, shape)
        #curr_frames = curr_frames[None, ...]
        
        curr_labels = [y_true_all[indices[idx]], trackers_results[indices[idx]]]
        
        frames[idx, :] = curr_frames
        roi_s[idx, :] = curr_roi_s
        labels.append(curr_labels)

    if not use_roi:
        return frames, np.array(labels, dtype='float32'), None
    else: 
        return frames, np.array(labels, dtype='float32'), roi_s

class Predictor(nn.Module):
    def __init__(self, num_classes=7) -> None:
        super(Predictor, self).__init__()
        
        self.transforms = transforms.Compose([
            transforms.Lambda(lambda t: t/255),
            transforms.Normalize((0.485, 0.456, 0.406), 
                                 (0.229, 0.224, 0.225))])
        
        self.backbone = timm.create_model('vit_small_patch16_224', 
                                          pretrained=True, num_classes=0)
        self.attention_map = nn.Conv2d(1, 384, kernel_size=(16, 16), 
                                       stride=(16, 16))
        self.flatten = nn.Flatten(start_dim=1, end_dim=2)
        
        # Classification heads
        #self.dense_1 = nn.Linear(self.backbone.embed_dim, 128)
        #self.act_1 = nn.ReLU()
        #self.dense_2 = nn.Linear(128, 16)
        #self.act_2 = nn.ReLU()
        
        #self.dense_c1 = nn.Linear(16, num_classes)
        #self.act_c1 = nn.Softmax(dim=-1)
        
        #self.dense_r1 = nn.Linear(16, num_classes)
        #self.act_r1 = nn.Sigmoid()
        
        self.dense_c1 = nn.Linear(self.backbone.embed_dim, num_classes)
        self.act_c1 = nn.Softmax(dim=-1)
        
        self.dense_r1 = nn.Linear(self.backbone.embed_dim, num_classes)
        self.act_r1 = nn.Sigmoid()
        
    
    def forward(self, x, roi_mask=None):
        y = self.transforms(x)
        
        y = self.backbone.patch_embed.proj(y)
        
        if roi_mask is not None:      # Apply roi map
            roi = torch.unsqueeze(torch.FloatTensor(roi_mask),
                                  axis=1).to("cuda")
            roi = self.attention_map(roi)
            y = y + roi 
        
        y = self.backbone.patch_embed.norm(y)
        y = self.backbone.pos_drop(y)
        y = self.backbone.patch_drop(y)
        y = self.backbone.norm_pre(y)
        
        y = y.permute((0, 2, 3, 1))
        y = self.flatten(y)
        
        y = self.backbone.blocks(y)
        y = self.backbone.norm(y)
        y = self.backbone.fc_norm(y)
        y = self.backbone.head_drop(y)
        #y = self.backbone.head(y)
        y = y[:, 0, :]
        
        #y = self.dense_1(y)
        #y = self.act_1(y)
        #y = self.dense_2(y)
        #y = self.act_2(y)
        
        y1 = self.dense_c1(y)
        class_prob = self.act_c1(y1)
        
        y2 = self.dense_r1(y)
        reg_val = self.act_r1(y2)
        
        return [class_prob, reg_val]
        
class CustomLoss(nn.Module):
    def __init__(self, class_weight=None):
        super(CustomLoss, self).__init__()
        
        self.class_weight = class_weight.to('cuda')
        self.crossentropy = nn.CrossEntropyLoss(weight=self.class_weight)
        self.mae = nn.L1Loss(reduction='none')

    def forward(self, outputs, labels):
        true_class, true_val = labels[:, 0, :], labels[:, 1, :]
        pred_proba, pred_val = outputs[0], outputs[1]
        loss_class = self.crossentropy(pred_proba, true_class)
        
        loss_diff_reg =  self.mae(pred_val, true_val)
        loss_diff_reg = torch.mean(loss_diff_reg, dim=0)
        
        if self.class_weight is not None:
            loss_diff_reg *= self.class_weight
        
        loss_diff_reg = torch.mean(loss_diff_reg) 
        
        a = torch.max(pred_proba, dim=-1).values
        a = torch.unsqueeze(a, -1)
        a = a.repeat(1, pred_proba.shape[1])
        pred_class = torch.as_tensor(pred_proba == a, dtype=torch.float)
        
        loss = loss_class if torch.sum(torch.abs(true_class - pred_class)) == 0 else \
                loss_class + loss_diff_reg

        #weight = tf.convert_to_tensor(class_weight, dtype=tf.float32)
        #loss = loss*weight
        return torch.sum(loss)
    

# Train and test the Classifier
def train():
    model = Predictor(num_classes=n_trackers)
    
    #print(summary(model, (n_frames, 3, IMG_SIZE, IMG_SIZE)))
    #model_graph = draw_graph(model, 
    #                         input_size=(n_frames, 3, IMG_SIZE, IMG_SIZE),
    #                         expand_nested=False)
    #model_graph.visual_graph
    
    #keras.utils.plot_model(model,to_file='model_vit.png')
    
    total_params = set_requires_grad(model, requires_grad=True)
    backbone_params = set_requires_grad(model.backbone, requires_grad=False)
    
    print(f"\n\nTotal Params = {total_params:,}")
    print(f"Trainable Params = {(total_params - backbone_params):,}")
    print(f"Non Trainable Params = {backbone_params:,}\n\n")
    
    if torch.cuda.is_available():
        model.to(torch.device("cuda"))
    optimizer = torch.optim.Adam(model.parameters()) 
    loss_fn = CustomLoss(class_weight=torch.tensor(class_weight)) if \
        use_class_weight else CustomLoss()
    acc_metric = perf_metric(task="multiclass", num_classes=n_trackers).to('cuda')
    
    
    #validation_split=0.3
    
    train_losses, train_acc, best_loss, best_acc, best_epoch = [], [], 100000, 0, 0
    dont_test, aborted = False, False
    
    try:
        for e in range(1, EPOCHS+1):
            print("===========================")
            print("\nStart of epoch %d of %d" % (e, EPOCHS))
            
            n_step = int(len(train_inds)/BATCH_SIZE)
            
            epoch_losses = []
            epoch_acc = []
            for i in range(n_step):
                train_images, train_labels, roi_s = get_data(train_inds[i*BATCH_SIZE: (i+1)*BATCH_SIZE],
                                                             use_roi=use_roi)
                
                train_images = torch.tensor(train_images)
                train_images = torch.permute(train_images, [0, 1, 4, 3, 2])
                train_images = torch.squeeze(train_images)
                train_images = train_images.to("cuda")
                
                train_labels = torch.tensor(train_labels).to('cuda')
                
                optimizer.zero_grad()
                outputs = model(train_images, np.squeeze(roi_s))
                loss = loss_fn(outputs, train_labels)
                loss.backward()
                optimizer.step()
                
                # Update current epoch training loss value
                loss = loss.cpu().detach().numpy()
                epoch_losses.append(loss)
                
                # Compute current epoch trainig accuracy value
                pred_proba = outputs[0]
                a = torch.max(pred_proba, dim=-1).values
                a = torch.unsqueeze(a, -1)
                a = a.repeat(1, pred_proba.shape[1])
                pred_class = torch.as_tensor(pred_proba == a, dtype=torch.float)
                acc = acc_metric(train_labels[:, 0, :], pred_class)
                acc = acc.cpu().detach().numpy()
                epoch_acc.append(acc)
                
                if (i%print_every==0) or ((i+1) == n_step):
                    print(f"   Epoch: {e+1}/{EPOCHS}, Step {i+1}/{n_step} >> \t Loss: {loss:.4f}, \t Acc: {acc:.4f}")
                
            # Update training loss and accuracy values   
            e_loss, e_acc =  np.mean(np.array(epoch_losses)),  np.mean(np.array(epoch_acc))
            train_losses.append(e_loss)
            train_acc.append(e_acc)
            
            print("")
            print(f"Epoch: {e}/{EPOCHS} >> \t Avg Loss: {e_loss:.4f}, \t Acc: {e_acc:.4f}")

            # Save the trained model
            if e_acc >= best_acc:
                best_acc, best_loss, best_epoch = e_acc, e_loss, e
                torch.save(model.state_dict(), filepath)
            print(f"\n {backbone} >> Best Acc: {best_acc:.4f}, at epoch {best_epoch}, Loss of {best_loss:.4f}.")

            # Early stopping
            if e > best_epoch + stop_after: 
                print("Early Stopping Activated.")
                break
            
    except KeyboardInterrupt:
        aborted = True
        print("Training Aborted")
    except Exception as e:
        print(f"Exception name: {e} occured.")
    finally:
        plt.figure()
        plt.plot(np.array(train_losses))
        plt.title('Model Training Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.savefig(f"{backbone}_classifier_train_loss.png")
        
        plt.figure()
        plt.plot(np.array(train_acc))
        plt.title('Model Training Accuracy')
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.savefig(f"{backbone}_classifier_train_acc.png")
        
        if not aborted: print("Training Completed")
        return dont_test

def test_single(model, test_images, roi):
    outputs = model(test_images, roi)
        
    predict = outputs[0]
    y_pred = torch.zeros_like(predict)
    max_val = torch.max(predict, dim=-1).values
    max_val = torch.unsqueeze(max_val, -1)
    max_val = max_val.repeat(1, predict.shape[1])
    
    y_pred[predict==max_val] = 1.0
    
    # If more than one tracker is outputed, randomly pick one
    sums = torch.sum(y_pred, dim=-1)
    mult_inds = torch.nonzero(sums-1.0, as_tuple=True)
    for mult_ind in mult_inds:
        t_inds = torch.nonzero(y_pred[mult_ind, :], as_tuple=True)
        y_pred[mult_ind, :] = 0.0
        perm = (torch.randperm(len(t_inds)))[:1]
        y_pred[mult_ind,  t_inds[perm]] = 1.0
    
    return y_pred   

def test(batch_size_test=16):
    print("Testing Model....")
    model = Predictor(num_classes=n_trackers)
    model.load_state_dict(torch.load(filepath))
    if torch.cuda.is_available():
        model.to(torch.device("cuda"))
    
    perf = perf_metric(task="multiclass", num_classes=n_trackers).to('cuda')
    pred_labels, test_labels, acc = [], [], []
    
    # Divide test set into n chunks
    from math import ceil
    def chunk_into_n(lst, n):
        size = ceil(len(lst) / n)
        return list(
            map(lambda x: lst[x * size:x * size + size],
            list(range(n)))
        )
    
    n = int(len(test_inds)/ batch_size_test) + 1
    test_inds_chunk = chunk_into_n(test_inds, n)
    
    for inds in test_inds_chunk:
        test_images, labels, roi_s = get_data(inds, use_roi=use_roi)
        
        test_images = torch.squeeze(
                        torch.permute(
                        torch.tensor(test_images), 
                                    [0, 1, 4, 3, 2])).to('cuda')
        y_pred = test_single(model, test_images, 
                             np.squeeze(roi_s))
    
        labels = torch.tensor(labels).to('cuda')
        acc_batch = perf(labels[:,0,:], y_pred)
        pred_labels.append(y_pred.cpu().detach().numpy())
        test_labels.append(labels[:,0,:].cpu().detach().numpy())

        acc.append(acc_batch.cpu().detach().numpy())
        
    acc = np.mean(np.array(acc))
    print(f"Test Accuracy = {acc:.4f}")
    
    # Compute confusion matrix
    test_labels = np.concatenate(test_labels, axis=0, dtype=np.float32)
    pred_labels = np.concatenate(pred_labels, axis=0, dtype=np.float32)
    #cf_mat = multilabel_confusion_matrix(test_labels, pred_labels)
    t = np.argmax(test_labels, axis=1)
    p = np.argmax(pred_labels, axis=1)
    cf_mat = confusion_matrix(t, p)
    print(f"{backbone} Confusion Matrix on Test Data:")
    print(cf_mat)
    
    print(classification_report(test_labels, pred_labels, target_names=trackers_list))
    #ConfusionMatrixDisplay.from_predictions(test_labels, pred_labels, display_labels=new_trackers_list,
    #                                        xticks_rotation="vertical")
    ConfusionMatrixDisplay.from_predictions(t, p, display_labels=trackers_list, xticks_rotation="vertical")
    plt.tight_layout()
    plt.savefig(f"{backbone}_test_confusion_matrix2.png")

if __name__ == "__main__":
    if train_model:
        #Train Model
        dont_test = train()

    if not dont_test:
        test()