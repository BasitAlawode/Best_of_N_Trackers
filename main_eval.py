from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import warnings

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from sympy import true
warnings.filterwarnings("ignore")

import os
import numpy as np

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
import json
import torch

from toolkit.evaluation import OPEBenchmark

#from toolkit.evaluation import AccuracyRobustnessBenchmark
#from toolkit.evaluation import EAOBenchmark
from basit_codes.ar_benchmark import AccuracyRobustnessBenchmark
from basit_codes.eao_benchmark import EAOBenchmark

#from toolkit.visualization import draw_success_precision
from basit_codes.utb import UTBDataset
from basit_codes.vot import VOTDataset
from basit_codes.create_json import *
from basit_codes.track_video import track_video
from basit_codes.draw_success_precision import draw_success_precision

base_dir = os.getcwd()

# "vot2018_SP32", "vot2018_SP64" fail as some frames are already empty
# and/or target is already zeroed out with 0 width and 0 height
vot_yr = 2021

# Temporal Subsampling (NB: TP_128 is essentially first and last frame)
#dataset_names = [f"vot{vot_yr}", f"vot{vot_yr}_TP2", f"vot{vot_yr}_TP4", f"vot{vot_yr}_TP8", \
#                 f"vot{vot_yr}_TP16", f"vot{vot_yr}_TP32", f"vot{vot_yr}_TP64", f"vot{vot_yr}_TP128"]

#dataset_names = [f"vot{vot_yr}_BKWD", f"vot{vot_yr}_TP2_BKWD", f"vot{vot_yr}_TP4_BKWD", f"vot{vot_yr}_TP8_BKWD", \
#                 f"vot{vot_yr}_TP16_BKWD", f"vot{vot_yr}_TP32_BKWD", f"vot{vot_yr}_TP64_BKWD", f"vot{vot_yr}_TP128_BKWD"]

# Spatial Subsampling
#dataset_names = [f"vot{vot_yr}", f"vot{vot_yr}_SP2", f"vot{vot_yr}_SP4", f"vot{vot_yr}_SP8", \
#                 f"vot{vot_yr}_SP16", f"vot{vot_yr}_SP32", f"vot{vot_yr}_SP64"]

#dataset_names = [f"vot{vot_yr}", f"vot{vot_yr}_SP2_GAUSS", f"vot{vot_yr}_SP4_GAUSS", f"vot{vot_yr}_SP8_GAUSS", \
#                 f"vot{vot_yr}_SP16_GAUSS", f"vot{vot_yr}_SP32_GAUSS", f"vot{vot_yr}_SP64_GAUSS"]

# Spatial Upsampling 
#dataset_names = [f"vot{vot_yr}", f"vot{vot_yr}_SP2_UP", f"vot{vot_yr}_SP4_UP", f"vot{vot_yr}_SP8_UP", \
#                 f"vot{vot_yr}_SP16_UP", f"vot{vot_yr}_SP32_UP", f"vot{vot_yr}_SP64_UP"]

# Accuracy Investigation: Performance Ordering for Tracking (Using TransT Tracker result)
#dataset_names = [f"vot{vot_yr}", f"vot{vot_yr}_ACC_LOW", f"vot{vot_yr}_ACC_HIGH", f"vot{vot_yr}_ACC_LL", \
#                 f"vot{vot_yr}_ACC_LH", f"vot{vot_yr}_ACC_HL", f"vot{vot_yr}_ACC_HH"]

# Effect of Target Object Area (A) Size on Tracking
#dataset_names = [f"vot{vot_yr}", f"vot{vot_yr}_A_1", f"vot{vot_yr}_A_1_2", \
#                 f"vot{vot_yr}_A_2_3", f"vot{vot_yr}_A_3_5", f"vot{vot_yr}_A_5_8", \
#                 f"vot{vot_yr}_A_8_15", f"vot{vot_yr}_A_15_30", f"vot{vot_yr}_A_30"]

# Similar/Disimilar object tracking
#dataset_names = [f"vot{vot_yr}", f"vot{vot_yr}_OBJ_SIM", f"vot{vot_yr}_OBJ_DSIM"]


datasets_dir = "testing_datasets" # Testing Datasets dir
trackers_results_dir = "trackers_results"  # Tracker results path 

# All trackers whose results are available (optimized)
trackers = ["TransT", "ToMP", "RTS", "STMTrack", "ARDiMP", "SparseTT", 
            "KeepTrack", "Vit_Opt", "ResNet_Opt"]     
# Used for generating result for a specific tracker
optim_trackers = ["TransT", "ToMP", "RTS", "STMTrack", 
                  "ARDiMP", "SparseTT", "KeepTrack"]  
bold_names = ["Vit_Opt", "ResNet_Opt"]  # For Legend bolding

# Combined dataset used for generating training data

#dataset_names, combined_dataset, par_compute = ["LASOT_train_GOT10k_train"], True, True
#compute_ope, compute_a_r_eao, save_excel, trackers = False, True, True, optim_trackers

#dataset_names, combined_dataset, par_compute = ["TrackingNet_train"], False, False
#compute_ope, compute_a_r_eao, save_excel, trackers = False, True, True, optim_trackers

#dataset_names, combined_dataset, par_compute = ["LASOT_train_GOT10k_train_TrackingNet_train"], True, False
#compute_ope, compute_a_r_eao, save_excel, trackers = False, False, False, optim_trackers

#dataset_names, combined_dataset, par_compute = [f"vot{vot_yr}"], False, True
#compute_ope, compute_a_r_eao, save_excel = False, True, False

dataset_names, combined_dataset, par_compute = ["LASOT_test", "GOT10k_val",
                                                "TrackingNet_val"], False, True
compute_ope, compute_a_r_eao, save_excel = True, False, False

#dataset_names, combined_dataset, par_compute = ["GOT10k_test", "TrackingNet_test"], False, True
#compute_ope, compute_a_r_eao, save_excel = False, False, False


show_video_level = False
plot_success_precision = True

def main(base_dir, datasets_dir, dataset_name, trackers_results_dir, trackers, num): 
        assert len(trackers) > 0
        num = min(num, len(trackers))

        trackers_results_path = os.path.join(base_dir, trackers_results_dir, dataset_name)
        dataset_root = os.path.join(base_dir, datasets_dir, dataset_name)

        # Create JSON file for the dataset if it does not exist
        if not os.path.exists(os.path.join(dataset_root, f"{dataset_name}.json")):
                print('Dataset JSON does not exit... Attempting to create one.')

                d_lower = dataset_name.lower()
                if d_lower.startswith("vot"):
                        mask_annotated = True if dataset_name.endswith("2021") else False  
                        mask_annotated = True if "vot2021_ACC" in dataset_name else mask_annotated  
                        mask_annotated = True if "vot2021_A_" in dataset_name else mask_annotated 
                        mask_annotated = True if "vot2021_OBJ_" in dataset_name else mask_annotated  
                        create_json_vot(f"{dataset_root}/*/", dataset_name, gt_file_name="groundtruth.txt", \
                                convert_region=True, mask_annotated=mask_annotated, delimiter=',', \
                                frames_folder_name="color")
                elif "got10k" in d_lower and "lasot" in d_lower and "trackingnet" not in d_lower:
                        dir_got = os.path.join(base_dir, datasets_dir, "GOT10k_train")
                        dir_lasot = os.path.join(base_dir, datasets_dir, "LASOT_train")
                        json_dir = os.path.join(base_dir, datasets_dir, dataset_name)
                        os.makedirs(json_dir, exist_ok=True)
                        create_got_lasot_json(f"{dir_got}/*/", f"{dir_lasot}/*/", json_dir, dataset_name)
                elif d_lower.startswith("got"):
                        create_json_got(f"{dataset_root}/*/", dataset_name)
                elif d_lower.startswith("lasot"):
                        create_json_lasot(f"{dataset_root}/*/", dataset_name)
                elif d_lower.startswith("trackingnet"):
                        create_json_trackingNet(dataset_root, dataset_name)
                print("JSON created and saved.")

        #Check if trackers results have been obtained, otherwise obtain it.
        with open(os.path.join(dataset_root, dataset_name+".json"), "r") as f:
            dataset_json = json.load(f)
        
        
        for tracker in trackers:
                if tracker.endswith("_Opt"):
                        # Get the trained model tracker predictor
                        
                        if not tracker.startswith('Vit'):
                                import classifier_all_data_1 as pred_file
                        else:
                                import classifier_all_data_2 as pred_file
                        predictor = pred_file.Predictor()
                        predictor.load_state_dict(torch.load(pred_file.filepath))
                        
                        if torch.cuda.is_available():
                                predictor.to(torch.device("cuda"))
                        
                if not os.path.exists(os.path.join(trackers_results_path, tracker)):
                        os.makedirs(os.path.join(trackers_results_path, tracker))
                        
                for i, video_name in enumerate(dataset_json.keys()):
                        pred_bbox_path = os.path.join(trackers_results_path, tracker, f"{video_name}.txt")
                        if not os.path.exists(pred_bbox_path):
                                #Run tracker for the video and save it in tracker result directory
                                print(f'{tracker} tracker results for {video_name} ({i+1}/{len(dataset_json.keys())}) of {dataset_name} does not exist')
                                print('...Running tracker on the video frames now... Please wait...')
                                video_details = dataset_json[video_name]
                                
                                # Obtain list of bounding boxes

                                if not tracker.endswith("_Opt"):
                                        pred_tracker = tracker
                                else:
                                        # Read video frames
                                        vid_path = os.path.join(base_dir, datasets_dir, dataset_name, video_name)
                                        gt_path = f"{vid_path}/groundtruth.txt" 
                                        
                                        if dataset_name.lower().startswith("vot"):
                                                vid_path = os.path.join(vid_path, "color")
                                        elif dataset_name.lower().startswith("lasot"):
                                                vid_path = os.path.join(vid_path, "img")
                                        elif dataset_name.lower().startswith("trackingnet"):
                                                vid_path = os.path.join(base_dir, datasets_dir, 
                                                                        dataset_name, "zips", video_name)
                                                gt_path = os.path.join(base_dir, datasets_dir, 
                                                                        dataset_name, "anno", f"{video_name}.txt")

                                        frame, shape = pred_file.get_frames(vid_path)
                                        frame = torch.permute(torch.tensor(frame), 
                                                        [0, 3, 2, 1]).to('cuda')
                                        roi = None
                                        if pred_file.use_roi:
                                                roi = np.squeeze(pred_file.get_roi_s(gt_path, shape),
                                                                     axis=-1) 
                                        
                                        # Predict the tracker for the video
                                        predict = pred_file.test_single(predictor, frame)
                                        predict = np.squeeze(predict.cpu().detach().numpy())
                                        pred_tracker = optim_trackers[int(np.argwhere(predict==1))]
                                
                                        print(f"\nTracking with: {pred_tracker} \n")
                                        
                                pred_bboxes = track_video(video_details, pred_tracker, base_dir, dataset_name, 
                                                        full_img_path=combined_dataset)
                                with open(pred_bbox_path, "w") as f:    # Save bounding boxes
                                        for pred_bbox in pred_bboxes:
                                                f.write(f"{pred_bbox[0]}\t{pred_bbox[1]}\t{pred_bbox[2]}\t{pred_bbox[3]}\n")
                                print('.....Done.')


        if compute_a_r_eao:
                # A, R, EAO Evaluation                
                # Create dataset and set the trackers
                dataset = VOTDataset(dataset_name, dataset_root, load_img=False)
                dataset.set_tracker(trackers_results_path, trackers)

                benchmark_ar = AccuracyRobustnessBenchmark(dataset)
                benchmark_eao = EAOBenchmark(dataset)
                
                ac_rob_ret = {} 
                eao_rob_ret = {}  
                if par_compute:
                               
                        with Pool(processes=num) as pool:
                                for ret in tqdm(pool.imap_unordered(benchmark_ar.eval,
                                trackers), desc='Accuracy Robustness', total=len(trackers), ncols=100):
                                        ac_rob_ret.update(ret)
                                        
                        with Pool(processes=num) as pool:
                                for ret in tqdm(pool.imap_unordered(benchmark_eao.eval,
                                trackers), desc='EAO', total=len(trackers), ncols=100):
                                        eao_rob_ret.update(ret)
                else:
                        print("Computing Accuracy and Robustness")
                        ac_rob_ret = benchmark_ar.eval(trackers)
                        print(".....Done")
                        
                        #print("Computing EAO")
                        #eao_rob_ret = benchmark_eao.eval(trackers)
                        #print(".....Done")
                
                if not par_compute:
                        benchmark_ar.show_result(ac_rob_ret, eao_result=eao_rob_ret, show_video_level=show_video_level)
                
                # Save results in excel file
                if save_excel:
                        benchmark_ar.save_to_excel(ac_rob_ret, f"{dataset_name}_result.xlsx")


        if compute_ope:
                # OPE Evaluation
                dataset = UTBDataset(dataset_name, dataset_root, load_img=False)
                dataset.set_tracker(trackers_results_path, trackers)
                benchmark = OPEBenchmark(dataset)

                success_ret, precision_ret, norm_precision_ret = {}, {}, {} 
                
                if par_compute:       
                        # Success evaluation
                        with Pool(processes=num) as pool:
                                for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                                trackers), desc='eval success', total=len(trackers), ncols=100):
                                        success_ret.update(ret)

                        # Precision evaluation
                        with Pool(processes=num) as pool:
                                for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                                trackers), desc='eval precision', total=len(trackers), ncols=100):
                                        precision_ret.update(ret)
                        
                        # Norm precision evaluation
                        with Pool(processes=num) as pool:
                                for ret in tqdm(pool.imap_unordered(benchmark.eval_norm_precision,
                                trackers), desc='eval precision', total=len(trackers), ncols=100):
                                        norm_precision_ret.update(ret)
                else:
                        print("Computing Success...")
                        success_ret = benchmark.eval_success(trackers)
                        print("Computing Precision...")
                        precision_ret = benchmark.eval_precision(trackers)
                        print("Computing Normalized Precision...")
                        norm_precision_ret = benchmark.eval_norm_precision(trackers)

                # Show results
                benchmark.show_result(success_ret, precision_ret, norm_precision_ret,
                                show_video_level=show_video_level)

                # Plottings
                if not os.path.exists(os.path.join(trackers_results_path, "plots")):
                        os.makedirs(os.path.join(trackers_results_path, "plots"))
                if plot_success_precision:
                        videos = [k for k in dataset_json.keys()]
                        draw_success_precision(success_ret, dataset_name, videos, 
                                               'ALL', precision_ret=precision_ret, 
                                               norm_precision_ret=norm_precision_ret,
                                               bold_name=bold_names)

        print('Completed....')


if __name__ == "__main__":
        for dataset_name in dataset_names:
                print(f'\n\n Results for Dataset {dataset_name}\n\n')
                main(base_dir, datasets_dir, dataset_name, trackers_results_dir, \
                        trackers, 1)        