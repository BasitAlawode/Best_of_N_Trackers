import os
from glob import glob
import numpy as np
from natsort import natsorted
import cv2
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from pysot.utils.bbox import get_axis_aligned_bbox, cxy_wh_2_rect  
from basit_codes.utb import UTBDataset
from toolkit.evaluation import OPEBenchmark

class VideoAttributes:
    def __init__(self, frames_dir, anno_path, anno_delim = ',', frame_level=False, bbox_type='xywh'):
        '''Computes the Attributes of a Video (According to the GoT-10k Paper)'''
        
        # Read all frame paths
        self.frame_paths = [pth for pth in natsorted(glob(f"{frames_dir}/*", recursive=True)) if pth.endswith(('.png', '.jpg'))]
        self.gt_rects = []
        self.frame_level = frame_level
        
        # Read all annotations
        with open(anno_path) as f:  
            gt_bboxes = f.readlines()
        
        for gt_bb in gt_bboxes:
            gt_bb = (gt_bb.split('\n')[0]).split(anno_delim)
            gt_bb = [int(x) for x in list(map(float, gt_bb))]
            
            # Convert bbox to xywh type
            if bbox_type != 'xywh':
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bb, dtype=np.float32))
                gt_bb = cxy_wh_2_rect((cx, cy,), (w, h))
                gt_bb = [int(gt_bb[0]), int(gt_bb[1]), int(gt_bb[2]), int(gt_bb[3])]
            self.gt_rects.append(gt_bb)
        
        # Ensure number of frames == number of bboxes
        assert len(self.frame_paths) == len(self.gt_rects) 
         
    def occlusion(self, visible_ratio=0.1, partial=False, return_bool=True):
        '''Full and Partial Occlusion Attributes'''
        if not partial:
            # Full Occlusion
            full_occl = [1 if rect[2]*rect[3] == 0 else 0 for rect in self.gt_rects]  
            attr_result = self.__return_attr(full_occl, "FO", return_bool=return_bool)
            return attr_result
        else:
            # Partial Occlusion
            raise NotImplementedError
    
    def scale_variation(self, T_past=5, variation_threshold=1.0, return_bool=True):
        '''Scale Variation Attribute'''
        scales = [np.sqrt(rect[2]*rect[3]) for rect in self.gt_rects]
        
        scales_T = [scales[i] if i < T_past else scales[i-T_past] for i in range(len(scales))]
                
        scale_var = [max(x/(y+1e-16), y/(x+1e-16)) for (x,y) in zip(scales, scales_T)]
        
        attr_result = self.__return_attr(scale_var, "SV", return_bool=return_bool, 
                           compare_val=variation_threshold, compare_type='>')
        return attr_result
    
    def deformation(self, T_past=5, deformation_threshold=1.0, return_bool=True):
        '''Deformation Attribute'''
        ratios = [np.sqrt(rect[3]/(rect[2]+1e-16)) for rect in self.gt_rects]
        
        ratios_T = [ratios[i] if i < T_past else ratios[i-T_past] for i in range(len(ratios))]
                
        deform = [max(x/(y+1e-16), y/(x+1e-16)) for (x,y) in zip(ratios, ratios_T)]
        
        attr_result = self.__return_attr(deform, "DF", return_bool=return_bool, 
                           compare_val=deformation_threshold, compare_type='>')
        
        return attr_result
    
    def fast_motion(self, motion_threshold=0.005, return_bool=True):
        '''Fast Motion Attribute'''
        scales = [np.sqrt(rect[2]*rect[3]) for rect in self.gt_rects]
        centres = [[(x+w)/2, (y+h)/2] for [x, y, w, h] in self.gt_rects]
        
        motion = [np.linalg.norm(np.array(centres[i]) - np.array(centres[i-1]))/(scales[i]*scales[i-1]+1e-16) 
                  if i > 0 else 0 for i in range(len(scales))]
        
        attr_result = self.__return_attr(motion, "FM", return_bool=return_bool, 
                               compare_val=motion_threshold, compare_type='>')
        
        return attr_result
    
    def illum_variation(self, illum_threshold=0.0025, return_bool=True):
        '''Illumination Variation Attribute'''
        
        # Read all frames
        frames_avg = [np.mean(cv2.imread(pth)/255, (0,1)) for pth in self.frame_paths]
            
        illum_var = [np.linalg.norm(frames_avg[i] - frames_avg[i-1], 1) if i > 0 else 0 
                    for i in range(len(frames_avg))]
        
        attr_result = self.__return_attr(illum_var, "IV", return_bool=return_bool, 
                           compare_val=illum_threshold, compare_type='>')
        return attr_result
    
    def low_resolution(self, res_threshold=0.2, return_bool=True):
        '''Low Resolution Attribute'''
        scales = [np.sqrt(rect[2]*rect[3]) for rect in self.gt_rects]
        s_median = np.median(scales)
        
        low_res = [x/(s_median+1e-16) for x in scales]
        
        attr_result = self.__return_attr(low_res, "LR", return_bool=return_bool, 
                           compare_val=res_threshold, compare_type='<')
        
        return attr_result
    
    def __return_attr(self, attr, attr_name, return_bool=True, compare_val=1, compare_type='='):
        if return_bool:
            if compare_type=='<':
                attr = [1 if a < compare_val else 0 for a in attr]
            elif compare_type=='=':
                attr = [1 if a == compare_val else 0 for a in attr]
            elif compare_type=='>':
                attr = [1 if a > compare_val else 0 for a in attr]
            compare_val, compare_type = 1, '='
        
        # Return Video Level
        if not self.frame_level:
            if compare_type=='<':
                if any(attr) < compare_val: return 1
            elif compare_type=='=':
                if any(attr) == compare_val: return 1 
            elif compare_type=='>':
                if any(attr) > compare_val: return 1 
            return 0
        
        # Return Frame Level
        return attr
        
def get_video_attributes(frames_dir, anno_path, video_name, anno_delim, frame_level, bbox_type):
    video_attr = VideoAttributes(frames_dir, anno_path, anno_delim=anno_delim, 
                                 frame_level=frame_level, bbox_type=bbox_type)
    
    #attr_order = ['FO', 'SV', 'DF', 'FM', 'IV', 'LR']
    attr_order = ['FO', 'SV', 'DF', 'FM', 'LR']
    vid_attr_val = {video_name: [
        video_attr.occlusion(partial=False),
        video_attr.scale_variation(),
        video_attr.deformation(),
        video_attr.fast_motion(),
        #video_attr.illum_variation(),
        video_attr.low_resolution()
    ]} 
    return [attr_order, vid_attr_val]

def get_dataset_attributes(dataset_name, video_names: list, videos_dir: list, anno_paths: list, 
                           anno_delim=',', frame_level=False, bbox_type='xywh'):
    '''Obtain the attributes of all videos in a dataset'''
    assert len(videos_dir) == len(anno_paths)
    
    arg_list = [(f_dir, a_path, vid_name, anno_delim, frame_level, bbox_type) 
                for (f_dir, a_path, vid_name) in zip(videos_dir, anno_paths, video_names)]
    
    dataset_attributes = []
    with Pool(processes=5) as pool:
        for ret in tqdm(pool.starmap(get_video_attributes, arg_list), 
                        desc=f'{dataset_name} Attributes', total=len(videos_dir), ncols=100):
            dataset_attributes.append(ret)
                
    return {dataset_name: dataset_attributes}

def get_dataset_attributes_with_json(dataset_name, json_path, eval_json_path, anno_delim=',', 
                                     frame_level=False, bbox_type='xywh'):
    '''Obtain the attributes of all videos in a dataset using the JSON file'''
    
    # Read the json file
    with open(os.path.join(json_path), "r") as f:
            dataset_json = json.load(f)
    
    with open(os.path.join(eval_json_path), "r") as f:
            eval_dataset_json = json.load(f)    
    
    # Get evaluated videos from the eval json file
    video_names = eval_dataset_json.keys()
    
    # Get all unique attributes in the dataset
    video_attributes = [list(dataset_json[vid]['attr']) for vid in video_names 
                        if vid in dataset_json.keys()]
    
    all_attributes = [attr for video_attr in video_attributes for attr in video_attr]
    all_attributes = list(set(all_attributes))
    
    dataset_attributes = []
    unavailable_videos = []
    for vid_name in video_names:
        if vid_name in dataset_json.keys():
            vid_json_attr = dataset_json[vid_name]['attr']
        else:
            unavailable_videos.append(vid_name)
        
        vid_attr_result = [1 if atr in vid_json_attr else 0 for atr in all_attributes]

        vid_attr = {vid_name: vid_attr_result}
        dataset_attributes.append([all_attributes, vid_attr])
    
    print(f'The following videos are not available: {unavailable_videos}')            
    return {dataset_name: dataset_attributes}
 
def get_dataset_trackers_attributes_results(trackers, dataset_name, dataset_root, 
                                    trackers_results_dir, dataset_attributes=None, 
                                    percent=False, bar_only=False): 
    '''Obtain the trackers attributes results for the dataset'''
    # Compute the trackers evaluation on the dataset
    dataset = UTBDataset(dataset_name, dataset_root, load_img=False)
    dataset.set_tracker(trackers_results_dir, trackers)
    benchmark = OPEBenchmark(dataset)
    
    success_ret = {}
    with Pool(processes=5) as pool:
        for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
        trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
    
    if not bar_only:
        # Attribues Computation
        attributes = dataset_attributes[dataset_name]
        
        # Get all videos in the dataset
        attr_order = attributes[0][0]
        all_video_attributes = [atr[1] for atr in attributes]
        
        all_videos_names = [k for d in all_video_attributes for k in d.keys()]
        all_attributes = [list(v) for d in all_video_attributes for v in d.values()]
    else:
        all_videos_names = [k for k in success_ret[trackers[0]].keys()]
        
    # Create Trackers attributes dataframe and initialize with zeros
    trackers_attributes = None
    if not bar_only:
        columns = ['tracker']
        columns.extend(attr_order)
        
        trackers_attributes = pd.DataFrame(data=np.zeros((len(trackers), len(columns))),
                                        columns=columns) if percent else \
                            pd.DataFrame(columns=columns)                 
            
        trackers_attributes['tracker'] = trackers
        #trackers_attributes.set_index('trackers', inplace=True)
        
        # For each tracker, obtain the attributewise performance
        if not percent:
            for tracker in trackers:
                tracker_results = success_ret[tracker]
                for i, attr in enumerate(attr_order):
                    # Get all videos where attribute is present (Binary Attributes)
                    attr_present_videos = [vid for vid, atr in zip(all_videos_names, all_attributes) if atr[i]==1]
                    
                    # Get tracker result for videos where attribute is present
                    tracker_attr_results = [tracker_results[video] for video in attr_present_videos]
            
                    # Update the trackers_attribute table with result
                    trackers_attributes.iloc[trackers.index(tracker)][attr] = 0.0 if len(tracker_attr_results) == 0 \
                                                                                else np.mean(tracker_attr_results)
        else:
            # For each attribute, obtain the maximum tracker value for each video
            for i, attr in enumerate(attr_order):
                # Get all videos where attribute is present (Binary Attributes)
                attr_present_videos = [vid for vid, atr in zip(all_videos_names, all_attributes) if atr[i]==1]
                
                # For each present video, get result for each tracker
                for video in attr_present_videos:
                    tracker_video_result = [np.mean(success_ret[tracker][video]) for tracker in trackers]
                    
                    # Increment tracker attribute value if it is equal to max
                    trackers_attributes[attr] += [int(v) for v in tracker_video_result==max(tracker_video_result)]
            
    # Compute number of video where each tracker is best
    columns = ['tracker', 'Number of Videos']
    trackers_n_video_best = pd.DataFrame(data=np.zeros((len(trackers), len(columns))), columns=columns) 
    trackers_n_video_best['tracker'] = trackers
    
    trackers_n_video_best['Number of Videos'] = [0]*len(trackers)
    for video in all_videos_names:
        tracker_video_result = [np.mean(success_ret[tracker][video]) for tracker in trackers]
        
        # Increment tracker value if it is equal to max
        trackers_n_video_best['Number of Videos'] += [int(v) for v in tracker_video_result==max(tracker_video_result)]
     
    # Return trackers attributes result for the dataset
    return trackers_attributes, trackers_n_video_best        
    
    
def plot_dataset_trackers_attributes(dataset_name, df_trackers_attributes: pd.DataFrame, percent=False):
    trackers = df_trackers_attributes['tracker'].tolist()
    attributes = list(df_trackers_attributes.columns[1:])
    attributes += attributes[:1]  #to close the radar, duplicate the first column
    
    if not percent:
        # Get minimum and maximum result for each tracker
        min_res = df_trackers_attributes.min()[1:].tolist()
        min_res += min_res[:1]
        
        max_res = df_trackers_attributes.max()[1:].tolist()
        max_res += max_res[:1]
        
        attributes = [attr.replace(' ', '\n') for attr in attributes]
        attributes = [attr+f'\n{round(mi,3),round(ma,3)}' 
                    for (attr, mi, ma) in zip(attributes, min_res, max_res)]
    else:
        # Get the attributes in percentage
        attr_n_videos = df_trackers_attributes.sum()[1:].tolist()
        attr_n_videos += attr_n_videos[:1]
        
        for attr, n_videos in zip(attributes[:-1], attr_n_videos):
            df_trackers_attributes[attr] = round((df_trackers_attributes[attr]/n_videos)*100) 
            
        # Get number of videos in each attributes
        attributes = [attr.replace(' ', '\n') for attr in attributes]
        attributes = [attr+f'\n({round(n)} videos)' 
                    for (attr, n) in zip(attributes, attr_n_videos)]
        
    n_attrs = len(attributes)

    # Angles need to be converted to radian so we multiply by 2*pi 
    # and create the list of angles:
    label_loc = np.linspace(start=0, stop=2 * np.pi, num=n_attrs)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True), facecolor="white")
    #fig, ax = plt.subplots(subplot_kw=dict(polar=True), facecolor="white")

    markers = ['o', '*', 'v', 'd', 's'] 
    colors = ['blue', 'orange', 'green', 'red', 'purple', 
              'brown', 'olive', 'springgreen']
    # Plot each tracker on the radar chart
    min_val, max_val = 1, 0
    for i, tracker in enumerate(trackers):
        tracker_res = df_trackers_attributes.loc[df_trackers_attributes['tracker'] == tracker]
        tracker_res = tracker_res.values.tolist()[0][1:]
        tracker_res += tracker_res[:1]
        ax.plot(label_loc, tracker_res, linewidth=2, color=colors[i%len(colors)], 
                marker=markers[i%len(markers)], label=tracker)
        #ax.fill(label_loc, tracker_res, alpha=0.15)
        
        # Store min and max value store
        min_val = min(min_val, min(tracker_res))
        max_val = max(max_val, max(tracker_res))
        
    # Fix axis to go in the right order and start at 12 o'clock.
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw axis lines for each angle and label.
    ax.set_thetagrids(np.degrees(label_loc), attributes)

    # Go through labels and adjust alignment based on where
    # it is in the circle.
    for label, angle in zip(ax.get_xticklabels(), label_loc):
        if 0 < angle < np.pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')
    
    # Ensure radar goes from 0 to 100.
    min_val = max(min_val-0.01, 0) if not percent else max(min_val-1, 0)
    max_val = min(max_val+0.01, 1) if not percent else min(max_val+1, 100)
        
    ax.set_ylim(min_val, max_val)
    # You can also set gridlines manually like this:
    # ax.set_rgrids([20, 40, 60, 80, 100])
    
    #grid_points = [0.01]
    #grid_points.extend(np.around(np.arange(0.1, max_val, 0.1),decimals=1).tolist())
    #ax.set_rgrids(grid_points)

    # Add some custom styling.f v   ff
    # Change the color of the tick labels.
    ax.tick_params(colors='#222222')

    # Make the y-axis (0-100) labels smaller.
    ax.tick_params(axis='y', labelsize=8)
    # Change the color of the circular gridlines.
    ax.grid(color='#AAAAAA',linestyle='--')
    # Change the color of the outermost gridline (the spine).
    ax.spines['polar'].set_color('#AAAAAA')
    # Change the linestyle of the outermost gridline (the spine).
    ax.spines['polar'].set_linestyle('--')
    # Change the lineweight of the outermost gridline (the spine).
    ax.spines['polar'].set_linewidth(0.5)
    # Change the background color inside the circle itself.
    ax.set_facecolor('#FFFFFF')

    # Add title.
    ax.set_title(f'{dataset_name} Attributes AUC Plot', y=1.08) if not percent else \
        ax.set_title(f'{dataset_name} Attributes Percentage Plot', y=1.08)

    # Add a legend as well. 
    #ax.legend(loc='upper right', bbox_to_anchor=(1.5, 1.1)) 
    n_col = int(len(trackers)/2) if len(trackers)%2 == 0 else int(len(trackers)/2) + 1 
    ax.legend(loc='upper center', ncol=n_col, 
              bbox_to_anchor=(0.5, -0.06), fancybox=True, shadow=True) 
    #ax.legend(loc='upper right') 
    
    # Save figure and excel
    if not percent:
        plt.savefig(f"attr_average_plot_{dataset_name}.png") 
        df_trackers_attributes.to_excel(f"attr_average_{dataset_name}.xlsx")
    else:
        plt.savefig(f"attr_percent_plot_{dataset_name}.png") 
        df_trackers_attributes.to_excel(f"attr_percent_{dataset_name}.xlsx")


def plot_bar(dataset_name, res: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 10), facecolor="white")
    res.plot.bar(x=res.columns[0], ax=ax, y=res.columns[1], rot=0)
    ax.tick_params(axis='x', labelrotation=90)
    ax.grid(visible=True, axis='y', linestyle='--', linewidth = 0.5)
    ax.set_title(f"{dataset_name}: Trackers AUC Bar Chart")
    ax.set_xlabel("Trackers")
    ax.set_ylabel('Number of Videos')
    plt.savefig(f"trackers_bar_plot_{dataset_name}.png") 
    res.to_excel(f"trackers_bar_{dataset_name}.xlsx")