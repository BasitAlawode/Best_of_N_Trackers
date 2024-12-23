from glob import glob
import json
import os
import numpy as np
import cv2

from pysot.utils.bbox import get_axis_aligned_bbox, cxy_wh_2_rect
from basit_codes.vot_mask_to_bbox import parse_groundtruth_file, mask2bbox

def create_json_vot(videos_folder_path, dataset_name, gt_file_name="groundtruth_rect.txt", \
    frames_folder_name="imgs", convert_region=False, mask_annotated=False, delimiter='\t'):
    '''Creates JSON file for our videos:
    It follows the OTB json creation format'''

    all_folders = glob(videos_folder_path, recursive = True)
    all_folders.sort()

    videos_folders = []

    #for x in all_folders:   # Select only annotated videos
    #    videos_folders.append(x) if "Video" in x else None
    videos_folders = all_folders

    videos_dicts = {}
    for video_folder in videos_folders:
        gt_rect = []
        img_names = []

        if '\\' not in video_folder:
            video_dir = video_folder.split('/')[-2]
        else:
            video_dir = video_folder.split('\\')[-2]

        # Read video frames paths
        for file in os.listdir(video_folder):
            
            if file.find(f"{frames_folder_name}") != -1:
                all_imgs = os.listdir(os.path.join(video_folder, file)) 
                all_imgs.sort()
                for img in all_imgs:
                    img_names.append(f"{video_dir}/{frames_folder_name}/{img}")

        # Read GT bboxes in the text file and convert to list of bboxes    
        for file in os.listdir(video_folder): 
            if file.endswith(gt_file_name):   
                with open(os.path.join(video_folder, file)) as f:   # Read all
                    gt_bboxes = f.readlines()
                
                # Initial GT bbox
                if mask_annotated:
                    frame = cv2.imread(f"{videos_folder_path.split('/*')[0]}/{img_names[0]}") 
                    init_rect = mask2bbox(parse_groundtruth_file(gt_bboxes[0], [frame.shape[1], frame.shape[0]]))
                else:
                    init_rect = (gt_bboxes[0].split('\n')[0]).split(delimiter)
                init_rect = [int(x) for x in list(map(float, init_rect))]

                if convert_region:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(init_rect, dtype=np.float32))
                    bb = cxy_wh_2_rect((cx, cy,), (w, h))
                    init_rect = [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])]

                # All GT bboxes
                for gt_bb in gt_bboxes:
                    if mask_annotated:
                        gt_bb = mask2bbox(parse_groundtruth_file(gt_bb, [frame.shape[1], frame.shape[0]]))
                    else:
                        gt_bb = (gt_bb.split('\n')[0]).split(delimiter)
                    rect = [int(x) for x in list(map(float, gt_bb))]

                    if convert_region:
                        cx,cy,w,h = get_axis_aligned_bbox(np.array(rect, dtype=np.float32))
                        bb = cxy_wh_2_rect((cx, cy,), (w, h))
                        rect = [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])]

                    gt_rect.append(rect)
        
        # Camera Motion Attribute
        camera_motion = []
        for file in os.listdir(video_folder): 
            if file.endswith("camera_motion.tag"):
                with open(os.path.join(video_folder, file)) as f: # Read attribute
                    attrs = f.readlines()
            for a in attrs:
                a = a.split('\n')[0]
                camera_motion.append(int(a))
        
        # Illumination Change Attribute 
        illum_change = []
        for file in os.listdir(video_folder): 
            if file.endswith("illum_change.tag"):
                with open(os.path.join(video_folder, file)) as f: # Read attribute
                    attrs = f.readlines()
            for a in attrs:
                a = a.split('\n')[0]
                illum_change.append(int(a))

        # Motion Change Attribute 
        motion_change = []
        for file in os.listdir(video_folder): 
            if file.endswith("motion_change.tag"):
                with open(os.path.join(video_folder, file)) as f: # Read attribute
                    attrs = f.readlines()
            for a in attrs:
                a = a.split('\n')[0]
                motion_change.append(int(a))

        # Size Change Attribute 
        size_change = []
        for file in os.listdir(video_folder): 
            if file.endswith("size_change.tag"):
                with open(os.path.join(video_folder, file)) as f: # Read attribute
                    attrs = f.readlines()
            for a in attrs:
                a = a.split('\n')[0]
                size_change.append(int(a))

        # Occlusion Attribute 
        occlusion = []
        for file in os.listdir(video_folder): 
            if file.endswith("occlusion.tag"):
                with open(os.path.join(video_folder, file)) as f: # Read attribute
                    attrs = f.readlines()
            for a in attrs:
                a = a.split('\n')[0]
                occlusion.append(int(a))

        
        # Create dictionary of the current video
        video_dict = {}
        video_dict["video_dir"] = video_dir
        video_dict["init_rect"] = init_rect
        video_dict["img_names"] = img_names
        video_dict["gt_rect"] = gt_rect
        video_dict["camera_motion"] = camera_motion if len(camera_motion) > 0 else None
        video_dict["illum_change"] = illum_change if len(illum_change) > 0 else None
        video_dict["motion_change"] = motion_change if len(motion_change) > 0 else None
        video_dict["size_change"] = size_change if len(size_change) > 0 else None
        video_dict["occlusion"] = occlusion if len(occlusion) > 0 else None

        # Append current video_dict to videos_dicts
        videos_dicts[video_dir] = video_dict
    
    # Save dict as json file
    json_dir = videos_folder_path.split("*/")[0]
    with open(os.path.join(json_dir,f"{dataset_name}.json"), "w") as f:
        json.dump(videos_dicts, f)



def create_json_got(videos_folder_path, dataset_name, gt_file_name="groundtruth.txt", delimiter=','):
    '''Creates JSON file for our videos:
    It follows the OTB json creation format'''

    all_folders = glob(videos_folder_path, recursive = True)
    all_folders.sort()

    videos_folders = []

    #for x in all_folders:   # Select only annotated videos
    #    videos_folders.append(x) if "Video" in x else None
    videos_folders = all_folders

    videos_dicts = {}
    for video_folder in videos_folders:
        gt_rect = []
        img_names = []

        if '\\' not in video_folder:
            video_dir = video_folder.split('/')[-2]
        else:
            video_dir = video_folder.split('\\')[-2]

        # Read video frames paths
        all_files = os.listdir(video_folder) 
        all_imgs = []
        for s in all_files:
            if s.endswith(".jpg"):
                all_imgs.append(s)
        all_imgs.sort()
        for img in all_imgs:
            img_names.append(f"{video_dir}/{img}")

        # Read GT bboxes in the text file and convert to list of bboxes    
        for file in all_files: 
            if file.endswith(gt_file_name):   
                with open(os.path.join(video_folder, file)) as f:   # Read all
                    gt_bboxes = f.readlines()
                
                # Initial GT bbox
                init_rect = (gt_bboxes[0].split('\n')[0]).split(delimiter)
                init_rect = [int(x) for x in list(map(float, init_rect))]

                
                if len(gt_bboxes) == 1:    # For the test set
                    for i in range(len(img_names)):     
                        rect = None
                        if i == 0:
                            gt_bb = (gt_bboxes[i].split('\n')[0]).split(delimiter)
                            rect = [int(x) for x in list(map(float, gt_bb))]
                        gt_rect.append(rect)
                else:
                    # All GT bboxes
                    for gt_bb in gt_bboxes:
                        gt_bb = (gt_bb.split('\n')[0]).split(delimiter)
                        rect = [int(x) for x in list(map(float, gt_bb))]

                        gt_rect.append(rect)

        # Create dictionary of the current video
        video_dict = {}
        video_dict["video_dir"] = video_dir
        video_dict["init_rect"] = init_rect
        video_dict["img_names"] = img_names
        video_dict["gt_rect"] = gt_rect
        video_dict["camera_motion"] = None
        video_dict["illum_change"] = None
        video_dict["motion_change"] = None
        video_dict["size_change"] = None
        video_dict["occlusion"] = None


        # Append current video_dict to videos_dicts
        videos_dicts[video_dir] = video_dict
    
    # Save dict as json file
    json_dir = videos_folder_path.split("*/")[0]
    with open(os.path.join(json_dir,f"{dataset_name}.json"), "w") as f:
        json.dump(videos_dicts, f)
        

def create_json_lasot(videos_folder_path, dataset_name, gt_file_name="groundtruth.txt", delimiter=','):
    '''Creates JSON file for our videos:
    It follows the OTB json creation format'''

    all_folders = glob(videos_folder_path, recursive = True)
    all_folders.sort()

    videos_folders = []

    #for x in all_folders:   # Select only annotated videos
    #    videos_folders.append(x) if "Video" in x else None
    videos_folders = all_folders

    videos_dicts = {}
    for video_folder in videos_folders:
        gt_rect = []
        img_names = []

        if '\\' not in video_folder:
            video_dir = video_folder.split('/')[-2]
        else:
            video_dir = video_folder.split('\\')[-2]

        # Read video frames paths
        all_files = os.listdir(video_folder) 

        # Read GT bboxes in the text file and convert to list of bboxes    
        for file in all_files: 
            if file.endswith("img"):
                all = os.listdir(f"{video_folder}/{file}")
                all.sort()
                
                for s in all:
                    if s.lower().endswith(('.png', '.jpg')):
                        img_names.append(f"{video_folder}/{file}/{s}")
                    
            if file.endswith(gt_file_name):   
                with open(os.path.join(video_folder, file)) as f:   # Read all
                    gt_bboxes = f.readlines()
                
                # Initial GT bbox
                init_rect = (gt_bboxes[0].split('\n')[0]).split(delimiter)
                init_rect = [int(x) for x in list(map(float, init_rect))]

                # All GT bboxes
                for gt_bb in gt_bboxes:
                    gt_bb = (gt_bb.split('\n')[0]).split(delimiter)
                    rect = [int(x) for x in list(map(float, gt_bb))]

                    gt_rect.append(rect)

        # Create dictionary of the current video
        video_dict = {}
        video_dict["video_dir"] = video_dir
        video_dict["init_rect"] = init_rect
        video_dict["img_names"] = img_names
        video_dict["gt_rect"] = gt_rect
        video_dict["camera_motion"] = None
        video_dict["illum_change"] = None
        video_dict["motion_change"] = None
        video_dict["size_change"] = None
        video_dict["occlusion"] = None


        # Append current video_dict to videos_dicts
        videos_dicts[video_dir] = video_dict
    
    # Save dict as json file
    json_dir = videos_folder_path.split("*/")[0]
    with open(os.path.join(json_dir,f"{dataset_name}.json"), "w") as f:
        json.dump(videos_dicts, f)
        

def create_got_lasot_json(got_videos_folder_path, lasot_videos_folder_path, json_dir,
                          dataset_name, gt_file_name="groundtruth.txt", delimiter=','):
    '''Creates JSON file for our videos:
    It follows the OTB json creation format'''

    all_folders = glob(got_videos_folder_path, recursive = True)
    all_folders.sort()

    videos_folders = []

    #for x in all_folders:   # Select only annotated videos
    #    videos_folders.append(x) if "Video" in x else None
    videos_folders = all_folders

    videos_dicts = {}
    
    # GOT Videos
    for video_folder in videos_folders:
        gt_rect = []
        img_names = []

        if '\\' not in video_folder:
            video_dir = video_folder.split('/')[-2]
        else:
            video_dir = video_folder.split('\\')[-2]

        # Read video frames paths
        all_files = os.listdir(video_folder) 
        all_imgs = []
        for s in all_files:
            if s.endswith(".jpg"):
                all_imgs.append(s)
        all_imgs.sort()
        for img in all_imgs:
            img_names.append(f"{video_folder}/{img}")

        # Read GT bboxes in the text file and convert to list of bboxes    
        for file in all_files: 
            if file.endswith(gt_file_name):   
                with open(os.path.join(video_folder, file)) as f:   # Read all
                    gt_bboxes = f.readlines()
                
                # Initial GT bbox
                init_rect = (gt_bboxes[0].split('\n')[0]).split(delimiter)
                init_rect = [int(x) for x in list(map(float, init_rect))]

                # All GT bboxes
                for gt_bb in gt_bboxes:
                    gt_bb = (gt_bb.split('\n')[0]).split(delimiter)
                    rect = [int(x) for x in list(map(float, gt_bb))]

                    gt_rect.append(rect)

        # Create dictionary of the current video
        video_dict = {}
        video_dict["video_dir"] = video_dir
        video_dict["init_rect"] = init_rect
        video_dict["img_names"] = img_names
        video_dict["gt_rect"] = gt_rect
        video_dict["camera_motion"] = None
        video_dict["illum_change"] = None
        video_dict["motion_change"] = None
        video_dict["size_change"] = None
        video_dict["occlusion"] = None


        # Append current video_dict to videos_dicts
        videos_dicts[video_dir] = video_dict
    
    
    # LASOT Videos
    all_folders = glob(lasot_videos_folder_path, recursive = True)
    all_folders.sort()

    videos_folders = []

    #for x in all_folders:   # Select only annotated videos
    #    videos_folders.append(x) if "Video" in x else None
    videos_folders = all_folders

    for video_folder in videos_folders:
        gt_rect = []
        img_names = []

        if '\\' not in video_folder:
            video_dir = video_folder.split('/')[-2]
        else:
            video_dir = video_folder.split('\\')[-2]

        # Read video frames paths
        all_files = os.listdir(video_folder) 

        # Read GT bboxes in the text file and convert to list of bboxes    
        for file in all_files: 
            if file.endswith("img"):
                all = os.listdir(f"{video_folder}/{file}")
                all.sort()
                
                for s in all:
                    if s.lower().endswith(('.png', '.jpg')):
                        img_names.append(f"{video_folder}/{file}/{s}")
                    
            if file.endswith(gt_file_name):   
                with open(os.path.join(video_folder, file)) as f:   # Read all
                    gt_bboxes = f.readlines()
                
                # Initial GT bbox
                init_rect = (gt_bboxes[0].split('\n')[0]).split(delimiter)
                init_rect = [int(x) for x in list(map(float, init_rect))]

                # All GT bboxes
                for gt_bb in gt_bboxes:
                    gt_bb = (gt_bb.split('\n')[0]).split(delimiter)
                    rect = [int(x) for x in list(map(float, gt_bb))]

                    gt_rect.append(rect)

        # Create dictionary of the current video
        video_dict = {}
        video_dict["video_dir"] = video_dir
        video_dict["init_rect"] = init_rect
        video_dict["img_names"] = img_names
        video_dict["gt_rect"] = gt_rect
        video_dict["camera_motion"] = None
        video_dict["illum_change"] = None
        video_dict["motion_change"] = None
        video_dict["size_change"] = None
        video_dict["occlusion"] = None


        # Append current video_dict to videos_dicts
        videos_dicts[video_dir] = video_dict
    
    # Save dict as json file
    with open(f"{json_dir}/{dataset_name}.json", "w") as f:
        json.dump(videos_dicts, f)


def create_json_trackingNet(videos_folder_path, dataset_name, delimiter=','):
    '''Creates JSON file for our videos:
    It follows the OTB json creation format'''
    from natsort import natsorted
    all__videos_path = glob(f"{videos_folder_path}/zips/*/", recursive = True)
    videos_folders = natsorted(all__videos_path)

    videos_dicts = {}
    for video_folder in videos_folders:
        gt_rect = []
        img_names = []

        if '\\' not in video_folder:
            video_dir = video_folder.split('/')[-2]
        else:
            video_dir = video_folder.split('\\')[-2]

        # Read video frames paths
        all_frames = natsorted(os.listdir(video_folder))
        
        # Read GT file  
        with open(f"{videos_folder_path}/anno/{video_dir}.txt") as f:   # Read all
            gt_bboxes = f.readlines()
        
        # Initial GT bbox
        init_rect = (gt_bboxes[0].split('\n')[0]).split(delimiter)
        init_rect = [int(x) for x in list(map(float, init_rect))]

        if len(gt_bboxes) == 1:    # For the test set
            for i, frame in enumerate(all_frames):
                img_names.append(f"{video_folder}{frame}")
                
                rect = None
                if i == 0:
                    gt_bb = (gt_bboxes[i].split('\n')[0]).split(delimiter)
                    rect = [int(x) for x in list(map(float, gt_bb))]
                gt_rect.append(rect)
        else:
            for frame, gt_bbox in zip(all_frames, gt_bboxes):
                img_names.append(f"{video_folder}{frame}")
                
                gt_bb = (gt_bbox.split('\n')[0]).split(delimiter)
                rect = [int(x) for x in list(map(float, gt_bb))]
                gt_rect.append(rect)

        # Create dictionary of the current video
        video_dict = {}
        video_dict["video_dir"] = video_dir
        video_dict["init_rect"] = init_rect
        video_dict["img_names"] = img_names
        video_dict["gt_rect"] = gt_rect
        video_dict["camera_motion"] = None
        video_dict["illum_change"] = None
        video_dict["motion_change"] = None
        video_dict["size_change"] = None
        video_dict["occlusion"] = None
    
        # Append current video_dict to videos_dicts
        videos_dicts[video_dir] = video_dict
    
    # Save dict as json file
    with open(f"{videos_folder_path}/{dataset_name}.json", "w") as f:
        json.dump(videos_dicts, f)

def create_json_uav123(videos_folder_path, dataset_name, delimiter=','):
    '''Creates JSON file for our videos:
    It follows the OTB json creation format'''
    from natsort import natsorted
    all__videos_path = glob(f"{videos_folder_path}/data_seq/*/", recursive = True)
    videos_folders = natsorted(all__videos_path)

    videos_dicts = {}
    for video_folder in videos_folders:
        gt_rect = []
        img_names = []

        if '\\' not in video_folder:
            video_dir = video_folder.split('/')[-2]
        else:
            video_dir = video_folder.split('\\')[-2]

        # Read video frames paths
        all_frames = natsorted(os.listdir(video_folder))
        
        # Read GT file  
        with open(f"{videos_folder_path}/anno/UAV123/{video_dir}.txt") as f:   # Read all
            gt_bboxes = f.readlines()
        
        # Initial GT bbox
        init_rect = (gt_bboxes[0].split('\n')[0]).split(delimiter)
        init_rect = [int(x) for x in list(map(float, init_rect))]

        if len(gt_bboxes) == 1:    # For the test set
            for i, frame in enumerate(all_frames):
                img_names.append(f"{video_folder}{frame}")
                
                rect = None
                if i == 0:
                    gt_bb = (gt_bboxes[i].split('\n')[0]).split(delimiter)
                    rect = [int(x) for x in list(map(float, gt_bb))]
                gt_rect.append(rect)
        else:
            for frame, gt_bbox in zip(all_frames, gt_bboxes):
                img_names.append(f"{video_folder}{frame}")
                
                gt_bb = (gt_bbox.split('\n')[0]).split(delimiter)
                gt_bb = [0 if g=='NaN' else g for g in gt_bb]
                rect = [int(x) for x in list(map(float, gt_bb))]
                gt_rect.append(rect)

        # Create dictionary of the current video
        video_dict = {}
        video_dict["video_dir"] = video_dir
        video_dict["init_rect"] = init_rect
        video_dict["img_names"] = img_names
        video_dict["gt_rect"] = gt_rect
        video_dict["camera_motion"] = None
        video_dict["illum_change"] = None
        video_dict["motion_change"] = None
        video_dict["size_change"] = None
        video_dict["occlusion"] = None
    
        # Append current video_dict to videos_dicts
        videos_dicts[video_dir] = video_dict
    
    # Save dict as json file
    with open(f"{videos_folder_path}/{dataset_name}.json", "w") as f:
        json.dump(videos_dicts, f)

def create_json_otb(videos_folder_path, dataset_name, delimiter=',',
                    gt_file_name='groundtruth_rect.txt'):
    '''Creates JSON file for our videos:
    It follows the OTB json creation format'''

    all_folders = glob(videos_folder_path, recursive = True)
    all_folders.sort()

    videos_folders = []

    #for x in all_folders:   # Select only annotated videos
    #    videos_folders.append(x) if "Video" in x else None
    videos_folders = all_folders

    videos_dicts = {}
    for video_folder in videos_folders:
        gt_rect = []
        img_names = []

        if '\\' not in video_folder:
            video_dir = video_folder.split('/')[-2]
        else:
            video_dir = video_folder.split('\\')[-2]

        # Read video frames paths
        all_files = os.listdir(video_folder) 

        # Read GT bboxes in the text file and convert to list of bboxes    
        for file in all_files: 
            if file.endswith("img"):
                all = os.listdir(f"{video_folder}/{file}")
                all.sort()
                
                for s in all:
                    if s.lower().endswith(('.png', '.jpg')):
                        img_names.append(f"{video_folder}/{file}/{s}")
                    
            if file.endswith(gt_file_name):   
                with open(os.path.join(video_folder, file)) as f:   # Read all
                    gt_bboxes = f.readlines()
                
                # Initial GT bbox
                init_rect = (gt_bboxes[0].split('\n')[0]).split(delimiter)
                init_rect = init_rect[0].split('\t') if len(init_rect) == 1 else init_rect
                init_rect = init_rect[0].split(' ') if len(init_rect) == 1 else init_rect
                init_rect = [int(x) for x in list(map(float, init_rect))]

                # All GT bboxes
                for gt_bb in gt_bboxes:
                    gt_bb = (gt_bb.split('\n')[0]).split(delimiter)
                    gt_bb = gt_bb[0].split('\t') if len(gt_bb) == 1 else gt_bb
                    gt_bb = gt_bb[0].split(' ') if len(gt_bb) == 1 else gt_bb
                    rect = [int(x) for x in list(map(float, gt_bb))]

                    gt_rect.append(rect)

        # Create dictionary of the current video
        video_dict = {}
        video_dict["video_dir"] = video_dir
        video_dict["init_rect"] = init_rect
        video_dict["img_names"] = img_names
        video_dict["gt_rect"] = gt_rect
        video_dict["camera_motion"] = None
        video_dict["illum_change"] = None
        video_dict["motion_change"] = None
        video_dict["size_change"] = None
        video_dict["occlusion"] = None


        # Append current video_dict to videos_dicts
        videos_dicts[video_dir] = video_dict
    
    # Save dict as json file
    json_dir = videos_folder_path.split("*/")[0]
    with open(os.path.join(json_dir,f"{dataset_name}.json"), "w") as f:
        json.dump(videos_dicts, f)