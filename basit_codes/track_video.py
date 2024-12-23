import os
import cv2

from basit_codes.tracker_model import tracker_path_config, build_tracker

def track_video(video_details, tracker_name, base_dir, dataset_name, full_img_path=False):

    init_rect = video_details['init_rect']
    img_names = video_details['img_names']

    # Load Tracker model
    model_config_path,  model_path = tracker_path_config(tracker_name)

    # create tracker model and load parameters
    tracker, hp = build_tracker(model_config_path, model_path, tracker_name)  # hp is needed by SiamCar
    
    # Track....
    pred_bboxes = []
    for f, img_name in enumerate(img_names):
        
        frame = cv2.imread(img_name) if full_img_path else \
                cv2.imread(os.path.join(base_dir, "testing_datasets", dataset_name, img_name))

        if f == 0:
            pred_bbox, state = track(tracker, tracker_name, f, frame, hp, init_rect=init_rect)
        else:
            pred_bbox, state = track(tracker, tracker_name, f, frame, hp, state=state)
        pred_bboxes.append([int(pred_bbox[0]), int(pred_bbox[1]), int(pred_bbox[2]), \
                            int(pred_bbox[3])])
        
        if f%100 == 0:
            print(f'.......{f} of {len(img_names)} frames processed..')

    print(f'.......All {len(img_names)} frames processed...')
    
    return pred_bboxes


def track(tracker, tracker_name, img_index, image, hp, init_rect=None, state=None):
    if img_index==0:            # Initialize tracker with first bbox
        assert init_rect is not None
        print(".....Initializing Tracker...")

        if tracker_name == "TransT":
            tracker.initialize(image, {'init_bbox':init_rect})
        elif tracker_name == "D3S":
            tracker.initialize(image, init_rect)
        elif tracker_name == "DiMP" or tracker_name == "ToMP" or \
            tracker_name == "TrDiMP" or tracker_name == "RTS" or \
            tracker_name == "TrSiam" or tracker_name == "KeepTrack":
                
            from collections import OrderedDict
            def _build_init_info(box):
                return {'init_bbox': OrderedDict({1: box}), 'init_object_ids': [1, ], 'object_ids': [1, ],
                        'sequence_object_ids': [1, ]}
            state = tracker.initialize(image, _build_init_info(init_rect))
        elif tracker_name == "STMTrack" or tracker_name == "SparseTT":
            tracker.init(image, init_rect)
        elif tracker_name in ["OSTrack", "GRM", "AiATrack", "ARTrack", "DropTrack", "CiteTracker"]:
            def _build_init_info(box):
                return {'init_bbox': box}
            tracker.initialize(image, _build_init_info(init_rect))
        elif tracker_name == "ARDiMP":
            import numpy as np
            init_info = {}
            init_info['init_bbox'] = init_rect
            tracker[0].initialize(image, init_info)
            tracker[1].initialize(image, np.array(init_rect))
        elif tracker_name == "AutoMatch":
            import numpy as np
            x,y,w,h = init_rect
            cx, cy = x+w/2, y+h/2
            target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
            init_inputs = {'image': image, 'pos': target_pos, 'sz': target_sz, 'model': tracker[0]}
            tracker[1].init(init_inputs)  # init tracker
        else:       # For siamcar, siamrpn, siammask, siamban, TrTr, siamfc, siamfcpp , siamgat
            tracker.init(image, init_rect)
        print(".....Tracker Initialized Successfully.")
        pred_bbox = init_rect
    else:               # Continue tracking after first frame
        if tracker_name == "STMTrack" or tracker_name == "SparseTT":
            pred_bbox = tracker.update(image)
        elif tracker_name in ["OSTrack", "GRM", "AiATrack", "ARTrack", "DropTrack", "CiteTracker"]:
            state = tracker.track(image)
            pred_bbox = [int(s) for s in state['target_bbox']]
        elif tracker_name == "TransT":
            out = tracker.track(image, {})
            pred_bbox = [int(s) for s in out['target_bbox']]
        elif tracker_name == "DiMP" or tracker_name == "ToMP" or \
            tracker_name == "TrDiMP" or tracker_name == "RTS" or \
            tracker_name == "TrSiam" or tracker_name == "KeepTrack":
            
            info = None
            if tracker_name == "RTS":
                info = {'previous_output': state}
            state = tracker.track(image, info)
            pred_bbox = [int(s) for s in state['target_bbox'][1]]
        
        elif tracker_name == "ARDiMP":
            import numpy as np
            outputs = tracker[0].track(image)
            pred_bbox = outputs['target_bbox']
            pred_bbox = tracker[1].refine(image, np.array(pred_bbox))

            # Update base tracker with Alpha-Refine Results
            from ardimp.pytracking.RF_utils import bbox_clip
            import torch

            H, W, _ = image.shape
            x1, y1, w, h = pred_bbox.tolist()
            x1, y1, x2, y2 = bbox_clip(x1, y1, x1 + w, y1 + h, (H, W))
            w, h = x2 - x1, y2 - y1
            new_pos = torch.from_numpy(np.array([y1 + h / 2, x1 + w / 2]).astype(np.float32))
            new_target_sz = torch.from_numpy(np.array([h, w]).astype(np.float32))
            new_scale = torch.sqrt(new_target_sz.prod() / tracker[0].base_target_sz.prod())

            tracker[0].pos = new_pos.clone()
            tracker[0].target_sz = new_target_sz
            tracker[0].target_scale = new_scale
            pred_bbox = list(map(int, pred_bbox))
        elif tracker_name == "AutoMatch":
            from automatch.lib.utils import box_helper
            state = tracker[1].track(image)
            pred_bbox = box_helper.cxy_wh_2_rect(state['pos'], state['sz'])
        else:       # For siamcar, siamrpn, siammask, siamban, TrTr, siamgat
            outputs = tracker.track(image)
            pred_bbox = outputs["bbox"]
    return pred_bbox, state         # State is needed by DaSiamRPN