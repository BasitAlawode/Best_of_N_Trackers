import torch
import os
import numpy as np
import random

def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results
    
    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_requires_grad(model: torch.nn.Module, requires_grad=True):
    def get_all_layers(block):
        # get children form model!
        children = list(block.children())
        flatt_children = []
        if children == []:
            # if model has no children; model is last child! :O
            return block
        else:
            # look for children from children... to the last child!
            for child in children:
                try:
                    flatt_children.extend(get_all_layers(child))
                except TypeError:
                    flatt_children.append(get_all_layers(child))
        
        return flatt_children

    total_params = 0
    for l in get_all_layers(model):        # Set requires_grad
        for param in l.parameters():
            param.requires_grad = requires_grad
            total_params += param.numel()
    
    return total_params

def customize_pytracking_tracker(tracker, debug=None, visdom_info=None, 
                                 model_name="pretrained_model.pth"):
    if tracker.name == 'trdimp':
        from TransformerTrack.pytracking.evaluation.multi_object_wrapper import MultiObjectWrapper
        params = tracker.get_parameters()
        params.net.net_path = model_name
    elif tracker.name == 'atom':
        from pytracking.evaluation.multi_object_wrapper import MultiObjectWrapper
        params = tracker.get_parameters()
        params.features.features[0].net_path = model_name
    elif tracker.name == 'tomp':
        from pytracking.evaluation.multi_object_wrapper import MultiObjectWrapper
        params = tracker.get_parameters()
        params.net.net_path = model_name

    debug_ = debug
    if debug is None:
        debug_ = getattr(params, 'debug', 0)
    params.debug = debug_

    params.tracker_name = tracker.name
    params.param_name = tracker.parameter_name

    if tracker.display_name != 'stark':
        tracker._init_visdom(visdom_info, debug_)

    multiobj_mode = getattr(params, 'multiobj_mode', getattr(tracker.tracker_class, \
        'multiobj_mode', 'default'))

    if multiobj_mode == 'default':
        tracker = tracker.create_tracker(params)
        if hasattr(tracker, 'initialize_features'):
            tracker.initialize_features()

    elif multiobj_mode == 'parallel':
        tracker = MultiObjectWrapper(tracker.tracker_class, params, tracker.visdom, \
             fast_load=True)
    else:
        raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

    return tracker