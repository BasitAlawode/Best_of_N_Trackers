import torch
import os
import sys

def tracker_path_config(tracker_name):
    '''Configure Tracker:
    Configure this separately for all trackers'''

    base_folder = "trained_trackers"
    if tracker_name == "SiamRPN":
        folder = f'{base_folder}/siamrpn_r50_l234_dwxcorr'
        model_config_path = f'{folder}/config.yaml'
        model_path = f'{folder}/model.pth'
    elif tracker_name == "SiamBAN":
        folder = f'{base_folder}/siamban_r50_l234'
        model_config_path = f'{folder}/config.yaml'
        model_path = f'{folder}/model.pth'
    elif tracker_name == "TransT":
        folder = f'{base_folder}/TransT'
        model_config_path = 'TransT'
        model_path = f'{folder}/transt.pth'
    elif tracker_name == "KeepTrack":    
        model_config_path = 'default'
        model_path = f'{base_folder}/pytracking/keep_track.pth.tar'
    elif tracker_name == "ToMP":    
        model_config_path = 'tomp50'
        model_path = f'{base_folder}/pytracking/tomp.pth.tar'
    elif tracker_name == "TrDiMP":    
        model_config_path = 'trdimp'
        model_path = f'{base_folder}/TransformerTrack/trdimp_net.pth.tar'
    elif tracker_name == "TrSiam":   
        model_config_path = 'trsiam'
        model_path = f'{base_folder}/TransformerTrack/trdimp_net.pth.tar'
    elif tracker_name == "RTS":    
        model_config_path = 'rts50'
        model_path = f'{base_folder}/pytracking/rts50.pth'
    elif tracker_name == "ARDiMP":
        model_config_path = None
        model_path = [f'{base_folder}/ardimp/super_dimp.pth.tar', 
                      f'{base_folder}/ardimp/SEcmnet_ep0040-c.pth.tar']
    elif tracker_name == "STMTrack":
        model_config_path = 'stmtrack/experiments/stmtrack/test/got10k/stmtrack-googlenet-got.yaml'
        model_path = f'{base_folder}/stmtrack/epoch-19_got10k.pkl'
    elif tracker_name == "SparseTT":
        model_config_path = 'sparsett/experiments/sparsett/test/got10k/sparsett_swin_got10k.yaml'
        model_path = f'{base_folder}/sparsett/model_got10k.pkl'
    elif tracker_name == "AutoMatch":
        model_config_path = 'automatch/experiments/AutoMatch.yaml'
        model_path = f'{base_folder}/automatch/AutoMatch.pth'
    elif tracker_name == "OSTrack":
        model_config_path = 'vitb_384_mae_ce_32x4_ep300'
        model_path = f'{base_folder}/ostrack/OSTrack_ep0300.pth.tar'
    elif tracker_name == "GRM":
        model_config_path = 'vitb_256_ep300'
        model_path = f'{base_folder}/grm/GRM_ep0300.pth.tar'
    elif tracker_name == "AiATrack":
        model_config_path = 'baseline'
        model_path = f'{base_folder}/AiATrack/AIATRACK_ep0500.pth.tar'
    elif tracker_name == "ARTrack":
        model_config_path = 'artrack_seq_256_full'
        model_path = f'{base_folder}/ARTrack/ARTrackSeq_ep0060.pth-001.tar'
    elif tracker_name == "DropTrack":
        model_config_path = 'vitb_384_mae_ce_32x4_ep300'
        model_path = f'{base_folder}/DropTrack/DropTrack_k700_800E_alldata.pth.tar'
    elif tracker_name == "CiteTracker":
        model_config_path = 'vitb_384_mae_ce_32x4_ep300'
        model_path = f'{base_folder}/CiteTracker/CiteTracker_ep0300.pth.tar' 
    else:
        raise ValueError('No Matching Tracker Name')

    return model_config_path, model_path

def build_tracker(model_config_path, model_path, tracker_name):
    if tracker_name == "SiamRPN":                 # SiamRPN and SiamMask Trackers
        insert_path("pysot")
        
        if 'cfg' in locals():
            del cfg
            
        from pysot.core.config import cfg 
        from pysot.models.model_builder import ModelBuilder
        from pysot.tracker.tracker_builder import build_tracker

        cfg.merge_from_file(model_config_path)
        cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
        device = torch.device('cuda' if cfg.CUDA else 'cpu')

        model = ModelBuilder()
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, 
                loc: storage.cpu()))
        model.eval().to(device)
        return build_tracker(model), 0 
    elif tracker_name == "SiamBAN":                             # SiamBAN Tracker
        # load tracker config
        insert_path("siamban")
        
        from siamban.core.config import cfg
        from siamban.models.model_builder import ModelBuilder
        from siamban.tracker.tracker_builder import build_tracker
        from siamban.utils.model_load import remove_prefix, check_keys

        cfg.merge_from_file(model_config_path)
        cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
        device = torch.device('cuda' if cfg.CUDA else 'cpu')

        # create  and load model
        model = ModelBuilder()

        pretrained_dict = torch.load(model_path,
            map_location=lambda storage, loc: storage.cpu())
        
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = remove_prefix(pretrained_dict['state_dict'],
                                            'module.')
        else:
            pretrained_dict = remove_prefix(pretrained_dict, 'module.')

        try:
            check_keys(model, pretrained_dict)
        except:
            raise "Unable to check keys"
        model.load_state_dict(pretrained_dict, strict=False)

        model.eval().to(device)
        return build_tracker(model), 0 
    elif tracker_name == "TransT":
        insert_path("transt")
        
        from transt.pysot_toolkit.trackers.tracker import Tracker as transtTracker
        from transt.pysot_toolkit.trackers.net_wrappers import NetWithBackbone
        net = NetWithBackbone(net_path=model_path, use_gpu=True)
        tracker = transtTracker(name='transt', net=net, window_penalty=0.49, \
            exemplar_size=128, instance_size=256)
        #del Tracker
        return tracker, 0
    elif tracker_name == "ToMP" or tracker_name == "RTS" or \
        tracker_name == "KeepTrack": # Pytracking trackers
        
        insert_path(None)
            
        from pytracking.evaluation import Tracker

        t_name = "keep_track" if tracker_name == "KeepTrack" else tracker_name.lower()

        tracker = Tracker(t_name, model_config_path, display_name=tracker_name)
        return customize_pytracking_tracker(tracker), 0  
    elif tracker_name == "TrDiMP" or tracker_name == "TrSiam":
        insert_path("TransformerTrack")

        from TransformerTrack.pytracking.evaluation import Tracker
        tracker = Tracker("trdimp", model_config_path, display_name=tracker_name)
        return customize_pytracking_tracker(tracker), 0
    elif tracker_name == "SparseTT":
        insert_path("sparsett")

        import os.path as osp
        from videoanalyst.config.config import cfg as root_cfg
        from videoanalyst.config.config import specify_task
        #from videoanalyst.engine.builder import build as tester_builder
        from videoanalyst.model import builder as model_builder
        from videoanalyst.pipeline import builder as pipeline_builder

        exp_cfg_path = osp.realpath(model_config_path)
        root_cfg.merge_from_file(exp_cfg_path)
        root_cfg = root_cfg.test

        root_cfg.defrost()
        root_cfg.track.model.task_model.SiamTrack.pretrain_model_path = model_path
        root_cfg.track.model.backbone.SwinTransformer.pretrained = \
                'trained_trackers/sparsett/swin_tiny_patch4_window7_224.pth'

        task, task_cfg = specify_task(root_cfg)
        task_cfg.freeze()
        torch.multiprocessing.set_start_method('spawn', force=True)

        model = model_builder.build("track", task_cfg.model)
        pipeline = pipeline_builder.build("track", task_cfg.pipeline, model)
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pipeline.set_device(dev)
        return pipeline, 0
    elif tracker_name == "STMTrack":
        insert_path("stmtrack")

        import os.path as osp
        from stmtrack.videoanalyst.config.config import cfg as root_cfg
        from stmtrack.videoanalyst.config.config import specify_task
        #from videoanalyst.engine.builder import build as tester_builder
        from stmtrack.videoanalyst.model import builder as model_builder
        from stmtrack.videoanalyst.pipeline import builder as pipeline_builder
        from stmtrack.videoanalyst.utils import complete_path_wt_root_in_cfg

        exp_cfg_path = osp.realpath(model_config_path)
        root_cfg.merge_from_file(exp_cfg_path)

        root_cfg = complete_path_wt_root_in_cfg(root_cfg, os.getcwd())
        root_cfg = root_cfg.test

        root_cfg.defrost()
        root_cfg.track.model.task_model.STMTrack.pretrain_model_path = model_path

        task, task_cfg = specify_task(root_cfg)
        task_cfg.freeze()
        torch.multiprocessing.set_start_method('spawn', force=True)

        model = model_builder.build("track", task_cfg.model)
        pipeline = pipeline_builder.build("track", task_cfg.pipeline, model)
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pipeline.set_device(dev)
        return pipeline, 0
    elif tracker_name == "ARDiMP":
        insert_path("ardimp")

        #from ardimp.demo import get_dimp, get_ar
        # Get DiMP tracker

        from ardimp.pytracking.parameter.dimp.super_dimp_demo import parameters
        from ardimp.pytracking.tracker.dimp.dimp import DiMP

        params = parameters(model_path[0])
        params.visualization = True
        params.debug = False
        params.visdom_info = {'use_visdom': False, 'server': '127.0.0.1', 'port': 8097}
        dimp_tracker = DiMP(params)

        # Get Refine module
        from ardimp.pytracking.refine_modules.refine_module import RefineModule
        selector_path = 0
        sr = 2.0; input_sz = int(128 * sr)  # 2.0 by default
        RF_module = RefineModule(model_path[1], selector_path, search_factor=sr, input_sz=input_sz)

        tracker = [dimp_tracker, RF_module]
        return tracker, 0
    elif tracker_name == "AutoMatch":
        insert_path("automatch")

        import automatch.lib.tracker.sot_tracker as tracker_builder
        import automatch.lib.utils.model_helper as loader
        import automatch.lib.utils.sot_builder as builder
        import automatch.lib.utils.read_file as reader
        from easydict import EasyDict as edict

        config = edict(reader.load_yaml(model_config_path))
        siam_tracker = tracker_builder.SiamTracker(config)
        siambuilder = builder.Siamese_builder(config)
        siam_net = siambuilder.build()

        siam_net = loader.load_pretrain(siam_net, model_path, addhead=True, print_unuse=False)
        siam_net.eval()
        siam_net = siam_net.cuda()
        tracker = [siam_net, siam_tracker]
        return tracker, 0
    elif tracker_name == "OSTrack":   
        insert_path(tracker_name)
        
        from OSTrack.lib.test.evaluation import Tracker
        
        tracker = Tracker("ostrack", model_config_path, 'lasot', display_name=tracker_name)
        params = tracker.get_parameters()
        params.checkpoint = f'{os.getcwd()}/{model_path}'
        params.debug = False
        tracker = tracker.create_tracker(params)
        
        return tracker, 0
    elif tracker_name == "GRM":   
        insert_path(tracker_name)
        
        from GRM.lib.test.evaluation import Tracker
        
        tracker = Tracker("grm", model_config_path, 'lasot', display_name=tracker_name)
        params = tracker.get_parameters()
        params.checkpoint = f'{os.getcwd()}/{model_path}'
        params.debug = False
        tracker = tracker.create_tracker(params)
        
        return tracker, 0 
    elif tracker_name == "AiATrack":   
        insert_path(tracker_name)
        
        from AiATrack.lib.test.evaluation import Tracker
        
        tracker = Tracker("aiatrack", model_config_path, 'lasot', display_name=tracker_name)
        params = tracker.get_parameters()
        params.checkpoint = f'{os.getcwd()}/{model_path}'
        params.debug = False
        tracker = tracker.create_tracker(params)
        
        return tracker, 0
    elif tracker_name == "ARTrack":   
        insert_path(tracker_name)
        
        from ARTrack.lib.test.evaluation import Tracker
        
        tracker = Tracker("artrack_seq", model_config_path, 'lasot', display_name=tracker_name)
        params = tracker.get_parameters()
        params.checkpoint = f'{os.getcwd()}/{model_path}'
        params.cfg.MODEL.PRETRAIN_PTH = f'{os.getcwd()}/{model_path}'
        params.debug = False
        tracker = tracker.create_tracker(params)
        
        return tracker, 0
    elif tracker_name == "DropTrack":   
        insert_path(tracker_name)
        
        from DropTrack.lib.test.evaluation import Tracker
        
        tracker = Tracker("ostrack", model_config_path, 'lasot', display_name=tracker_name)
        params = tracker.get_parameters()
        params.checkpoint = f'{os.getcwd()}/{model_path}'
        params.debug = False
        tracker = tracker.create_tracker(params)
        
        return tracker, 0 
    elif tracker_name == "CiteTracker":   
        insert_path(tracker_name)
        
        from CiteTracker.lib.test.evaluation import Tracker
        
        tracker = Tracker("citetrack", model_config_path, 'lasot', display_name=tracker_name)
        params = tracker.get_parameters()
        params.checkpoint = f'{os.getcwd()}/{model_path}'
        params.debug = False
        tracker = tracker.create_tracker(params)
        
        return tracker, 0 
    else:
        raise 'No Matching Tracker Name'


def customize_pytracking_tracker(tracker, debug=None, visdom_info=None):
    from pytracking.evaluation.multi_object_wrapper import MultiObjectWrapper
    
    params = tracker.get_parameters()

    debug_ = debug
    if debug is None:
        debug_ = getattr(params, 'debug', 0)
    params.debug = debug_

    params.tracker_name = tracker.name
    params.param_name = tracker.parameter_name

    if tracker.display_name != 'STARK':
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

def insert_path(path_name=None):
    import sys
    
    pp = os.getcwd()
    pp = f'{pp}/{path_name}/' if path_name is not None else pp
    
    to_check = ['transt', 'ltr', 'pytracking']
    for p in sys.path:
        if p==pp or any(k in p for k in to_check):
            sys.path.remove(p)
    
    sys.path.insert(0, pp)