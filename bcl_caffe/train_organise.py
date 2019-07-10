import os
import fire
import caffe
from caffe import layers as L, params as P
from models import seg_obj, seg_net, obj_net, prior_seg_net, voxel_seg_net, bcl_net
from solver import solver_function_org
from tools import some_useful_tools as sut
from solver.dual_model_solver import DualEvalWrapper
from solver.solver_function_org import load_config

def Finder_LR_Solver(exp_dir, train_proto_path, eval_proto_path, config_path,
                        pretrain_path, args):

    solver = solver_function_org.SolverWrapper(
                                            train_proto_path,
                                            eval_proto_path,
                                            model_dir=exp_dir,
                                            config_path=config_path,
                                            prefix='pp',
                                            pretrain=pretrain_path,
                                            lr_policy= 'exp_increase', #'clr',step
                                            base_lr=1e-6, #0.0001, #fd 1e-7
                                            max_lr=1, #use when in "clr"
                                            momentum = 0.9, #0.85 ///  0.9 for no momentum decay
                                            cycle_steps=1872,
                                            stepsize=1, # 7508, 11220, #learning rate decay step, 1 for one cycle  # fd 936
                                            max_iter= 1872, #40700, #69600, # 69600 = 37.5*1856 # fd 936*7 = 1638
                                            solver_type='ADAM',
                                            weight_decay=0.0001, # 0.001,
                                            iter_size=1, #'number of mini-batches per iteration', batchsize*itersize = real_batch size
                                            display=4,
                                            random_seed=0,
                                            args = args)
    return solver

def Train_Solver(exp_dir, train_proto_path, eval_proto_path, config_path,
                        pretrain_path, args):
    # Load solver wrapper function
    solver = solver_function_org.SolverWrapper(
                                            train_proto_path,
                                            eval_proto_path,
                                            model_dir=exp_dir,
                                            config_path=config_path,
                                            prefix='pp',
                                            pretrain=pretrain_path,
                                            lr_policy= 'clr_full', #'clr',step
                                            warmup_step=936, # 936 how many steps to warmup (936=1epcoh for batch=4), debug=20
                                            warmup_start_lr=2e-4,
                                            lr_ratio=1e-3, # base_lr * lr_ratio = final_lr
                                            end_ratio=0.2, # 4640(batch4,clr) max_iter * end_ratio = drop_lr_steps
                                            base_lr=1e-4, # 0.0001, #fd 1e-7
                                            max_lr=1e-3, # use when in "clr/clr_full"
                                            momentum = 0.85, # 0.85 ///  0.9 for no momentum decay
                                            max_momentum = 0.95,#0.95, # 0 means with out momentum decay in clr
                                            gamma= 0.8, # 0.8 decay factor #fd 10
                                            stepsize=1, #11220, #learning rate decay step
                                            cycle_steps=9250, #9250(clr) debug=40
                                            max_iter=2320*13,#27840, # 23200(batch4,clr), 40700(batch8,clr), #69600(batch4,step), max_iter + warmup = total_step #debug=100
                                            test_interval=2320, #2320(clr) #3712(step),  #debug=100
                                            snapshot=2320,#2320 #3712
                                            solver_type='ADAM',
                                            weight_decay=0.0001, # 0.001,
                                            iter_size=1, #'number of mini-batches per iteration', batchsize*itersize = real_batch size
                                            display=50, #debug=1
                                            random_seed=0,
                                            args = args)
    return solver

def caffe_model(args, restore, pretrain_path, only_eval, demo, lr_finder):
    exp_dir = args['model_dir']
    config_path = args['config_path']
    # Set caffe to operate from GPU
    caffe.set_mode_gpu()
    caffe.set_device(0)
    # RPN pillar config
    # Define train and eval prototxt path
    train_proto_path = os.path.join(exp_dir, 'train.prototxt')
    eval_proto_path = os.path.join(exp_dir, 'eval.prototxt')
    # Model parameters

    #pillar rpn
    args["layer_nums"] = [3,5,5]
    args["num_filters"] = [64,128,256]
    args["layer_strides"] = [2,2,2]
    args["upsample_strides"] = [1, 2, 4]
    args["num_upsample_filters"] = [128, 128, 128]

    #secon rpn
    # args["layer_nums"] = [5]
    # args["num_filters"] = [128]
    # args["layer_strides"] = [2] #for ini_conv
    # args["upsample_strides"] = [None]
    # args["num_upsample_filters"] = [None]

    #for prior seg net
    args['feat_map_size'] = [1,32,24,400,352]#[1,16,80,200,176] #(B,C,N,H,W) [1,32,100,100,88], [1,32,10,400,352]
    args['point_cloud_range'] = [0, -40, -3, 70.4, 40, 1]
    args['seg_thresh'] = 0 #0.01
    args['use_depth'] = False
    args['use_score'] = False
    args['use_points'] = False
    args["seg_keep_points"] = 14000 #16000, 10000, 8000
    args["seg_keep_points_eval"] = 18000

    # NOTE: for voxel seg net
    args['max_voxels'] = 8000
    args['points_per_voxel'] = 200
    args["num_points_per_voxel"] = args['points_per_voxel']  #80
    args["bcl_keep_voxels"] = args['max_voxels'] #10000 for second, # 6500 for pillar
    args["bcl_keep_voxels_eval"] = args['max_voxels'] #10000 for second, # 6500 for pillar


    args["box_code_size"] = 7
    args["num_anchor_per_loc"] = 2 #1 for voxel_wise_seg
    args["num_cls"] = 1
    args["segmentation"] = True #True for segmentation
    args['anchors_cachae'] = True #True FOR Pillar, False For BCL

    _, eval_input_cfg, _, _ = load_config(exp_dir, config_path)
    args["eval_batch_size"] = eval_input_cfg.batch_size
    # Load model
    if args["segmentation"]:
        # train_net = seg_net.seg_object_detection(phase='train', dataset_params=args)
        # eval_net = seg_net.seg_object_detection(phase='eval', dataset_params=args)
        # train_net = prior_seg_net.seg_object_detection(phase='train', dataset_params=args)
        # eval_net = prior_seg_net.seg_object_detection(phase='eval', dataset_params=args)
        # train_net = voxel_seg_net.seg_object_detection(phase='train', dataset_params=args)
        # eval_net = voxel_seg_net.seg_object_detection(phase='eval', dataset_params=args)
        train_net = bcl_net.seg_object_detection(phase='train', dataset_params=args)
        eval_net = bcl_net.seg_object_detection(phase='eval', dataset_params=args)
    else:
        train_net = obj_net.seg_object_detection(phase='train', dataset_params=args)
        eval_net = obj_net.seg_object_detection(phase='eval', dataset_params=args)
        # train_net = bcl_model_v5.bilateral_baseline(phase='train', dataset_params=args)
        # eval_net = bcl_model_v5.bilateral_baseline(phase='eval', dataset_params=args)
    # Write proto to directory only in train mode
    if (not restore) and (not only_eval) and (not demo):
        # if os.path.isfile(train_proto_path):
        #     print("[ERROR] Train proto already exist")
        #     print("[ERROR] If re train please delete file")
        print("[WARNING] Writing Proto >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ")
        sut.write_proto([train_proto_path, train_net],[eval_proto_path, eval_net])
        sut.file_backup(exp_dir, config_path)

    if lr_finder:
        solver = Finder_LR_Solver(exp_dir, train_proto_path, eval_proto_path,
                                config_path, pretrain_path, args = args)
        solver.lr_finder()
        exit()
    # Load train solver
    solver = Train_Solver(exp_dir, train_proto_path, eval_proto_path,
                            config_path, pretrain_path, args = args)
    # Restore previous caffemodel to continue training
    if restore:
        try:
            train_proto_path = os.path.join(exp_dir, 'train.prototxt')
            print("[Info] Load prototxt from path :", train_proto_path)
        except Exception as e:
            print("\n[Info] Train prototxt not existing plz set retrain = False")
            exit()
        recent_model = sut.restore_latest_checkpoints(exp_dir)
        solver.solver.net.copy_from(os.path.join(exp_dir, "{}.caffemodel".format(str(recent_model))))
        solver.solver.restore(os.path.join(exp_dir, "{}.solverstate".format(str(recent_model))))
    if only_eval:
        solver.eval_model()
        print("[info] End of Evaluation")
        exit()
    if demo:
        solver.demo()
        print("[info] End of demo")
        exit()
    solver.train_model()

def dual_model_eval(seg_path, seg_weight, obj_path, obj_weight, model_dir, config_path):
    caffe.set_mode_gpu()
    caffe.set_device(0)
    eval_wrapper = DualEvalWrapper(seg_path, seg_weight, obj_path, obj_weight,
                                    model_dir, config_path)
    eval_wrapper.evaluation()
    exit()

def train(config_path, model_dir, restore=False, only_eval=False, demo=False, lr_finder=False):
    args = {}
    args['config_path'] = config_path
    args['model_dir'] = model_dir
    ## Test dual model eval
    # obj_path = "./exp/Debug_BaseLine"
    # obj_weight = "pp_iter_296960.caffemodel"
    # seg_path = "./exp/FM_Seg_V4"
    # seg_weight = "pp_iter_42688.caffemodel"
    # dual_model_eval(seg_path, seg_weight, obj_path, obj_weight, model_dir, config_path)
    # If need to place weight to pretrained model, enter path and weight name
    # If any is None, pretrain will not load
    pretrain_path = {}
    pretrain_path['path'] = None #"./exp/FM_Seg_V11" #"./exp/LEO_BaseLine_points_8000"
    pretrain_path['weight'] = None #"pp_iter_25984.caffemodel" #"pp_iter_35264.caffemodel"
    pretrain_path['skip_layer'] = [None] #[None], ['cls_head','reg_head']
    sut.create_model_folder(model_dir, config_path)
    caffe_model(args, restore, pretrain_path, only_eval, demo, lr_finder)

if __name__ == '__main__':
    fire.Fire()
