import os
import fire
import caffe
from caffe import layers as L, params as P
from models import bcl_model_v2, bcl_model_v3, bcl_model_v4, seg_obj, seg_net, bcl_model_v5
from solver import solver_function_org
from tools import some_useful_tools as sut


def Train_Solver(exp_dir, train_proto_path, eval_proto_path, config_path,
                        pretrain_path, visual_log, args):
    if visual_log==True:
        log_path =os.path.join(exp_dir, 'visual_log')
    else:
        log_path=None
    # Load solver wrapper function
    solver = solver_function_org.SolverWrapper(
                                            train_proto_path,
                                            eval_proto_path,
                                            model_dir=exp_dir,
                                            config_path=config_path,
                                            save_path=exp_dir,
                                            prefix='pp',
                                            pretrain=pretrain_path,
                                            base_lr=0.0002,
                                            gamma=0.8, #decay factor
                                            stepsize=18560, #learning rate decay
                                            test_iter=3769, # 10 #number of iterations to use at each testing phase 3769
                                            test_interval=7424, # 'test every such iterations' 1856 (test every 5 epoches) 9280
                                            max_iter=296960, # 296960 = 160*1856 #185600
                                            snapshot=1856, # how many steps save a model 9280 (1856*2=3712) save 2 epoches
                                            solver_type='ADAM',
                                            weight_decay=0.001, # 0.0001,
                                            iter_size=2, #'number of mini-batches per iteration', batchsize*itersize = real_batch size
                                            display=50,
                                            random_seed=0,
                                            log_path=log_path,
                                            args = args)
    return solver

def caffe_model(args, restore, pretrain_path, visual_log):
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
    args["num_filters"] = [64,128,256]
    args["layer_strides"] = [2,2,2]
    args["layer_nums"] = [3,5,5]
    args["upsample_strides"] = [1, 2, 4]
    args["num_upsample_filters"] = [128, 128, 128]
    args["anchors_fp_w"] = 352 #432 #1408
    args["anchors_fp_h"] = 400 #496 #1600

    args["num_points_per_voxel"] = 80
    args["bcl_keep_voxels"] = 6500
    args["seg_keep_points"] = 8000

    args["bcl_keep_voxels_eval"] = 6500
    args["seg_keep_points_eval"] = 18800
    args["box_code_size"] = 7
    args["num_anchor_per_loc"] = 2
    args["num_cls"] = 1
    args["segmentation"] = False #True for segmentation
    # Load model
    # train_net = seg_net.seg_object_detection(phase='train', dataset_params=args)
    # eval_net = seg_net.seg_object_detection(phase='eval', dataset_params=args)
    # train_net = seg_obj.seg_object_detection(phase='train', dataset_params=args)
    # eval_net = seg_obj.seg_object_detection(phase='eval', dataset_params=args)
    train_net = bcl_model_v5.bilateral_baseline(phase='train', dataset_params=args)
    eval_net = bcl_model_v5.bilateral_baseline(phase='eval', dataset_params=args)
    # Write proto to directory
    sut.write_proto([train_proto_path, train_net],[eval_proto_path, eval_net])
    # Load train solver
    solver = Train_Solver(exp_dir, train_proto_path, eval_proto_path,
                            config_path, pretrain_path, visual_log, args = args)
    # Restore previous caffemodel to continue training
    if restore:
        try:
            train_proto_path = os.path.join(exp_dir, 'train.prototxt')
            print("[info] Load prototxt from path :", train_proto_path)
        except Exception as e:
            print("\n[Info] Train prototxt not existing plz set retrain = False")
            exit()
        recent_model = sut.restore_latest_checkpoints(exp_dir)
        solver.solver.net.copy_from(os.path.join(exp_dir, "{}.caffemodel".format(str(recent_model))))
        solver.solver.restore(os.path.join(exp_dir, "{}.solverstate".format(str(recent_model))))
    solver.train_model()

def train(config_path, model_dir, restore=True, visual_log=False):
    args = {}
    args['config_path'] = config_path
    args['model_dir'] = model_dir
    # If need to place weight to pretrained model, enter path and weight name
    # If any is None, pretrain will not load
    pretrain_path = {}
    pretrain_path['path'] = None #"./exp/LEO_BaseLine_points_8000"
    pretrain_path['weight'] = None #"pp_iter_35264.caffemodel"
    sut.create_model_folder(model_dir, config_path)
    caffe_model(args, restore, pretrain_path, visual_log)

if __name__ == '__main__':
    fire.Fire()
