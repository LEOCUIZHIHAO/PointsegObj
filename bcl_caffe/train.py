import os
import fire
import caffe
from caffe import layers as L, params as P
from models import bcl_model_v2, bcl_model_v3, bcl_model_v4, seg_obj
from solver import solver_function
from tools import some_useful_tools as sut


def Train_Solver(exp_dir, trian_proto_path, pretrained, visual_log):
    if visual_log==True:
        log_path =os.path.join(exp_dir, 'visual_log')
    else:
        log_path=None
    solver = solver_function.TrainSolverWrapper(
                                            trian_proto_path,
                                            save_path = exp_dir,
                                            prefix = 'pp',
                                            pretrained=pretrained,
                                            base_lr= 0.0002,
                                            gamma= 0.8, #decay factor
                                            stepsize= 27840, #learning rate decay
                                            test_iter= 3769, # 10 #number of iterations to use at each testing phase 3769
                                            test_interval= 999999999, # 'test every such iterations' 1856 (test every 5 epoches) 9280
                                            max_iter= 296960, # 296960 = 160*1856 #185600
                                            snapshot=1856, # how many steps save a model 9280 (1856*2=3712) save 2 epoches
                                            solver_type='ADAM',
                                            weight_decay= 0.0001, # 0.0001,
                                            iter_size=2, #'number of mini-batches per iteration', batchsize*itersize = real_batch size
                                            display=50,
                                            random_seed=0,
                                            log_path=log_path)
    return solver

def Eval_Solver(exp_dir, config_path, trian_proto_path, eval_proto_path, pretrained):
    solver = solver_function.SolverWrapperTest(
                                            trian_proto_path,
                                            eval_proto_path,
                                            os.path.join(exp_dir, 'pp'),
                                            pretrained=pretrained,
                                            test_iter= 3769, # 10 #number of iterations to use at each testing phase 3769
                                            test_interval= 999999999, # 'test every such iterations' 1856 (test every 5 epoches) 9280
                                            iter_size=1, #'number of mini-batches per iteration', batchsize*itersize = real_batch size
                                            save_path=os.path.join(exp_dir, 'solver.prototxt'),
                                            model_dir = exp_dir,
                                            config_path = config_path)

    return solver

def caffe_model(args, restore, pretrained, start_eval, visual_log):
    exp_dir = args['model_dir']
    config_path = args['config_path']
    caffe.set_mode_gpu()
    caffe.set_device(0)

    trian_proto_path = os.path.join(exp_dir, 'train.prototxt')
    eval_proto_path = os.path.join(exp_dir, 'eval.prototxt')

    if not start_eval:

        train_net = seg_obj.seg_object_detection(phase='train', dataset_params=args)
        eval_net = seg_obj.seg_object_detection(phase='eval', dataset_params=args)
        # train_net = bcl_model_v3.bilateral_baseline(phase='train', dataset_params=args)
        # eval_net = bcl_model_v3.bilateral_baseline(phase='eval', dataset_params=args)

        sut.write_proto([trian_proto_path, train_net],[eval_proto_path, eval_net])

        solver = Train_Solver(exp_dir, trian_proto_path, pretrained, visual_log)

        if restore:
            try:
                trian_proto_path = os.path.join(exp_dir, 'train.prototxt')
                print("[info] Load prototxt from path :", trian_proto_path)
            except Exception as e:
                print("\n[Info] Train prototxt not existing plz set retrain = False")
                exit()
            recent_model = sut.restore_latest_checkpoints(exp_dir)
            _solver = solver.load_solver()
            _solver.net.copy_from(os.path.join(exp_dir, "{}.caffemodel".format(str(recent_model))))
            _solver.restore(os.path.join(exp_dir, "{}.solverstate".format(str(recent_model))))

    if start_eval:
        try:
            solver = Eval_Solver(exp_dir, config_path, trian_proto_path, eval_proto_path, pretrained)
            recent_model = sut.restore_latest_checkpoints(exp_dir)
            solver.solver.net.copy_from(os.path.join(exp_dir, "{}.caffemodel".format(str(recent_model))))
        except Exception as e:
            print("\n[Info] Ckeck Train prototxt if not existing plz start_eval = False ")
            print("\n[Info] Ckeck SetUp DataInput shape! ")
            print("\n[Info] Ckeck check DataLoader generate_anchors_cachae! ")
            raise
            exit()

    solver.train_model()

def train(config_path, model_dir, restore=False, pretrained=False, start_eval=False, visual_log=False):
    args = {}
    args['config_path'] = config_path
    args['model_dir'] = model_dir
    sut.create_model_folder(model_dir, config_path)
    caffe_model(args, restore, pretrained, start_eval, visual_log)

if __name__ == '__main__':
    fire.Fire()



################################################################################
# if start_eval:
#     try:
#         eval_proto_path = os.path.join(exp_dir, 'eval.prototxt')
#         print("[info] Load prototxt from path :", eval_proto_path)
#     except Exception as e:
#         print("\n[Info] Eval prototxt not existing")
#         exit()
#
#     recent_model = restore_latest_checkpoints(exp_dir)
#     eval_weight_path=os.path.join(exp_dir, "{}.caffemodel".format(str(recent_model)))
#     solver = solver_function.EvalSolverWrapper( eval_proto_path,
#                                                 eval_weight_path,
#                                                 log_path=log_path)
################################################################################
