import os
import fire

import caffe
from caffe import layers as L, params as P
from models import bcl_model_v2
from solver import solver_function

import shutil
from pathlib import Path
from google.protobuf import text_format
from second.protos import pipeline_pb2
import torchplus

def load_model_config(model_dir, config_path):
    model_dir = str(Path(model_dir).resolve())
    if Path(model_dir).exists():
        model_dir = torchplus.train.create_folder(model_dir)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

def load_recent_model(exp_dir):
    # try
    #     exp_dir:
    # except Exception as e:
    #     pirnt("ple enter model dir")
    #   exit()
    maxit = 0
    file = os.listdir(exp_dir)
    print(file)
    caffemodel = [os.path.splitext(model)[0] for model in file if model.endswith('.caffemodel')]
    print(caffemodel)
    if len(caffemodel)==0:
        print("\n[Info] No model existing please retrain")
        exit()
    for idx, model in enumerate(caffemodel):
        ite=int(model.split('_')[-1])
        if ite>maxit:
            maxit=ite
            maxid=idx
    recent_model = caffemodel[maxid]
    if (str(recent_model) + '.solverstate'):
        print("\n[Info] Load existing model {}\n".format(str(exp_dir+recent_model)))
    return recent_model

def Train_Solver(exp_dir, trian_proto_path, pretrained, log_path):
    solver = solver_function.TrainSolverWrapper(trian_proto_path,
                                            os.path.join(exp_dir, 'pp'),
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
                                            iter_size=1, #'number of mini-batches per iteration', batchsize*itersize = real_batch size
                                            display=50,
                                            debug_info=False,
                                            random_seed=19930416,
                                            save_path=os.path.join(exp_dir, 'solver.prototxt'),
                                            log_path=log_path)
    # solver = solver_function.SolverWrapper(trian_proto_path,
    #                                         os.path.join(exp_dir, 'pp'),
    #                                         pretrained=pretrained,
    #                                         base_lr= 0.0002,
    #                                         gamma= 0.8, #decay factor
    #                                         stepsize= 27840, #learning rate decay
    #                                         test_iter= 3769, # 10 #number of iterations to use at each testing phase 3769
    #                                         test_interval= 999999999, # 'test every such iterations' 1856 (test every 5 epoches) 9280
    #                                         max_iter= 296960, # 296960 = 160*1856 #185600
    #                                         snapshot=1856, # how many steps save a model 9280 (1856*2=3712) save 2 epoches
    #                                         solver_type='ADAM',
    #                                         weight_decay= 0.0001, # 0.0001,
    #                                         iter_size=1, #'number of mini-batches per iteration', batchsize*itersize = real_batch size
    #                                         display=50,
    #                                         debug_info=False,
    #                                         random_seed=19930416,
    #                                         save_path=os.path.join(exp_dir, 'solver.prototxt'),
    #                                         log_path=log_path)
    return solver
def caf_model(exp_dir, args, restore, pretrained, start_eval, visual_log):

    caffe.set_mode_gpu()
    caffe.set_device(0)

    if visual_log==True:
        log_path =os.path.join(exp_dir, 'visual_log')
    else:
        log_path =None

    # if (not restore) and (not start_eval):
    if (not start_eval):
        trian_proto_path = os.path.join(exp_dir, 'train.prototxt')
        eval_proto_path = os.path.join(exp_dir, 'eval.prototxt')
        # deploy_proto_path = os.path.join(exp_dir, 'deploy.prototxt')

        train_net = bcl_model_v2.bilateral_baseline(phase='train', dataset_params=args)
        eval_net = bcl_model_v2.bilateral_baseline(phase='eval', dataset_params=args)

        #train_net = bcl_model.bilateral_baseline(phase='train', dataset_params=args)
        #eval_net = bcl_model.bilateral_baseline(phase='eval', dataset_params=args)

        with open(trian_proto_path, 'w') as f:
            print(train_net, file=f)
        with open(eval_proto_path, 'w') as f:
            print(eval_net, file=f)

        solver = Train_Solver(exp_dir, trian_proto_path, pretrained, log_path)
        # solver = SolverWrapper(exp_dir, trian_proto_path, pretrained, log_path)

        if restore:
            try:
                trian_proto_path = os.path.join(exp_dir, 'train.prototxt')
                print("[info] Load prototxt from path :", trian_proto_path)
            except Exception as e:
                print("\n[Info] Train prototxt not existing")
                exit()
            # solver = Train_Solver(exp_dir, trian_proto_path, pretrained, log_path)
            recent_model = load_recent_model(exp_dir)
            _solver = solver.load_solver()
            _solver.net.copy_from(os.path.join(exp_dir, "{}.caffemodel".format(str(recent_model))))
            _solver.restore(os.path.join(exp_dir, "{}.solverstate".format(str(recent_model))))

    if start_eval:
        try:
            eval_proto_path = os.path.join(exp_dir, 'eval.prototxt')
            print("[info] Load prototxt from path :", eval_proto_path)
        except Exception as e:
            print("\n[Info] Eval prototxt not existing")
            exit()

        recent_model = load_recent_model(exp_dir)
        eval_weight_path=os.path.join(exp_dir, "{}.caffemodel".format(str(recent_model)))
        solver = solver_function.EvalSolverWrapper( eval_proto_path,
                                                    eval_weight_path,
                                                    log_path=log_path)

    start_eval_v2=False
    if start_eval_v2:
        model_dir = os.path.join(exp_dir)
        trian_proto_path = os.path.join(exp_dir, 'train.prototxt')
        eval_proto_path = os.path.join(exp_dir, 'eval.prototxt')
        solver = solver_function.SolverWrapperTest(
                                                trian_proto_path,
                                                eval_proto_path,
                                                os.path.join(exp_dir, 'pp'),
                                                pretrained=pretrained,
                                                test_iter= 3769, # 10 #number of iterations to use at each testing phase 3769
                                                test_interval= 999999999, # 'test every such iterations' 1856 (test every 5 epoches) 9280
                                                iter_size=1, #'number of mini-batches per iteration', batchsize*itersize = real_batch size
                                                save_path=os.path.join(exp_dir, 'solver.prototxt'),
                                                log_path=log_path,
                                                model_dir = args['model_dir'],
                                                config_path = args['config_path'])

        print("[INFO], exp_dir PATH : " , exp_dir)

        # recent_model = load_recent_model(exp_dir)
        # _solver = solver.load_solver()
        # print(_solver.net)
        # _solver.net.copy_from(os.path.join(exp_dir, "{}.caffemodel".format(str(recent_model))))
        # print(_solver.net)
        # solver.solver = _solver
        # exit()
        solver.solver.net.copy_from(os.path.join(exp_dir, "{}.caffemodel".format(str(recent_model))))
    solver.train_model()

def train(config_path, model_dir, restore=False, pretrained=False, start_eval=False, visual_log=False):

    args = {}
    args['config_path'] = config_path
    args['model_dir'] = model_dir
    load_model_config(model_dir, config_path)
    caf_model(model_dir, args, restore, pretrained, start_eval, visual_log)

if __name__ == '__main__':
    fire.Fire()
