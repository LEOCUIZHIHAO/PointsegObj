import os
from pathlib import Path

#create experiment folder
def create_model_folder(model_dir, config_path):
    model_dir = str(Path(model_dir).resolve())
    if Path(model_dir).exists():
       print("[INFO] File already exits")
    _model_dir = Path(model_dir)
    _model_dir.mkdir(parents=True, exist_ok=True)

#write proto
def write_proto(*args):
    """
    arg : [proto1, net1],[proto2, net2]
    """
    for arg in args:
        with open(arg[0], 'w') as f:
            print(arg[1], file=f)

#restore lastest checkpoint
def restore_latest_checkpoints(exp_dir):
    maxit = 0
    file = os.listdir(exp_dir)
    caffemodel = [os.path.splitext(model)[0] for model in file if model.endswith('.caffemodel')]
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
        print("\n[Info] Load existing model >>>>>>>>>>>>>>>>>>>>>>{}\n".format(str(exp_dir+recent_model)))
    return recent_model

#clear caffe_model on time
def clear_caffemodel(exp_dir, max_keep=12):
    file = os.listdir(exp_dir)
    caffemodel = [os.path.splitext(model)[0] for model in file if model.endswith('.caffemodel')]
    key = [int(model.split('_')[-1]) for model in caffemodel]
    if len(caffemodel)>max_keep:
        model_keys=dict(zip(key, caffemodel))
        keys = model_keys.keys()
        skeys=sorted(keys)
        model_ordered=[]
        for k in skeys:
            model_ordered.append(model_keys[k])
        remove_list = model_ordered[:len(model_ordered)-max_keep]

        for r in remove_list:
            try:
                os.remove(os.path.join(exp_dir, r + ".caffemodel"))
            except Exception as e:
                print("No such caffemodel")
            try:
                os.remove(os.path.join(exp_dir, r + ".solverstate"))
            except Exception as e:
                print("No such solverstate")
