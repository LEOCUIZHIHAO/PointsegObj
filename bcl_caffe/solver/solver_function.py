import os
import tempfile
import numpy as np
from pathlib import Path
import time
import torch
import caffe
from caffe.proto import caffe_pb2
from tqdm import tqdm
from visualdl import LogWriter

from google.protobuf import text_format
import second.data.kitti_common as kitti
from second.builder import target_assigner_builder, voxel_builder
from second.pytorch.core import box_torch_ops
from second.data.preprocess import merge_second_batch, merge_second_batch_multigpu
from second.protos import pipeline_pb2
from second.pytorch.builder import box_coder_builder, input_reader_builder
from second.pytorch.models.voxel_encoder import get_paddings_indicator_np #for pillar
from second.utils.log_tool import SimpleModelLog
from tools import some_useful_tools as sut

def get_prototxt(solver_proto, save_path=None):
    if save_path:
        f = open(save_path, mode='w+')
    else:
        f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    f.write(str(solver_proto))
    f.close()

    return f.name

class TrainSolverWrapper:
    def __init__(self,  train_net,
                        save_path,
                        prefix,
                        pretrained,
                        solver_type='ADAM',
                        weight_decay=0.001,
                        base_lr=0.002,
                        gamma=0.8, #0.1 for lr_policy
                        stepsize=100,
                        test_iter=3769,
                        test_interval=50, #set test_interval to 999999999 if not it will auto run validation
                        max_iter=1e5,
                        iter_size=1,
                        snapshot=1000,
                        display=1,
                        random_seed=0,
                        debug_info=False,
                        create_prototxt=True,
                        log_path=None):
        """Initialize the SolverWrapper."""
        self.solver_param = caffe_pb2.SolverParameter()
        self.solver_param.train_net = train_net

        self.solver_param.test_initialization = False

        self.solver_param.base_lr = base_lr
        self.solver_param.lr_policy = 'step'  # "fixed" #exp
        self.solver_param.gamma = gamma
        self.solver_param.stepsize = stepsize

        self.solver_param.display = display
        self.solver_param.max_iter = max_iter
        self.solver_param.iter_size = iter_size
        self.solver_param.snapshot = snapshot
        self.solver_param.snapshot_prefix = os.path.join(save_path, prefix)
        self.solver_param.random_seed = random_seed

        self.solver_param.solver_mode = caffe_pb2.SolverParameter.GPU
        if solver_type is 'SGD':
            self.solver_param.solver_type = caffe_pb2.SolverParameter.SGD
        elif solver_type is 'ADAM':
            self.solver_param.solver_type = caffe_pb2.SolverParameter.ADAM
        self.solver_param.momentum = 0.9
        self.solver_param.momentum2 = 0.999

        self.solver_param.weight_decay = weight_decay
        self.solver_param.debug_info = debug_info

        if create_prototxt:
            solver_prototxt = get_prototxt(self.solver_param, os.path.join(save_path, 'solver.prototxt'))
            print(solver_prototxt)

        self.solver = caffe.get_solver(solver_prototxt)

        self.pretrained = pretrained
        self.test_interval = 1856*2 #1856  #replace self.solver_param.test_interval #9280
        self.save_path = save_path

        self.log_path = log_path
        if self.log_path is not None:
            self.logw = LogWriter(self.log_path, sync_cycle=100)
            with self.logw.mode('train') as logger:
                self.sc_train_reg_loss = logger.scalar("reg_loss")
                self.sc_train_cls_loss = logger.scalar("cls_loss")

    def load_solver(self):
        return self.solver

    def train_model(self):
        if self.pretrained:
            print("\n[info] Load Pretrained Model\n")
            # self.caffe_load_weight()
            self.load_pretrained_weight()

        cur_iter = 0
        while cur_iter < self.solver_param.max_iter:
            for i in range(self.test_interval):
                self.solver.step(1)
                if self.log_path is not None:
                    step = self.solver.iter
                    reg_loss = self.solver.net.blobs['reg_loss'].data
                    cls_loss = self.solver.net.blobs['cls_loss'].data

                    self.sc_train_reg_loss.add_record(step, reg_loss)
                    self.sc_train_cls_loss.add_record(step, cls_loss) # for logger

            #always keep top 12 caffemodel
            sut.clear_caffemodel(self.save_path, 12)
            cur_iter += self.test_interval

    def caffe_load_weight(self):

        exp_dir = "./exp/Bilateral_BaseLine_Debug/"
        weight_name = "pp_iter_139200"

        weight_path = os.path.join(exp_dir + weight_name)
        net_ = caffe.Net(os.path.join(exp_dir + "train.prototxt"), "{}.caffemodel".format(weight_path), caffe.TRAIN)

        def rpn_layer(layer_name):
            self.solver.net.params[str(layer_name[0])][0].data[...] = net_.params[str(layer_name[0])][0].data[...]#w

            self.solver.net.params[str(layer_name[1])][0].data[...] = net_.params[str(layer_name[1])][0].data[...]#w
            self.solver.net.params[str(layer_name[1])][1].data[...] = net_.params[str(layer_name[1])][1].data[...]#b

            self.solver.net.params[str(layer_name[2])][0].data[...] = net_.params[str(layer_name[2])][0].data[...]#mean
            self.solver.net.params[str(layer_name[2])][1].data[...] = net_.params[str(layer_name[2])][1].data[...]#var
            self.solver.net.params[str(layer_name[2])][2].data[...] = net_.params[str(layer_name[2])][2].data[...]

        ##################################block1################################
        caffe_layre = ['ini_conv1_0', 'ini_conv1_sc_0', 'ini_conv1_bn_0']
        rpn_layer(caffe_layre)

        for idx in range(3):
            caffe_layre = ['rpn_conv1_{}'.format(idx), 'rpn_conv1_sc_{}'.format(idx), 'rpn_conv1_bn_{}'.format(idx)]
            rpn_layer(caffe_layre)

        caffe_layre = ['rpn_deconv1', 'rpn_deconv1_sc', 'rpn_deconv1_bn']
        rpn_layer(caffe_layre)

        ##################################block2################################
        caffe_layre = ['ini_conv2_0', 'ini_conv2_sc_0', 'ini_conv2_bn_0']
        rpn_layer(caffe_layre)


        for idx in range(5):
            caffe_layre = ['rpn_conv2_{}'.format(idx), 'rpn_conv2_sc_{}'.format(idx), 'rpn_conv2_bn_{}'.format(idx)]
            rpn_layer(caffe_layre)

        caffe_layre = ['rpn_deconv2', 'rpn_deconv2_sc', 'rpn_deconv2_bn']
        rpn_layer(caffe_layre)

        ##################################block3################################
        caffe_layre = ['ini_conv3_0', 'ini_conv3_sc_0', 'ini_conv3_bn_0']
        rpn_layer(caffe_layre)

        for idx in range(5):
            caffe_layre = ['rpn_conv3_{}'.format(idx), 'rpn_conv3_sc_{}'.format(idx), 'rpn_conv3_bn_{}'.format(idx)]
            rpn_layer(caffe_layre)

        caffe_layre = ['rpn_deconv3', 'rpn_deconv3_sc', 'rpn_deconv3_bn']
        rpn_layer(caffe_layre)

        ################################# Head ################################

        self.solver.net.params['cls_head'][0].data[...] = net_.params['cls_head'][0].data[...]
        self.solver.net.params['cls_head'][1].data[...] = net_.params['cls_head'][1].data[...]
        self.solver.net.params['reg_head'][0].data[...] = net_.params['reg_head'][0].data[...]
        self.solver.net.params['reg_head'][1].data[...] = net_.params['reg_head'][1].data[...]
    """load pytourch weight"""
    def prepare_pretrained(self, weights_path, layer_name):
        weights = os.listdir(weights_path)

        graph = [w for w in weights if layer_name in w]
        layer_key = [int(m.split('_')[0]) for m in graph]
        layer_dict = dict(zip(layer_key, graph))

        keys = layer_dict.keys()
        keys = sorted(keys)
        layer_ordered = []
        for k in keys:
            layer_ordered.append(layer_dict[k])
        # print(layer_dict)
        # print(layer_ordered) #debug
        return layer_ordered
    def load_pretrained_weight(self):

        weights_path = '/home/ubuntu/second.pytorch/second/output/model_weights/'
        layer_name = 'voxel_feature'
        mlp_Layer = self.prepare_pretrained(weights_path, layer_name)

        self.solver.net.params['mlp_0'][0].data[...] = np.load(weights_path + mlp_Layer[0]) #w
        self.solver.net.params['mlp_sc_0'][0].data[...] = np.load(weights_path + mlp_Layer[1]) #w
        self.solver.net.params['mlp_sc_0'][1].data[...] = np.load(weights_path + mlp_Layer[2]) #b
        self.solver.net.params['mlp_bn_0'][0].data[...] = np.load(weights_path + mlp_Layer[3]) #mean
        self.solver.net.params['mlp_bn_0'][1].data[...] = np.load(weights_path + mlp_Layer[4]) #var
        self.solver.net.params['mlp_bn_0'][2].data[...] = 1

        def rpn_layer(layer_name, block, idx, stride):
            self.solver.net.params[str(layer_name[0])][0].data[...] = np.load(weights_path + block[idx*stride]) #w

            self.solver.net.params[str(layer_name[1])][0].data[...] = np.load(weights_path + block[idx*stride+1]) #w
            self.solver.net.params[str(layer_name[1])][1].data[...] = np.load(weights_path + block[idx*stride+2]) #b

            self.solver.net.params[str(layer_name[2])][0].data[...] = np.load(weights_path + block[idx*stride+3]) #mean
            self.solver.net.params[str(layer_name[2])][1].data[...] = np.load(weights_path + block[idx*stride+4]) #var
            self.solver.net.params[str(layer_name[2])][2].data[...] = 1

        ##################################block1################################
        layer_name = 'rpn.blocks.0'
        rpn_block = self.prepare_pretrained(weights_path, layer_name)
        # print(rpn_block)
        caffe_layre = ['ini_conv1_0', 'ini_conv1_sc_0', 'ini_conv1_bn_0']
        rpn_layer(caffe_layre, rpn_block, idx=0, stride=5)

        for idx in range(3):
            caffe_layre = ['rpn_conv1_{}'.format(idx), 'rpn_conv1_sc_{}'.format(idx), 'rpn_conv1_bn_{}'.format(idx)]
            rpn_layer(caffe_layre, rpn_block, idx=idx+1, stride=5)

        layer_name = 'rpn.deblocks.0'
        rpn_deconv = self.prepare_pretrained(weights_path, layer_name)
        # print(rpn_deconv)
        caffe_layre = ['rpn_deconv1', 'rpn_deconv1_sc', 'rpn_deconv1_bn']
        rpn_layer(caffe_layre, rpn_deconv, idx=0, stride=5)



        ##################################block2################################
        layer_name = 'rpn.blocks.1'
        rpn_block = self.prepare_pretrained(weights_path, layer_name)
        # print(rpn_block)
        caffe_layre = ['ini_conv2_0', 'ini_conv2_sc_0', 'ini_conv2_bn_0']
        rpn_layer(caffe_layre, rpn_block, idx=0, stride=5)

        for idx in range(5):
            caffe_layre = ['rpn_conv2_{}'.format(idx), 'rpn_conv2_sc_{}'.format(idx), 'rpn_conv2_bn_{}'.format(idx)]
            rpn_layer(caffe_layre, rpn_block, idx=idx+1, stride=5)

        layer_name = 'rpn.deblocks.1'
        rpn_deconv = self.prepare_pretrained(weights_path, layer_name)
        # print(rpn_deconv)
        caffe_layre = ['rpn_deconv2', 'rpn_deconv2_sc', 'rpn_deconv2_bn']
        rpn_layer(caffe_layre, rpn_deconv, idx=0, stride=5)


        ##################################block3################################
        layer_name = 'rpn.blocks.2'
        rpn_block = self.prepare_pretrained(weights_path, layer_name)
        # print(rpn_block)
        caffe_layre = ['ini_conv3_0', 'ini_conv3_sc_0', 'ini_conv3_bn_0']
        rpn_layer(caffe_layre, rpn_block, idx=0, stride=5)

        for idx in range(5):
            caffe_layre = ['rpn_conv3_{}'.format(idx), 'rpn_conv3_sc_{}'.format(idx), 'rpn_conv3_bn_{}'.format(idx)]
            rpn_layer(caffe_layre, rpn_block, idx=idx+1, stride=5)

        layer_name = 'rpn.deblocks.2'
        rpn_deconv = self.prepare_pretrained(weights_path, layer_name)
        # print(rpn_deconv)
        caffe_layre = ['rpn_deconv3', 'rpn_deconv3_sc', 'rpn_deconv3_bn']
        rpn_layer(caffe_layre, rpn_deconv, idx=0, stride=5)

        ################################# Head ################################
        layer_name = 'rpn.conv'
        rpn_conv = self.prepare_pretrained(weights_path, layer_name)
        self.solver.net.params['cls_head'][0].data[...] = np.load(weights_path + rpn_conv[0]) #w cls_head
        self.solver.net.params['cls_head'][1].data[...] = np.load(weights_path + rpn_conv[1]) #b cls_head
        self.solver.net.params['reg_head'][0].data[...] = np.load(weights_path + rpn_conv[2]) #w box_head
        self.solver.net.params['reg_head'][1].data[...] = np.load(weights_path + rpn_conv[3]) #b box_head


def build_network(model_cfg, measure_time=False):
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)
    # box_coder.custom_ndim = target_assigner._anchor_generators[0].custom_ndim

    return voxel_generator, target_assigner

def _worker_init_fn(worker_id):
    time_seed = np.array(time.time(), dtype=np.int32)
    np.random.seed(time_seed + worker_id)
    print(f"WORKER {worker_id} seed:", np.random.get_state()[1][0])

def load_config(model_dir, config_path):
    model_dir = str(Path(model_dir).resolve())
    model_dir = Path(model_dir)
    config_file_bkp = "pipeline.config"
    if isinstance(config_path, str):
        # directly provide a config object. this usually used
        # when you want to train with several different parameters in
        # one script.
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = config_path
        proto_str = text_format.MessageToString(config, indent=2)
    with (model_dir / config_file_bkp).open("w") as f:
        f.write(proto_str)

    input_cfg = config.train_input_reader
    eval_input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config


    return (input_cfg, eval_input_cfg, model_cfg, train_cfg)

def log_function(model_dir, config_path):
    model_logging = SimpleModelLog(model_dir)
    model_logging.open()

    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)

    model_logging.log_text(proto_str + "\n", 0, tag="config")
    return model_logging

def load_dataloader(eval_input_cfg, model_cfg, voxel_generator, target_assigner,generate_anchors_cachae):
    eval_dataset = input_reader_builder.build(
        eval_input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner,
        generate_anchors_cachae=generate_anchors_cachae) #True FOR Pillar, False For BCL

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_input_cfg.batch_size, # only support multi-gpu train
        shuffle=False,
        num_workers=eval_input_cfg.preprocess.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch)

    return eval_dataloader, eval_dataset

def example_convert_to_torch(example, dtype=torch.float32,
                             device=None) -> dict:
    device = device or torch.device("cuda:0")
    example_torch = {}
    float_names = [
        "voxels", "anchors", "reg_targets", "reg_weights", "bev_map", "importance"
    ]
    for k, v in example.items():
        if k in float_names:
            # slow when directly provide fp32 data with dtype=torch.half
            example_torch[k] = torch.tensor(
                v, dtype=torch.float32, device=device).to(dtype)
        elif k in ["coordinates", "labels", "num_points"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.int32, device=device)
        elif k in ["anchors_mask"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.uint8, device=device)
        elif k == "calib":
            calib = {}
            for k1, v1 in v.items():
                calib[k1] = torch.tensor(
                    v1, dtype=dtype, device=device).to(dtype)
            example_torch[k] = calib
        elif k == "num_voxels":
            example_torch[k] = torch.tensor(v)
        elif k in ["box_preds", "cls_preds"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.float32, device=device).to(dtype)
        else:
            example_torch[k] = v
    return example_torch

class SolverWrapperTest:
    def __init__(self,  train_net,
                        test_net,
                        prefix,
                        pretrained,
                        test_iter=3769,
                        test_interval=999999999, #set test_interval to 999999999 if not it will auto run validation
                        iter_size=1,
                        create_prototxt=True,
                        save_path=None,
                        model_dir=None,
                        config_path=None,):
        """Initialize the SolverWrapper."""
        self.solver_param = caffe_pb2.SolverParameter()
        self.solver_param.train_net = train_net
        self.solver_param.test_net.extend([test_net])

        self.solver_param.test_iter.extend([test_iter])
        self.solver_param.test_interval = test_interval
        self.solver_param.test_initialization = False
        self.solver_param.iter_size = iter_size

        if create_prototxt:
            solver_prototxt = get_prototxt(self.solver_param, save_path)
            print(solver_prototxt)

        self.solver = caffe.get_solver(solver_prototxt)
        self.pretrained = pretrained
        self.model_dir = model_dir
        self.config_path = config_path

    def load_solver(self):
        return self.solver
    def train_model(self):
        caffe.set_mode_gpu()
        caffe.set_device(0)
        # self.load_pretrained_weight() #mannully load weights''='
        # self.caffe_load_weight()
        self.eval_on_val()
    def eval_on_val(self):
        self.solver.test_nets[0].share_with(self.solver.net)
        _, eval_input_cfg, model_cfg, train_cfg = load_config(self.model_dir, self.config_path)
        voxel_generator, self.target_assigner = build_network(model_cfg)
        dataloader, eval_dataset = load_dataloader(eval_input_cfg, model_cfg, voxel_generator, self.target_assigner, generate_anchors_cachae=False) #True FOR Pillar, False For BCL
        model_logging = log_function(self.model_dir, self.config_path)
        data_iter=iter(dataloader)

        self._box_coder=self.target_assigner.box_coder
        self._num_class=1
        self._encode_background_as_zeros =True
        self._nms_class_agnostic=False
        self._class_name="Car"
        self._use_multi_class_nms=False
        self._nms_pre_max_sizes=[1000]
        self._multiclass_nms=False
        self._use_sigmoid_score=True
        self._num_anchor_per_loc=2
        self._box_code_size = 7

        self._use_rotate_nms=False  #False for pillar, True for second
        self._nms_post_max_sizes=[100] #300 for pillar, 100 for second
        self._nms_score_thresholds=[0.8] # 0.4 in submit, but 0.3 can get better hard performance #pillar use 0.05, second 0.3
        self._nms_iou_thresholds=[0.5] ## NOTE: double check #pillar use 0.5, second use 0.01
        self._post_center_range=list(model_cfg.post_center_limit_range) ## NOTE: double check
        self._use_direction_classifier=False ## NOTE: double check

        detections = []
        t = time.time()

        for i in tqdm(range(len(data_iter))):
            example = next(data_iter)
            example.pop("metrics")
            voxels = example['voxels']
            coors = example['coordinates']
            num_points = example['num_points']
            # gt_coords = example['gt_boxes_coords']
            # pos_reg_targets =example['pos_reg_targets']
            # pos_labels =example['pos_labels']
            #
            # #for new method
            # reg_targets, labels = self.GroundTruth2FeatMap(pos_labels, pos_reg_targets, gt_coords, self.fp_w, self.fp_h)
            # reg_targets = reg_targets.reshape(1,-1, pos_reg_targets.shape[-1])
            # labels = labels.reshape(1,-1)

            # For pillar
            # voxels=self.VoxelFeatureNet(voxels, coors, num_points) #(V,100,C) -> (B=1, C=9, V, N=100) #for pillar
            #(B=1, C=9, V, N=100)

            # For BCL
            voxels = voxels.transpose(2, 1, 0) #(V=fixed,N=1,C=4) -> (C=4, N=1, V=fixed)
            voxels = np.expand_dims(voxels, axis=0) # (C=4, N=1, V=fixed) -> (B=1, C=4, N=1, V=fixed)

            self.solver.test_nets[0].blobs['top_prev'].reshape(*voxels.shape)
            self.solver.test_nets[0].blobs['top_prev'].data[...] = voxels
            self.solver.test_nets[0].blobs['top_lat_feats'].reshape(*coors.shape)
            self.solver.test_nets[0].blobs['top_lat_feats'].data[...] = coors
            self.solver.test_nets[0].forward()
            cls_preds = self.solver.test_nets[0].blobs['f_cls_preds'].data[...]
            box_preds = self.solver.test_nets[0].blobs['f_box_preds'].data[...]

            ####################################################################
            # C, H, W = box_preds.shape[1:]
            # box_preds = box_preds.reshape(-1, self._num_anchor_per_loc,
            #                            self._box_code_size, H, W).transpose(
            #                                0, 1, 3, 4, 2)
            #
            # cls_preds = cls_preds.reshape(-1, self._num_anchor_per_loc,
            #                            self._num_class, H, W).transpose(
            #                                0, 1, 3, 4, 2)

            ####################################################################

            preds_dict = {"box_preds":box_preds, "cls_preds":cls_preds}

            example = example_convert_to_torch(example, torch.float32)
            preds_dict = example_convert_to_torch(preds_dict, torch.float32)
            detections += self.predict(example, preds_dict)
            # print(detections[-1])
        sec_per_ex = len(data_iter) / (time.time() - t)
        global_step = 1 ## TODO:
        model_dir = str(Path(self.model_dir).resolve())
        model_dir = Path(model_dir)
        result_path = model_dir / 'results'
        result_path_step = result_path / f"step_{1}"
        result_path_step.mkdir(parents=True, exist_ok=True)
        model_logging.log_text(
            f'generate label finished({sec_per_ex:.2f}/s). start eval:',
            global_step)
        result_dict = eval_dataset.dataset.evaluation(
            detections, str(result_path_step))
        for k, v in result_dict["results"].items():
            model_logging.log_text("Evaluation {}".format(k), global_step)
            model_logging.log_text(v, global_step)
        model_logging.log_metrics(result_dict["detail"], global_step)
    def predict(self, example, preds_dict):
        """start with v1.6.0, this function don't contain any kitti-specific code.
        Returns:
            predict: list of pred_dict.
            pred_dict: {
                box3d_lidar: [N, 7] 3d box.
                scores: [N]
                label_preds: [N]
                metadata: meta-data which contains dataset-specific information.
                    for kitti, it contains image idx (label idx),
                    for nuscenes, sample_token is saved in it.
            }
        """
        batch_size = example['anchors'].shape[0]
        if "metadata" not in example or len(example["metadata"]) == 0:
            meta_list = [None] * batch_size
        else:
            meta_list = example["metadata"]

        batch_anchors = example["anchors"].view(batch_size, -1, example["anchors"].shape[-1])
        if "anchors_mask" not in example:
            batch_anchors_mask = [None] * batch_size
        else:
            batch_anchors_mask = example["anchors_mask"].view(batch_size, -1)

        t = time.time()
        batch_box_preds = preds_dict["box_preds"]
        batch_cls_preds = preds_dict["cls_preds"]
        batch_box_preds = batch_box_preds.view(batch_size, -1,
                                               self._box_coder.code_size)

        num_class_with_bg = self._num_class
        if not self._encode_background_as_zeros:
            num_class_with_bg = self._num_class + 1

        batch_cls_preds = batch_cls_preds.view(batch_size, -1,
                                               num_class_with_bg)
        batch_box_preds = self._box_coder.decode_torch(batch_box_preds,
                                                       batch_anchors)
        if self._use_direction_classifier:
            batch_dir_preds = preds_dict["dir_cls_preds"]
            batch_dir_preds = batch_dir_preds.view(batch_size, -1,
                                                   self._num_direction_bins)
        else:
            batch_dir_preds = [None] * batch_size

        predictions_dicts = []
        post_center_range = None
        if len(self._post_center_range) > 0:
            post_center_range = torch.tensor(
                self._post_center_range,
                dtype=batch_box_preds.dtype,
                device=batch_box_preds.device).float()
        for box_preds, cls_preds, dir_preds, a_mask, meta in zip(
                batch_box_preds, batch_cls_preds, batch_dir_preds,
                batch_anchors_mask, meta_list):
            if a_mask is not None:
                box_preds = box_preds[a_mask]
                cls_preds = cls_preds[a_mask]
            box_preds = box_preds.float()
            cls_preds = cls_preds.float()
            if self._use_direction_classifier:
                if a_mask is not None:
                    dir_preds = dir_preds[a_mask]
                dir_labels = torch.max(dir_preds, dim=-1)[1]
            if self._encode_background_as_zeros:
                # this don't support softmax
                assert self._use_sigmoid_score is True
                total_scores = torch.sigmoid(cls_preds)
            else:
                # encode background as first element in one-hot vector
                if self._use_sigmoid_score:
                    total_scores = torch.sigmoid(cls_preds)[..., 1:]
                else:
                    total_scores = F.softmax(cls_preds, dim=-1)[..., 1:]
            # Apply NMS in birdeye view
            # Jim added
            # print("each score: ", total_scores)
            # print("total_scores > 0.05:" , np.unique(total_scores.cpu().detach().numpy() > 0.05,
            #                                             return_counts = True))
            # print("total_scores > 0.3:" , np.unique(total_scores.cpu().detach().numpy() > 0.3,
            #                                             return_counts = True))
            if self._use_rotate_nms:
                nms_func = box_torch_ops.rotate_nms
            else:
                nms_func = box_torch_ops.nms
            feature_map_size_prod = batch_box_preds.shape[
                1] // self.target_assigner.num_anchors_per_location
            if self._multiclass_nms:
                assert self._encode_background_as_zeros is True
                boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                if not self._use_rotate_nms:
                    box_preds_corners = box_torch_ops.center_to_corner_box2d(
                        boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                        boxes_for_nms[:, 4])
                    boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                        box_preds_corners)

                selected_boxes, selected_labels, selected_scores = [], [], []
                selected_dir_labels = []

                scores = total_scores
                boxes = boxes_for_nms
                selected_per_class = []
                score_threshs = self._nms_score_thresholds
                pre_max_sizes = self._nms_pre_max_sizes
                post_max_sizes = self._nms_post_max_sizes
                iou_thresholds = self._nms_iou_thresholds
                for class_idx, score_thresh, pre_ms, post_ms, iou_th in zip(
                        range(self._num_class),
                        score_threshs,
                        pre_max_sizes, post_max_sizes, iou_thresholds):
                    if self._nms_class_agnostic:
                        class_scores = total_scores.view(
                            feature_map_size_prod, -1,
                            self._num_class)[..., class_idx]
                        class_scores = class_scores.contiguous().view(-1)
                        class_boxes_nms = boxes.view(-1,
                                                     boxes_for_nms.shape[-1])
                        class_boxes = box_preds
                        class_dir_labels = dir_labels
                    else:
                        anchors_range = self.target_assigner.anchors_range(class_idx)
                        class_scores = total_scores.view(
                            -1,
                            self._num_class)[anchors_range[0]:anchors_range[1], class_idx]
                        class_boxes_nms = boxes.view(-1,
                            boxes_for_nms.shape[-1])[anchors_range[0]:anchors_range[1], :]
                        class_scores = class_scores.contiguous().view(-1)
                        class_boxes_nms = class_boxes_nms.contiguous().view(
                            -1, boxes_for_nms.shape[-1])
                        class_boxes = box_preds.view(-1,
                            box_preds.shape[-1])[anchors_range[0]:anchors_range[1], :]
                        class_boxes = class_boxes.contiguous().view(
                            -1, box_preds.shape[-1])
                        if self._use_direction_classifier:
                            class_dir_labels = dir_labels.view(-1)[anchors_range[0]:anchors_range[1]]
                            class_dir_labels = class_dir_labels.contiguous(
                            ).view(-1)
                    if score_thresh > 0.0:
                        class_scores_keep = class_scores >= score_thresh
                        if class_scores_keep.shape[0] == 0:
                            selected_per_class.append(None)
                            continue
                        class_scores = class_scores[class_scores_keep]
                    if class_scores.shape[0] != 0:
                        if score_thresh > 0.0:
                            class_boxes_nms = class_boxes_nms[
                                class_scores_keep]
                            class_boxes = class_boxes[class_scores_keep]
                            class_dir_labels = class_dir_labels[
                                class_scores_keep]
                        keep = nms_func(class_boxes_nms, class_scores, pre_ms,
                                        post_ms, iou_th)
                        if keep.shape[0] != 0:
                            selected_per_class.append(keep)
                        else:
                            selected_per_class.append(None)
                    else:
                        selected_per_class.append(None)
                    selected = selected_per_class[-1]

                    if selected is not None:
                        selected_boxes.append(class_boxes[selected])
                        selected_labels.append(
                            torch.full([class_boxes[selected].shape[0]],
                                       class_idx,
                                       dtype=torch.int64,
                                       device=box_preds.device))
                        if self._use_direction_classifier:
                            selected_dir_labels.append(
                                class_dir_labels[selected])
                        selected_scores.append(class_scores[selected])
                selected_boxes = torch.cat(selected_boxes, dim=0)
                selected_labels = torch.cat(selected_labels, dim=0)
                selected_scores = torch.cat(selected_scores, dim=0)
                if self._use_direction_classifier:
                    selected_dir_labels = torch.cat(selected_dir_labels, dim=0)
            else:
                # get highest score per prediction, than apply nms
                # to remove overlapped box.
                if num_class_with_bg == 1:
                    top_scores = total_scores.squeeze(-1)
                    top_labels = torch.zeros(
                        total_scores.shape[0],
                        device=total_scores.device,
                        dtype=torch.long)
                else:
                    top_scores, top_labels = torch.max(
                        total_scores, dim=-1)
                if self._nms_score_thresholds[0] > 0.0:
                    top_scores_keep = top_scores >= self._nms_score_thresholds[0]
                    top_scores = top_scores.masked_select(top_scores_keep)
                    print("nms_score_thresholds is {} and found {} cars ".format(self._nms_score_thresholds, len(top_scores)))
                if top_scores.shape[0] != 0:
                    if self._nms_score_thresholds[0] > 0.0:
                        box_preds = box_preds[top_scores_keep]
                        if self._use_direction_classifier:
                            dir_labels = dir_labels[top_scores_keep]
                        top_labels = top_labels[top_scores_keep]
                    boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                    if not self._use_rotate_nms:
                        box_preds_corners = box_torch_ops.center_to_corner_box2d(
                            boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                            boxes_for_nms[:, 4])
                        boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                            box_preds_corners)
                    # the nms in 3d detection just remove overlap boxes.
                    selected = nms_func(
                        boxes_for_nms,
                        top_scores,
                        pre_max_size=self._nms_pre_max_sizes[0],
                        post_max_size=self._nms_post_max_sizes[0],
                        iou_threshold=self._nms_iou_thresholds[0],
                    )
                    print("IOU_thresholds is {} and remove overlap found {} cars ".format(self._nms_iou_thresholds, len(top_scores)))
                else:
                    selected = []
                # if selected is not None:
                selected_boxes = box_preds[selected]
                if self._use_direction_classifier:
                    selected_dir_labels = dir_labels[selected]
                selected_labels = top_labels[selected]
                selected_scores = top_scores[selected]
            # finally generate predictions.
            if selected_boxes.shape[0] != 0:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                if self._use_direction_classifier:
                    dir_labels = selected_dir_labels
                    period = (2 * np.pi / self._num_direction_bins)
                    dir_rot = box_torch_ops.limit_period(
                        box_preds[..., 6] - self._dir_offset,
                        self._dir_limit_offset, period)
                    box_preds[
                        ...,
                        6] = dir_rot + self._dir_offset + period * dir_labels.to(
                            box_preds.dtype)
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                if post_center_range is not None:
                    mask = (final_box_preds[:, :3] >=
                            post_center_range[:3]).all(1)
                    mask &= (final_box_preds[:, :3] <=
                             post_center_range[3:]).all(1)
                    predictions_dict = {
                        "box3d_lidar": final_box_preds[mask],
                        "scores": final_scores[mask],
                        "label_preds": label_preds[mask],
                        "metadata": meta,
                    }
                else:
                    predictions_dict = {
                        "box3d_lidar": final_box_preds,
                        "scores": final_scores,
                        "label_preds": label_preds,
                        "metadata": meta,
                    }
            else:
                dtype = batch_box_preds.dtype
                device = batch_box_preds.device
                predictions_dict = {
                    "box3d_lidar":
                    torch.zeros([0, box_preds.shape[-1]],
                                dtype=dtype,
                                device=device),
                    "scores":
                    torch.zeros([0], dtype=dtype, device=device),
                    "label_preds":
                    torch.zeros([0], dtype=top_labels.dtype, device=device),
                    "metadata":
                    meta,
                }

            predictions_dicts.append(predictions_dict)

        return predictions_dicts
    def SimpleVoxel(self, voxels, coors, num_points):
        points_mean = np.sum(voxels[:, :, :], axis=1, keepdims=True) / (num_points.reshape(-1,1,1))
        return points_mean
    def VoxelFeatureNet(self, voxels, coors, num_points):
        # for VoxelFeatureNet
        point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
        voxel_size = [0.16, 0.16, 4]
        vx = voxel_size[0]
        vy = voxel_size[1]
        x_offset = vx / 2 + point_cloud_range[0]
        y_offset = vy / 2 + point_cloud_range[1]

        points_mean = np.sum(voxels[:, :, :3], axis=1, keepdims=True) / num_points.reshape(-1,1,1)
        f_cluster = voxels[:, :, :3] - points_mean

        f_center = np.zeros_like(voxels[:, :, :2]) #huge bug here if not zero like it will directly change the value
        f_center[:, :, 0] = voxels[:, :, 0] - (np.expand_dims(coors[:, 3].astype(float), axis=1) * vx + x_offset)
        f_center[:, :, 1] = voxels[:, :, 1] - (np.expand_dims(coors[:, 2].astype(float), axis=1) * vy + y_offset)

        features_ls = [voxels, f_cluster, f_center]
        features = np.concatenate(features_ls, axis=-1) #[num_voxles, points_num, features]

        points_per_voxels = features.shape[1]
        mask = get_paddings_indicator_np(num_points, points_per_voxels, axis=0)
        mask = np.expand_dims(mask, axis=-1)
        features *= mask
        #(voxel, npoint, channel) -> (channel, voxels, npoints)
        features = np.expand_dims(features, axis=0)
        features = features.transpose(0,3,1,2)
        return features
    def GroundTruth2FeatMap(self, labels, reg_targets, gt_coords, fp_w, fp_h):
        gt_coords = gt_coords.squeeze()
        reg_targets = reg_targets.squeeze()
        nchannels = reg_targets.shape[-1]
        canvas = np.zeros(shape=(fp_w , fp_h, nchannels)).astype(int)  #(7, 176, 200) #reg_head = 7
        label_canvas = -1*np.ones(shape=(fp_w, fp_h)).astype(int)  #(7, 176, 200) #reg_head = 7
        pc_range = [0,-40,-3,70.4,40,1]
        # convert from real space coordinate to feature map index
        w = np.floor((gt_coords[:,0] * fp_w) / 70.4).astype(int)
        # Add half of the fp_h to convert negative y to positive fp_h
        y = np.floor((gt_coords[:,1] * fp_h) / 80 + fp_h/2).astype(int)
        canvas[w,y,:] = reg_targets
        label_canvas[w,y] = labels
        return canvas, label_canvas

    """Load weight fromm caffe model"""
    def caffe_load_weight(self):
        exp_dir = "./exp/leo_debug/"
        weight_name = "pp_iter_18560"
        weight_path = os.path.join(exp_dir + weight_name)
        net_ = caffe.Net(os.path.join(exp_dir + "train.prototxt"), "{}.caffemodel".format(weight_path), caffe.TRAIN)

        self.solver.net.params['mlp_0'][0].data[...] = net_.params['mlp_0'][0].data[...]#w
        self.solver.net.params['mlp_sc_0'][0].data[...] = net_.params['mlp_sc_0'][0].data[...]#w
        self.solver.net.params['mlp_sc_0'][1].data[...] = net_.params['mlp_sc_0'][1].data[...]#b
        self.solver.net.params['mlp_bn_0'][0].data[...] = net_.params['mlp_bn_0'][0].data[...]#mean
        self.solver.net.params['mlp_bn_0'][1].data[...] = net_.params['mlp_bn_0'][1].data[...]#var
        self.solver.net.params['mlp_bn_0'][2].data[...] = net_.params['mlp_bn_0'][2].data[...]

        def rpn_layer(layer_name):
            self.solver.net.params[str(layer_name[0])][0].data[...] = net_.params[str(layer_name[0])][0].data[...]#w

            self.solver.net.params[str(layer_name[1])][0].data[...] = net_.params[str(layer_name[1])][0].data[...]#w
            self.solver.net.params[str(layer_name[1])][1].data[...] = net_.params[str(layer_name[1])][1].data[...]#b

            self.solver.net.params[str(layer_name[2])][0].data[...] = net_.params[str(layer_name[2])][0].data[...]#mean
            self.solver.net.params[str(layer_name[2])][1].data[...] = net_.params[str(layer_name[2])][1].data[...]#var
            self.solver.net.params[str(layer_name[2])][2].data[...] = net_.params[str(layer_name[2])][2].data[...]

        ##################################block1################################
        caffe_layre = ['ini_conv1_0', 'ini_conv1_sc_0', 'ini_conv1_bn_0']
        rpn_layer(caffe_layre)

        for idx in range(3):
            caffe_layre = ['rpn_conv1_{}'.format(idx), 'rpn_conv1_sc_{}'.format(idx), 'rpn_conv1_bn_{}'.format(idx)]
            rpn_layer(caffe_layre)

        caffe_layre = ['rpn_deconv1', 'rpn_deconv1_sc', 'rpn_deconv1_bn']
        rpn_layer(caffe_layre)

        ##################################block2################################
        caffe_layre = ['ini_conv2_0', 'ini_conv2_sc_0', 'ini_conv2_bn_0']
        rpn_layer(caffe_layre)

        for idx in range(5):
            caffe_layre = ['rpn_conv2_{}'.format(idx), 'rpn_conv2_sc_{}'.format(idx), 'rpn_conv2_bn_{}'.format(idx)]
            rpn_layer(caffe_layre)

        caffe_layre = ['rpn_deconv2', 'rpn_deconv2_sc', 'rpn_deconv2_bn']
        rpn_layer(caffe_layre)

        ##################################block3################################
        caffe_layre = ['ini_conv3_0', 'ini_conv3_sc_0', 'ini_conv3_bn_0']
        rpn_layer(caffe_layre)

        for idx in range(5):
            caffe_layre = ['rpn_conv3_{}'.format(idx), 'rpn_conv3_sc_{}'.format(idx), 'rpn_conv3_bn_{}'.format(idx)]
            rpn_layer(caffe_layre)

        caffe_layre = ['rpn_deconv3', 'rpn_deconv3_sc', 'rpn_deconv3_bn']
        rpn_layer(caffe_layre)

        ################################# Head ################################
        self.solver.net.params['cls_head'][0].data[...] = net_.params['cls_head'][0].data[...]
        self.solver.net.params['cls_head'][1].data[...] = net_.params['cls_head'][1].data[...]
        self.solver.net.params['reg_head'][0].data[...] = net_.params['reg_head'][0].data[...]
        self.solver.net.params['reg_head'][1].data[...] = net_.params['reg_head'][1].data[...]

    """load pytourch weight"""
    def prepare_pretrained(self, weights_path, layer_name):
        weights = os.listdir(weights_path)

        graph = [w for w in weights if layer_name in w]
        layer_key = [int(m.split('_')[0]) for m in graph]
        layer_dict = dict(zip(layer_key, graph))

        keys = layer_dict.keys()
        keys = sorted(keys)
        layer_ordered = []
        for k in keys:
            layer_ordered.append(layer_dict[k])
        # print(layer_ordered) #debug
        return layer_ordered
    def load_pretrained_weight(self):

        weights_path = '/home/ubuntu/second.pytorch/second/output/model_weights/'
        layer_name = 'voxel_feature'
        mlp_Layer = self.prepare_pretrained(weights_path, layer_name)

        self.solver.net.params['mlp_0'][0].data[...] = np.load(weights_path + mlp_Layer[0]) #w
        self.solver.net.params['mlp_sc_0'][0].data[...] = np.load(weights_path + mlp_Layer[1]) #w
        self.solver.net.params['mlp_sc_0'][1].data[...] = np.load(weights_path + mlp_Layer[2]) #b
        self.solver.net.params['mlp_bn_0'][0].data[...] = np.load(weights_path + mlp_Layer[3]) #mean
        self.solver.net.params['mlp_bn_0'][1].data[...] = np.load(weights_path + mlp_Layer[4]) #var
        self.solver.net.params['mlp_bn_0'][2].data[...] = 1

        def rpn_layer(layer_name, block, idx, stride):
            self.solver.net.params[str(layer_name[0])][0].data[...] = np.load(weights_path + block[idx*stride]) #w

            self.solver.net.params[str(layer_name[1])][0].data[...] = np.load(weights_path + block[idx*stride+1]) #w
            self.solver.net.params[str(layer_name[1])][1].data[...] = np.load(weights_path + block[idx*stride+2]) #b

            self.solver.net.params[str(layer_name[2])][0].data[...] = np.load(weights_path + block[idx*stride+3]) #mean
            self.solver.net.params[str(layer_name[2])][1].data[...] = np.load(weights_path + block[idx*stride+4]) #var
            self.solver.net.params[str(layer_name[2])][2].data[...] = 1

        ##################################block1################################
        layer_name = 'rpn.blocks.0'
        rpn_block = self.prepare_pretrained(weights_path, layer_name)
        # print(rpn_block)
        caffe_layre = ['ini_conv1_0', 'ini_conv1_sc_0', 'ini_conv1_bn_0']
        rpn_layer(caffe_layre, rpn_block, idx=0, stride=5)

        for idx in range(3):
            caffe_layre = ['rpn_conv1_{}'.format(idx), 'rpn_conv1_sc_{}'.format(idx), 'rpn_conv1_bn_{}'.format(idx)]
            rpn_layer(caffe_layre, rpn_block, idx=idx+1, stride=5)

        layer_name = 'rpn.deblocks.0'
        rpn_deconv = self.prepare_pretrained(weights_path, layer_name)
        # print(rpn_deconv)
        caffe_layre = ['rpn_deconv1', 'rpn_deconv1_sc', 'rpn_deconv1_bn']
        rpn_layer(caffe_layre, rpn_deconv, idx=0, stride=5)



        ##################################block2################################
        layer_name = 'rpn.blocks.1'
        rpn_block = self.prepare_pretrained(weights_path, layer_name)
        # print(rpn_block)
        caffe_layre = ['ini_conv2_0', 'ini_conv2_sc_0', 'ini_conv2_bn_0']
        rpn_layer(caffe_layre, rpn_block, idx=0, stride=5)

        for idx in range(5):
            caffe_layre = ['rpn_conv2_{}'.format(idx), 'rpn_conv2_sc_{}'.format(idx), 'rpn_conv2_bn_{}'.format(idx)]
            rpn_layer(caffe_layre, rpn_block, idx=idx+1, stride=5)

        layer_name = 'rpn.deblocks.1'
        rpn_deconv = self.prepare_pretrained(weights_path, layer_name)
        # print(rpn_deconv)
        caffe_layre = ['rpn_deconv2', 'rpn_deconv2_sc', 'rpn_deconv2_bn']
        rpn_layer(caffe_layre, rpn_deconv, idx=0, stride=5)

        ##################################block3################################
        layer_name = 'rpn.blocks.2'
        rpn_block = self.prepare_pretrained(weights_path, layer_name)
        # print(rpn_block)
        caffe_layre = ['ini_conv3_0', 'ini_conv3_sc_0', 'ini_conv3_bn_0']
        rpn_layer(caffe_layre, rpn_block, idx=0, stride=5)


        for idx in range(5):
            caffe_layre = ['rpn_conv3_{}'.format(idx), 'rpn_conv3_sc_{}'.format(idx), 'rpn_conv3_bn_{}'.format(idx)]
            rpn_layer(caffe_layre, rpn_block, idx=idx+1, stride=5)

        layer_name = 'rpn.deblocks.2'
        rpn_deconv = self.prepare_pretrained(weights_path, layer_name)
        # print(rpn_deconv)
        caffe_layre = ['rpn_deconv3', 'rpn_deconv3_sc', 'rpn_deconv3_bn']
        rpn_layer(caffe_layre, rpn_deconv, idx=0, stride=5)

        ################################# Head ################################
        layer_name = 'rpn.conv'
        rpn_conv = self.prepare_pretrained(weights_path, layer_name)
        self.solver.net.params['cls_head'][0].data[...] = np.load(weights_path + rpn_conv[0]) #w cls_head
        self.solver.net.params['cls_head'][1].data[...] = np.load(weights_path + rpn_conv[1]) #b cls_head
        self.solver.net.params['reg_head'][0].data[...] = np.load(weights_path + rpn_conv[2]) #w box_head
        self.solver.net.params['reg_head'][1].data[...] = np.load(weights_path + rpn_conv[3]) #b box_head

"""
class EvalSolverWrapper:
    def __init__(self,  test_net,
                        weight_path,
                        log_path=None):

        self.log_path = log_path
        self.test_net = test_net
        self.weight = weight_path

        if self.log_path is not None:
            self.logw = LogWriter(self.log_path, sync_cycle=100)
            with self.logw.mode('val') as logger:
                self.sc_val_3d_easy_7 = logger.scalar("mAP3D_easy_7")
                self.sc_val_3d_moder_7 = logger.scalar("mAP3D_moderate_7")
                self.sc_val_3d_hard_7 = logger.scalar("mAP3D__hard_7")

    def train_model(self):
        caffe.set_mode_gpu()
        caffe.set_device(0)
        net = self.eval_kitti(self.test_net, self.weight)
        for i in tqdm(range(3769)):
            net.forward()
            print(net.blobs['cls_preds'].data[...])


        if self.log_path is not None:
            step = self.solver.iter
            map3d_easy_7 = net.blobs['e7'].data
            map3d_moder_7 = net.blobs['m7'].data
            map3d_hard_7 = net.blobs['h7'].data
            self.sc_val_3d_easy_7.add_record(step, map3d_easy_7)
            self.sc_val_3d_moder_7.add_record(step, map3d_moder_7)
            self.sc_val_3d_easy_7.add_record(step, map3d_hard_7)

    def eval_kitti(self, eval_model, weight):
        net = caffe.Net(eval_model, weight, caffe.TEST)
        return net
"""
