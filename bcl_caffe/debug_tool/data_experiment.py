import copy
from pathlib import Path
import pickle
import shutil
import time
import numpy as np
import torch
import torchplus
from tqdm import tqdm
import numpy_indexed as npi

from google.protobuf import text_format
import second.data.kitti_common as kitti
from second.builder import target_assigner_builder, voxel_builder
from second.pytorch.core import box_torch_ops
from second.data.preprocess import merge_second_batch, merge_second_batch_multigpu
from second.protos import pipeline_pb2
from second.pytorch.builder import box_coder_builder, input_reader_builder
from second.pytorch.models.voxel_encoder import get_paddings_indicator_np #for pillar
from second.utils.log_tool import SimpleModelLog
import caffe
from enum import Enum

def create_model_folder(model_dir, config_path):
    model_dir = str(Path(model_dir).resolve())
    if Path(model_dir).exists():
        pass
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

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

def load_dataloader(input_cfg, eval_input_cfg, model_cfg, generate_anchors_cachae):
    if phase == 'train':
        dataset = input_reader_builder.build(
            input_cfg,
            model_cfg,
            training=True,
            voxel_generator=voxel_generator,
            target_assigner=target_assigner,
            multi_gpu=False,
            generate_anchors_cachae=generate_anchors_cachae) #True FOR Pillar, False For BCL

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=input_cfg.batch_size,
            shuffle=True,
            num_workers =input_cfg.preprocess.num_workers,
            pin_memory=False,
            collate_fn=merge_second_batch,
            worker_init_fn=_worker_init_fn,
            drop_last=not False)
        return dataloader

    elif phase == 'eval':
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
        return eval_dataloader

def Voxel3DStack2D(voxels, coors, num_points):
    coords_xy = np.delete(coors, obj=1, axis=1) #byx
    _coors, voxels = npi.group_by(coords_xy[1:]).mean(voxels) #feature for Voxels mean or max
    _, num_points = npi.group_by(coords_xy[1:]).sum(num_points)
    return voxels, _coors, num_points

def prepare_loss_weights(labels, dtype="float32"):

    pos_cls_weight=1.0 
    neg_cls_weight=1.0

    cared = labels >= 0
    print("label ", np.unique(labels, return_counts=True))
    # cared: [N, num_anchors]
    positives = labels > 0
    negatives = labels == 0
    negative_cls_weights = negatives.astype(dtype) * neg_cls_weight
    posetive_cls_weights = positives.astype(dtype) * pos_cls_weight #(1, 107136)
    cls_weights = negative_cls_weights + posetive_cls_weights
    reg_weights = positives.astype(dtype)

    pos_normalizer = np.sum(positives, 1, keepdims=True).astype(dtype)
    reg_weights /= np.clip(pos_normalizer, a_min=1.0, a_max=None) #(1, 107136)
    cls_weights /= np.clip(pos_normalizer, a_min=1.0, a_max=None) #(1, 107136)

    return cls_weights, reg_weights, cared

generate_anchors_cachae = False #True FOR Pillar/Seocnd, False For BCL
phase = "train" #"eval", "train"
model_dir = "experiment"
config_path = "../../second/configs/car.fhd.config" #../../second/configs/poiltpillars/xyzer_16.config
create_model_folder(model_dir, config_path)
input_cfg, eval_input_cfg, model_cfg, train_cfg = load_config(model_dir, config_path)
voxel_generator, target_assigner = build_network(model_cfg)
dataloader = load_dataloader(input_cfg, eval_input_cfg, model_cfg, generate_anchors_cachae) #True FOR Pillar/Seocnd, False For BCL
if not generate_anchors_cachae:
    print("[INFO] BCL TEST!!!!!!!!!!!!!!!!!!!")
elif generate_anchors_cachae:
    print("[INFO] SECOND TEST!!!!!!!!!!!!!!!!")

avg=[]
min = 9999999
max = 0
num_point_features=4 #xyzi
max_voxels = 12000
voxels_arr = []

for example in tqdm(dataloader):
    seg_points = example['seg_points']
    seg_labels = example['seg_labels']
    reg_targets = example['reg_targets']
    labels = example['labels']
    
    cls_weights, reg_weights, cared = prepare_loss_weights(labels)

    #print("cls_weights", np.unique(cls_weights, return_counts=True))
    #print("reg_weights", np.unique(reg_weights, return_counts=True))
    #print("cared", np.unique(cared, return_counts=True))

################################################################################
#     voxels_arr.append(len(voxels))
#     if len(voxels)<min:
#         min = len(voxels)
#         # ranking = np.argsort(num_points)[::-1]
#         print("cur min ", min)
#     if len(voxels)>max:
#         max = len(voxels)
#         # ranking = np.argsort(num_points)[::-1]
#         print("cur max ", max)
#
# avg.append(len(voxels))
# _avg = np.array(avg, dtype=int)
# print("avg voxels num", np.mean(_avg))
################################################################################
