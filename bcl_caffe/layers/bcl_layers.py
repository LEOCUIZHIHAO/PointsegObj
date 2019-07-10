
from pathlib import Path
import pickle
import shutil
import time, timeit
import numpy as np
import torch
import torchplus

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
import numpy_indexed as npi
from numba import jit
from numba import njit, prange
from second.core import box_np_ops

def build_network(model_cfg, measure_time=False):
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)

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

class LossNormType(Enum):
    NormByNumPositives = "norm_by_num_positives"
    NormByNumExamples = "norm_by_num_examples"
    NormByNumPosNeg = "norm_by_num_pos_neg"

class DataFeature(caffe.Layer):
    def setup(self, bottom, top):
        params = {}
        params.update(eval(self.param_str))
        bcl_keep_voxels_eval = params['bcl_keep_voxels_eval']
        seg_keep_points_eval = params['seg_keep_points_eval']
        num_points_per_voxel = params['num_points_per_voxel']
        is_segmentation = params['segmentation']
        try:
            batch_size = params["eval_batch_size"]
        except Exception as e:
            batch_size = 1
        # BCL
        if is_segmentation:
            top[0].reshape(*(batch_size, seg_keep_points_eval, 4)) # for pillar shape should (B,C=9,V,N=100), For second (B,C=1,V,N=5)
        else:
            # top[0].reshape(*(bcl_keep_voxels_eval, num_points_per_voxel, 4)) #pillar
            top[0].reshape(*(batch_size, bcl_keep_voxels_eval, 4)) #pillar
    def reshape(self, bottom, top):
        pass
    def forward(self, bottom, top):
        pass
    def backward(self, top, propagate_down, bottom):
        pass

class VoxelSegNetInput(caffe.Layer):
    def setup(self, bottom, top):
        params = {}
        params.update(eval(self.param_str))
        max_voxels = params['max_voxels']
        points_per_voxel = params['points_per_voxel']
        seg_keep_points_eval = params['seg_keep_points_eval']
        top[0].reshape(*(1, seg_keep_points_eval, 4)) # seg points
        top[1].reshape(*(1, max_voxels, 3)) # Coords
        top[2].reshape(*(1, seg_keep_points_eval, 3)) # p2voxel_idx
    def reshape(self, bottom, top):
        pass
    def forward(self, bottom, top):
        pass
    def backward(self, top, propagate_down, bottom):
        pass

class LatticeFeature(caffe.Layer):
    def setup(self, bottom, top):
        params = {}
        params.update(eval(self.param_str))
        bcl_keep_voxels_eval = params['bcl_keep_voxels_eval']
        seg_keep_points_eval = params['seg_keep_points_eval']
        is_segmentation = params['segmentation']
        # BCL
        if is_segmentation:
            top[0].reshape(*(seg_keep_points_eval,4)) #(V, C=4) # TODO:
        else:
            top[0].reshape(*(bcl_keep_voxels_eval,4)) # for pillar
    def reshape(self, bottom, top):
        pass
    def forward(self, bottom, top):
        pass
    def backward(self, top, propagate_down, bottom):
        pass

#for point-wise segmentation
class InputKittiData(caffe.Layer):
    def setup(self, bottom, top):
        params = dict(batch_size=1)
        params.update(eval(self.param_str))

        model_dir = params['model_dir']
        config_path = params['config_path']
        self.phase = params['subset']
        self.input_cfg, self.eval_input_cfg, self.model_cfg, train_cfg = load_config(model_dir, config_path)
        self.voxel_generator, self.target_assigner = build_network(self.model_cfg)
        self.dataloader = self.load_dataloader(self.input_cfg, self.eval_input_cfg,
                                                        self.model_cfg, args=params)
        # for point segmentation detection
        for example in self.dataloader:
            seg_points = example['seg_points']
            seg_labels =example['seg_labels']
            break
        self.data_iter = iter(self.dataloader)

        # for point object segmentation
        top[0].reshape(*seg_points.shape)
        top[1].reshape(*seg_labels.shape)

    def reshape(self, bottom, top):
        pass
    def forward(self, bottom, top):
        try:
            example = next(self.data_iter)
        except Exception as e:
            print("\n[info] start a new epoch for {} data\n".format(self.phase))
            self.data_iter = iter(self.dataloader)
            example = next(self.data_iter)

        seg_points = example['seg_points']
        seg_labels = example['seg_labels']

        # """shuffle car seg points""" #move to preprocess
        # indices = np.arange(seg_labels.shape[1])
        # np.random.shuffle(indices)
        # seg_points = seg_points[:,indices]
        # seg_labels = seg_labels[:,indices]

        # for point object segmentation
        top[0].reshape(*seg_points.shape)
        top[1].reshape(*seg_labels.shape)
        top[0].data[...] = seg_points
        top[1].data[...] = seg_labels
        #print("[debug] train img idx : ", example["metadata"])

    def backward(self, top, propagate_down, bottom):
        pass
    def load_dataloader(self, input_cfg, eval_input_cfg, model_cfg, args):
        try: segmentation = args["segmentation"]
        except: segmentation = True
        try: bcl_keep_voxels = args["bcl_keep_voxels"]
        except: bcl_keep_voxels = 6000
        try: seg_keep_points = args["seg_keep_points"]
        except: seg_keep_points = 8000
        dataset = input_reader_builder.build(
            input_cfg,
            model_cfg,
            training=True,
            voxel_generator=self.voxel_generator,
            target_assigner=self.target_assigner,
            segmentation=segmentation,
            bcl_keep_voxels=bcl_keep_voxels,
            seg_keep_points=seg_keep_points,
            multi_gpu=False,
            generate_anchors_cachae=args['anchors_cachae']) #True FOR Pillar, False For BCL

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=input_cfg.batch_size,
            shuffle=True,
            num_workers=input_cfg.preprocess.num_workers,
            pin_memory=False,
            collate_fn=merge_second_batch,
            worker_init_fn=_worker_init_fn,
            drop_last=not False)
        return dataloader

#for voxel-wise object detection
class InputKittiDataV2(caffe.Layer):

    def setup(self, bottom, top):

        params = dict(batch_size=1)
        params.update(eval(self.param_str))

        model_dir = params['model_dir']
        config_path = params['config_path']
        self.phase = params['subset']
        self.input_cfg, self.eval_input_cfg, self.model_cfg, train_cfg = load_config(model_dir, config_path)
        self.voxel_generator, self.target_assigner = build_network(self.model_cfg)
        self.dataloader = self.load_dataloader(self.input_cfg, self.eval_input_cfg,
                                                        self.model_cfg, args=params)

        # for point segmentation detection
        for example in self.dataloader:
            voxels = example['voxels']
            coors = example['coordinates']
            labels = example['labels']
            reg_targets = example['reg_targets']
            break
        self.data_iter = iter(self.dataloader)

        # for point object segmentation
        top[0].reshape(*voxels.shape)
        top[1].reshape(*coors.shape)
        top[2].reshape(*labels.shape)
        top[3].reshape(*reg_targets.shape)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        try:
            example = next(self.data_iter)
        except Exception as e:
            print("\n[info] start a new epoch for {} data\n".format(self.phase))
            self.data_iter = iter(self.dataloader)
            example = next(self.data_iter)

        voxels = example['voxels']
        coors = example['coordinates']
        labels = example['labels']
        reg_targets = example['reg_targets']

        # for point object segmentation
        # top[0].reshape(*voxels.shape)
        # top[1].reshape(*coors.shape)
        # top[2].reshape(*labels.shape)
        # top[3].reshape(*reg_targets.shape)
        top[0].data[...] = voxels
        top[1].data[...] = coors
        top[2].data[...] = labels
        top[3].data[...] = reg_targets
        #print("[debug] train img idx : ", example["metadata"])

    def backward(self, top, propagate_down, bottom):
        pass

    def load_dataloader(self, input_cfg, eval_input_cfg, model_cfg, args):
        try: segmentation = args["segmentation"]
        except: segmentation = False
        try: bcl_keep_voxels = args["bcl_keep_voxels"]
        except: bcl_keep_voxels = 6000
        try: seg_keep_points = args["seg_keep_points"]
        except: seg_keep_points = 8000
        dataset = input_reader_builder.build(
            input_cfg,
            model_cfg,
            training=True,
            voxel_generator=self.voxel_generator,
            target_assigner=self.target_assigner,
            segmentation=segmentation,
            bcl_keep_voxels=bcl_keep_voxels,
            seg_keep_points=seg_keep_points,
            multi_gpu=False,
            generate_anchors_cachae=args['anchors_cachae']) #True FOR Pillar, False For BCL

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=input_cfg.batch_size,
            shuffle=True,
            num_workers=input_cfg.preprocess.num_workers,
            pin_memory=False,
            collate_fn=merge_second_batch,
            worker_init_fn=_worker_init_fn,
            drop_last=not False)
        return dataloader

#for point-wise object detection & segmentation
class InputKittiDataV3(caffe.Layer):

    def setup(self, bottom, top):

        params = dict(batch_size=1)
        params.update(eval(self.param_str))

        model_dir = params['model_dir']
        config_path = params['config_path']
        self.phase = params['subset']
        self.generate_anchors_cachae = params['anchors_cachae'] #True FOR Pillar, False For BCL
        self.input_cfg, self.eval_input_cfg, self.model_cfg, train_cfg = load_config(model_dir, config_path)
        self.voxel_generator, self.target_assigner = build_network(self.model_cfg)
        self.dataloader = self.load_dataloader(self.input_cfg, self.eval_input_cfg, self.model_cfg)

        # for point segmentation detection
        for example in self.dataloader:
            points = example['points']
            coors = example['coordinates']
            labels = example['labels']
            reg_targets = example['reg_targets']
            break
        self.data_iter = iter(self.dataloader)

        # for point object segmentation
        top[0].reshape(*points.shape)
        top[1].reshape(*coors.shape)
        top[2].reshape(*labels.shape)
        top[3].reshape(*reg_targets.shape)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        try:
            example = next(self.data_iter)
        except Exception as e:
            print("\n[info] start a new epoch for {} data\n".format(self.phase))
            self.data_iter = iter(self.dataloader)
            example = next(self.data_iter)

        points = example['points']
        coors = example['coordinates']
        labels = example['labels']
        reg_targets = example['reg_targets']

        # for point object segmentation
        top[0].reshape(*points.shape)
        top[1].reshape(*coors.shape)
        top[2].reshape(*labels.shape)
        top[3].reshape(*reg_targets.shape)
        top[0].data[...] = points
        top[1].data[...] = coors
        top[2].data[...] = labels
        top[3].data[...] = reg_targets
        #print("[debug] train img idx : ", example["metadata"])

    def backward(self, top, propagate_down, bottom):
        pass

    def load_dataloader(self, input_cfg, eval_input_cfg, model_cfg, args):
        dataset = input_reader_builder.build(
            input_cfg,
            model_cfg,
            training=True,
            voxel_generator=self.voxel_generator,
            target_assigner=self.target_assigner,
            multi_gpu=False,
            #generate_anchors_cachae=self.generate_anchors_cachae
            ) #True FOR Pillar, False For BCL

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=input_cfg.batch_size,
            shuffle=True,
            num_workers=input_cfg.preprocess.num_workers,
            pin_memory=False,
            collate_fn=merge_second_batch,
            worker_init_fn=_worker_init_fn,
            drop_last=not False)
        return dataloader

#for point-wise object detection
class InputKittiDataV4(caffe.Layer):

    def setup(self, bottom, top):

        params = dict(batch_size=1)
        params['anchors_cachae']=False #False For BCL, Anchor Free
        params.update(eval(self.param_str))

        model_dir = params['model_dir']
        config_path = params['config_path']
        self.phase = params['subset']
        self.input_cfg, self.eval_input_cfg, self.model_cfg, train_cfg = load_config(model_dir, config_path)
        self.voxel_generator, self.target_assigner = build_network(self.model_cfg)
        self.dataloader = self.load_dataloader(self.input_cfg, self.eval_input_cfg,
                                                        self.model_cfg, args=params)

        for example in self.dataloader:
            points = example['points']
            labels = example['labels']
            reg_targets = example['reg_targets']
            break
        self.data_iter = iter(self.dataloader)

        top[0].reshape(*points.shape)
        top[1].reshape(*labels.shape)
        top[2].reshape(*reg_targets.shape)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        try:
            example = next(self.data_iter)
        except Exception as e:
            print("\n[info] start a new epoch for {} data\n".format(self.phase))
            self.data_iter = iter(self.dataloader)
            example = next(self.data_iter)

        points = example['points']
        labels = example['labels']
        reg_targets = example['reg_targets']

        top[0].reshape(*points.shape)
        top[1].reshape(*labels.shape)
        top[2].reshape(*reg_targets.shape)
        top[0].data[...] = points
        top[1].data[...] = labels
        top[2].data[...] = reg_targets
        #print("[debug] train img idx : ", example["metadata"])

    def backward(self, top, propagate_down, bottom):
        pass

    def load_dataloader(self, input_cfg, eval_input_cfg, model_cfg, args):
        dataset = input_reader_builder.build(
            input_cfg,
            model_cfg,
            training=True,
            voxel_generator=self.voxel_generator,
            target_assigner=self.target_assigner,
            segmentation=segmentation,
            bcl_keep_voxels=bcl_keep_voxels,
            seg_keep_points=seg_keep_points,
            multi_gpu=False,
            generate_anchors_cachae=args['anchors_cachae']) #True FOR Pillar, False For BCL

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=input_cfg.batch_size,
            shuffle=True,
            num_workers=input_cfg.preprocess.num_workers,
            pin_memory=False,
            collate_fn=merge_second_batch,
            worker_init_fn=_worker_init_fn,
            drop_last=not False)
        return dataloader

#for seg_feature map
class InputKittiDataV5(caffe.Layer):
    def setup(self, bottom, top):
        params = dict(batch_size=1)
        params.update(eval(self.param_str))

        model_dir = params['model_dir']
        config_path = params['config_path']
        self.phase = params['subset']
        self.input_cfg, self.eval_input_cfg, self.model_cfg, train_cfg = load_config(model_dir, config_path)
        self.voxel_generator, self.target_assigner = build_network(self.model_cfg)
        self.dataloader = self.load_dataloader(self.input_cfg, self.eval_input_cfg,
                                                        self.model_cfg, args=params)
        # for point segmentation detection
        for example in self.dataloader:
            seg_points = example['seg_points']
            seg_labels =example['seg_labels']
            labels = example['labels']
            reg_targets =example['reg_targets']

            break
        self.data_iter = iter(self.dataloader)

        # for point object segmentation
        top[0].reshape(*seg_points.shape)
        top[1].reshape(*seg_labels.shape)
        top[2].reshape(*labels.shape)
        top[3].reshape(*reg_targets.shape)

    def reshape(self, bottom, top):
        pass
    def forward(self, bottom, top):
        try:
            example = next(self.data_iter)
        except Exception as e:
            print("\n[info] start a new epoch for {} data\n".format(self.phase))
            self.data_iter = iter(self.dataloader)
            example = next(self.data_iter)

        seg_points = example['seg_points']
        seg_labels = example['seg_labels']
        labels = example['labels']
        reg_targets =example['reg_targets']

        # """shuffle car seg points""" #moved to preprocess
        # for point object segmentation
        top[0].data[...] = seg_points
        top[1].data[...] = seg_labels
        top[2].data[...] = labels
        top[3].data[...] = reg_targets
        #print("[debug] train img idx : ", example["metadata"])

    def backward(self, top, propagate_down, bottom):
        pass
    def load_dataloader(self, input_cfg, eval_input_cfg, model_cfg, args):
        try: segmentation = args["segmentation"]
        except: segmentation = True
        try: bcl_keep_voxels = args["bcl_keep_voxels"]
        except: bcl_keep_voxels = 6000
        try: seg_keep_points = args["seg_keep_points"]
        except: seg_keep_points = 8000
        dataset = input_reader_builder.build(
            input_cfg,
            model_cfg,
            training=True,
            voxel_generator=self.voxel_generator,
            target_assigner=self.target_assigner,
            segmentation=segmentation,
            bcl_keep_voxels=bcl_keep_voxels,
            seg_keep_points=seg_keep_points,
            multi_gpu=False,
            generate_anchors_cachae=args['anchors_cachae']) #True FOR Pillar, False For BCL

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=input_cfg.batch_size,
            shuffle=True,
            num_workers=input_cfg.preprocess.num_workers,
            pin_memory=False,
            collate_fn=merge_second_batch,
            worker_init_fn=_worker_init_fn,
            drop_last=not False)
        return dataloader

class InputKittiDataV6(caffe.Layer):
    def setup(self, bottom, top):
        params = dict(batch_size=1)
        params.update(eval(self.param_str))

        model_dir = params['model_dir']
        config_path = params['config_path']
        self.phase = params['subset']
        self.input_cfg, self.eval_input_cfg, self.model_cfg, train_cfg = load_config(model_dir, config_path)
        self.voxel_generator, self.target_assigner = build_network(self.model_cfg)
        self.dataloader = self.load_dataloader(self.input_cfg, self.eval_input_cfg,
                                                        self.model_cfg, args=params)
        # for point segmentation detection
        for example in self.dataloader:
            seg_points = example['seg_points']
            seg_labels =example['seg_labels']
            gt_box = example['gt_boxes']
            break
        self.data_iter = iter(self.dataloader)

        # for point object segmentation
        top[0].reshape(*seg_points.shape)
        top[1].reshape(*seg_labels.shape)
        top[2].reshape(*gt_box.shape)

    def reshape(self, bottom, top):
        pass
    def forward(self, bottom, top):
        try:
            example = next(self.data_iter)
        except Exception as e:
            print("\n[info] start a new epoch for {} data\n".format(self.phase))
            self.data_iter = iter(self.dataloader)
            example = next(self.data_iter)

        seg_points = example['seg_points']
        seg_labels = example['seg_labels']
        gt_box = example['gt_boxes']
        # """shuffle car seg points""" #moved to preprocess
        # for point object segmentation
        top[0].data[...] = seg_points
        top[1].data[...] = seg_labels
        top[2].reshape(*gt_box.shape)
        top[2].data[...] = gt_box
        #print("[debug] train img idx : ", example["metadata"])

    def backward(self, top, propagate_down, bottom):
        pass
    def load_dataloader(self, input_cfg, eval_input_cfg, model_cfg, args):
        try: segmentation = args["segmentation"]
        except: segmentation = True
        try: bcl_keep_voxels = args["bcl_keep_voxels"]
        except: bcl_keep_voxels = 6000
        try: seg_keep_points = args["seg_keep_points"]
        except: seg_keep_points = 8000
        dataset = input_reader_builder.build(
            input_cfg,
            model_cfg,
            training=True,
            voxel_generator=self.voxel_generator,
            target_assigner=self.target_assigner,
            segmentation=segmentation,
            bcl_keep_voxels=bcl_keep_voxels,
            seg_keep_points=seg_keep_points,
            multi_gpu=False,
            generate_anchors_cachae=args['anchors_cachae']) #True FOR Pillar, False For BCL

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=input_cfg.batch_size,
            shuffle=True,
            num_workers=input_cfg.preprocess.num_workers,
            pin_memory=False,
            collate_fn=merge_second_batch,
            worker_init_fn=_worker_init_fn,
            drop_last=not False)
        return dataloader

class InputKittiDataV7(caffe.Layer):
    def setup(self, bottom, top):
        params = dict(batch_size=1)
        params.update(eval(self.param_str))

        model_dir = params['model_dir']
        config_path = params['config_path']
        self.phase = params['subset']
        self.input_cfg, self.eval_input_cfg, self.model_cfg, train_cfg = load_config(model_dir, config_path)
        self.voxel_generator, self.target_assigner = build_network(self.model_cfg)
        self.dataloader = self.load_dataloader(self.input_cfg, self.eval_input_cfg,
                                                        self.model_cfg, args=params)
        # for point segmentation detection
        for example in self.dataloader:
            seg_points = example['seg_points']
            seg_labels = example['seg_labels']
            coords = example['coords']
            p2voxel_idx = example['p2voxel_idx']
            cls_labels = example['cls_labels']
            reg_targets = example['reg_targets']
            break
        self.data_iter = iter(self.dataloader)

        # for point object segmentation
        top[0].reshape(*seg_points.shape)
        top[1].reshape(*seg_labels.shape)
        top[2].reshape(*coords.shape)
        top[3].reshape(*p2voxel_idx.shape)
        top[4].reshape(*cls_labels.shape)
        top[5].reshape(*reg_targets.shape)

    def reshape(self, bottom, top):
        pass
    def forward(self, bottom, top):
        try:
            example = next(self.data_iter)
        except Exception as e:
            print("\n[info] start a new epoch for {} data\n".format(self.phase))
            self.data_iter = iter(self.dataloader)
            example = next(self.data_iter)

        seg_points = example['seg_points']
        seg_labels = example['seg_labels']
        coords = example['coords']
        p2voxel_idx = example['p2voxel_idx']
        cls_labels = example['cls_labels']
        reg_targets = example['reg_targets']

        # """shuffle car seg points""" #moved to preprocess
        # for point object segmentation
        top[0].data[...] = seg_points
        top[1].data[...] = seg_labels
        top[2].data[...] = coords
        top[3].data[...] = p2voxel_idx
        top[4].data[...] = cls_labels
        top[5].data[...] = reg_targets

        #print("[debug] train img idx : ", example["metadata"])

    def backward(self, top, propagate_down, bottom):
        pass
    def load_dataloader(self, input_cfg, eval_input_cfg, model_cfg, args):
        try: segmentation = args["segmentation"]
        except: segmentation = True
        try: bcl_keep_voxels = args["bcl_keep_voxels"]
        except: bcl_keep_voxels = 6000
        try: seg_keep_points = args["seg_keep_points"]
        except: seg_keep_points = 8000
        try: points_per_voxel = args["points_per_voxel"]
        except: points_per_voxel = 200
        dataset = input_reader_builder.build(
            input_cfg,
            model_cfg,
            training=True,
            voxel_generator=self.voxel_generator,
            target_assigner=self.target_assigner,
            segmentation=segmentation,
            bcl_keep_voxels=bcl_keep_voxels,
            seg_keep_points=seg_keep_points,
            multi_gpu=False,
            generate_anchors_cachae=args['anchors_cachae'],
            points_per_voxel=points_per_voxel) #True FOR Pillar, False For BCL

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=input_cfg.batch_size,
            shuffle=True,
            num_workers=input_cfg.preprocess.num_workers,
            pin_memory=False,
            collate_fn=merge_second_batch,
            worker_init_fn=_worker_init_fn,
            drop_last=not False)
        return dataloader

class Scatter(caffe.Layer):
    def setup(self, bottom, top):
        param = eval(self.param_str)
        output_shape = param['output_shape']
        self.ny = output_shape[0]
        self.nx = output_shape[1]
        self.nchannels = output_shape[2]
        self.batch_size = 1

        voxel_features = bottom[0].data
        voxel_features = np.squeeze(voxel_features) #(1, 64, 1, Voxel) -> (64,Voxel)
        coords = bottom[1].data # reverse_index is True, output coordinates will be zyx format
        batch_canvas, _ = self.ScatterNet(voxel_features, coords, self.nchannels, self.nx, self.ny)

        top[0].reshape(*batch_canvas.shape)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        voxel_features = bottom[0].data #(1,64,-1,1)
        voxel_features = np.squeeze(voxel_features) #(1, 64, -1, 1) -> (64,-1)
        coords = bottom[1].data
        batch_canvas, self.indices = self.ScatterNet(voxel_features, coords, self.nchannels, self.nx, self.ny)

        top[0].data[...] = batch_canvas

    def backward(self, top, propagate_down, bottom):
        diff = top[0].diff.reshape(self.batch_size, self.nchannels, self.nx * self.ny)[:,:,self.indices]
        bottom[0].diff[...] = np.expand_dims(diff, axis=2)

    def ScatterNet(self, voxel_features, coords, nchannels, feature_map_x, feature_map_y):
        canvas = np.zeros(shape=(nchannels, feature_map_x * feature_map_y)) #(nchannels,-1)
        # Only include non-empty pillars
        indices = coords[:, 2] * feature_map_x + coords[:, 3]
        indices = indices.astype(int)
        canvas[:, indices] = voxel_features
        canvas = canvas.reshape(self.batch_size, nchannels, feature_map_y, feature_map_x)
        return canvas, indices

    def Voxel3DStack2D(self, voxel_features, coors):
        # coors = np.delete(coors, obj=1, axis=1) #delete z column
        coors = coors[:,2:]
        voxel_group = npi.group_by(coors) #features mean
        coors_idx, voxel_features = voxel_group.mean(voxel_features) #features max
        return voxel_features, coors_idx, voxel_group

class Point2FeatMap(caffe.Layer):
    def setup(self, bottom, top):
        param = eval(self.param_str)
        # (1,4,100,100,80)
        self.feat_map_size = param['feat_map_size']
        self.point_cloud_range = np.array(param['point_cloud_range'])
        try: self.use_depth = param['use_depth']
        except: self.use_depth = False
        try: self.use_score = param['use_score']
        except: self.use_score = False
        try: self.use_points = param['use_points']
        except: self.use_points = False
        self.thresh = param['thresh']
        self.num_feat = self.feat_map_size[1]
        self.num_points = self.feat_map_size[2]
        self.feat_h = self.feat_map_size[3]
        self.feat_w = self.feat_map_size[4]
        self.feat_map_size = np.array(self.feat_map_size)
        top[0].reshape(1, self.num_feat*self.num_points, self.feat_h, self.feat_w)
        # top[0].reshape(1, self.num_feat, self.num_points, self.feat_h*self.feat_w) #leo added to (1,c,n,h*w)
        # if self.num_feat != 4 and self.num_feat != 5:
        #     print("[Error] Feature number other than 4 and 5 is not yet implemented")
        #     raise NotImplementedError
    def reshape(self, bottom, top):
        pass
    def forward(self, bottom, top):
        points = bottom[0].data[...].squeeze()
        point_xy = points[:,:2]
        score = bottom[1].data[...].squeeze()
        if not self.use_depth:
            points = points[:,:3]
        if self.use_score:
            points = np.concatenate((points, score.reshape(-1,1)), axis = -1)
        if len(bottom) > 2:
            extra_feat = bottom[2].data[...].squeeze().transpose()
            self.extra_feat_shape = extra_feat.shape
            points = np.concatenate((points, extra_feat), axis = -1)
        if not self.use_points:
            points = points[:,3:]
        self.p2feat_idx = np.zeros((points.shape[0], 3), dtype=np.int_)
        points = points[score>self.thresh,:]
        point_xy = point_xy[score>self.thresh,:]
        p2feat_idx = self.p2feat_idx[score>self.thresh,:]
        # Calculate grid size of feature map
        # voxel size of [w, h]
        voxel_size = (self.point_cloud_range[3:5]-self.point_cloud_range[:2])/np.array([self.feat_w, self.feat_h])
        # create a feature map of cooresponding shape
        feat_map = np.zeros((1, self.num_feat, self.num_points, self.feat_h, self.feat_w), dtype=np.float32)
        points_in_feat_map = np.zeros((self.feat_h, self.feat_w), dtype=np.int_)
        #point to voxel indices (num, h, w)
        offset = np.array(self.point_cloud_range[:2])
        # Indices (w, h)
        indices = np.floor((point_xy-offset)/voxel_size).astype(np.int_)
        # remove points and indices that are out put range
        feat_map, p2feat_idx=self.to_feat_map(points, feat_map, indices, points_in_feat_map,
                                                p2feat_idx, self.num_points)
        self.p2feat_idx[score>self.thresh,:] = p2feat_idx
        feat_map = feat_map.reshape(1, -1, self.feat_h, self.feat_w)
        # feat_map = feat_map.reshape(1, self.num_feat, self.num_points, self.feat_h*self.feat_w) #leo added to (1,c,n,h*w)
        top[0].data[...] = feat_map
    def backward(self, top, propagate_down, bottom):
        diff = top[0].diff.reshape(1,self.num_feat,self.num_points,self.feat_h,
                                                            self.feat_w)
        backward = np.zeros((1,1,1,self.p2feat_idx.shape[0]))
        mask = (self.p2feat_idx > 0).any(-1)
        indices = self.p2feat_idx[mask]
        diff = diff[:,:,indices[:,0],indices[:,1],indices[:,2]].squeeze().transpose()
        if len(bottom) > 2:
            backward_extra = np.zeros((1,self.extra_feat_shape[1],1,self.extra_feat_shape[0]))
            # OPTIMIZE: get rid of two expand_dims
            extra_feat_backward = diff[:,-self.extra_feat_shape[1]:].transpose()
            extra_feat_backward = np.expand_dims(extra_feat_backward,0)
            extra_feat_backward = np.expand_dims(extra_feat_backward,2)
            backward_extra[..., mask] = extra_feat_backward
            bottom[2].diff[...] = backward_extra
            if self.use_score:
                backward[..., mask] = diff[:,(-self.extra_feat_shape[1]-1)]
                bottom[1].diff[...] = backward
        else:
            if self.use_score:
                backward[..., mask] = diff[:,-1]
                bottom[1].diff[...] = backward
    @staticmethod
    @njit#(nopython=True)#, parallel=True)
    def to_feat_map(points, feat_map, indices, points_in_feat_map, p2feat_idx, num_p_feat = 10):
        # Indices is (width, height)
        for idx in prange(len(indices)):
            feat_index = indices[idx]
            num = points_in_feat_map[feat_index[1],feat_index[0]]
            if num < num_p_feat:
                feat_map[:,:,num,feat_index[1],feat_index[0]] = points[idx]
                points_in_feat_map[feat_index[1],feat_index[0]] += 1
                p2feat_idx[idx,0] = num
                p2feat_idx[idx,1] = feat_index[1]
                p2feat_idx[idx,2] = feat_index[0]
        return feat_map, p2feat_idx

#return (B,C,N,H,W)
class Point2FeatMapV3(caffe.Layer):
    def setup(self, bottom, top):
        param = eval(self.param_str)
        # (1,4,100,100,80)
        self.feat_map_size = param['feat_map_size']
        self.point_cloud_range = np.array(param['point_cloud_range'])
        try: self.use_depth = param['use_depth']
        except: self.use_depth = False
        try: self.use_score = param['use_score']
        except: self.use_score = False
        try: self.use_points = param['use_points']
        except: self.use_points = False
        self.thresh = param['thresh']
        self.num_feat = self.feat_map_size[1]
        self.num_points = self.feat_map_size[2]
        self.feat_h = self.feat_map_size[3]
        self.feat_w = self.feat_map_size[4]
        self.feat_map_size = np.array(self.feat_map_size)
        # top[0].reshape(1, self.num_feat*self.num_points, self.feat_h, self.feat_w)
        top[0].reshape(1, self.num_feat, self.num_points, self.feat_h* self.feat_w) #leo added to (1,c,n,h*w)
        # if self.num_feat != 4 and self.num_feat != 5:
        #     print("[Error] Feature number other than 4 and 5 is not yet implemented")
        #     raise NotImplementedError
    def reshape(self, bottom, top):
        pass
    def forward(self, bottom, top):
        points = bottom[0].data[...].squeeze()
        point_xy = points[:,:2]
        #score = bottom[1].data[...].squeeze()
        if not self.use_depth:
            points = points[:,:3]
        if self.use_score:
            points = np.concatenate((points, score.reshape(-1,1)), axis = -1)
        if len(bottom) > 1:
            extra_feat = bottom[1].data[...].squeeze().transpose()
            self.extra_feat_shape = extra_feat.shape
            points = np.concatenate((points, extra_feat), axis = -1)
        if not self.use_points:
            points = points[:,3:]
        self.p2feat_idx = np.zeros((points.shape[0], 3), dtype=np.int_)
        #points = points[score>self.thresh,:]
        #point_xy = point_xy[score>self.thresh,:]
        # p2feat_idx = self.p2feat_idx#[score>self.thresh,:]
        # Calculate grid size of feature map
        # voxel size of [w, h]
        voxel_size = (self.point_cloud_range[3:5]-self.point_cloud_range[:2])/np.array([self.feat_w, self.feat_h])
        # create a feature map of cooresponding shape
        feat_map = np.zeros((1, self.num_feat, self.num_points, self.feat_h, self.feat_w), dtype=np.float32)
        points_in_feat_map = np.zeros((self.feat_h, self.feat_w), dtype=np.int_)
        #point to voxel indices (num, h, w)
        offset = np.array(self.point_cloud_range[:2])
        # Indices (w, h)
        indices = np.floor((point_xy-offset)/voxel_size).astype(np.int_)
        # remove points and indices that are out put range
        feat_map, p2feat_idx=self.to_feat_map(points, feat_map, indices, points_in_feat_map,
                                                self.p2feat_idx, self.num_points)
        # self.p2feat_idx[score>self.thresh,:] = p2feat_idx
        self.p2feat_idx = p2feat_idx
        # feat_map = feat_map.reshape(1, -1, self.feat_h, self.feat_w)
        feat_map = feat_map.reshape(1, self.num_feat, self.num_points, self.feat_h* self.feat_w) #leo added to (1,c,n,h*w)
        top[0].data[...] = feat_map
    def backward(self, top, propagate_down, bottom):
        diff = top[0].diff.reshape(1,self.num_feat,self.num_points,self.feat_h,
                                                            self.feat_w)
        #backward = np.zeros((1,1,1,self.p2feat_idx.shape[0]))
        mask = (self.p2feat_idx > 0).any(-1)
        indices = self.p2feat_idx[mask]
        diff = diff[:,:,indices[:,0],indices[:,1],indices[:,2]].squeeze().transpose()
        if len(bottom) > 1:
            # backward_extra = np.zeros((1,self.extra_feat_shape[1],1,self.extra_feat_shape[0])) #old
            # OPTIMIZE: get rid of two expand_dims
            extra_feat_backward = diff[:,-self.extra_feat_shape[1]:].transpose()
            extra_feat_backward = np.expand_dims(extra_feat_backward,0)
            extra_feat_backward = np.expand_dims(extra_feat_backward,2)
            # backward_extra[..., mask] = extra_feat_backward #old
            # bottom[1].diff[...] = backward_extra #old

            #####################Test new backward##############################
            bottom[1].diff[...] = 0
            bottom[1].diff[..., mask] = extra_feat_backward
            #####################Test new backward##############################

            if self.use_score:
                pass
        else:
            if self.use_score:
                pass
    @staticmethod
    @njit#(nopython=True)#, parallel=True)
    def to_feat_map(points, feat_map, indices, points_in_feat_map, p2feat_idx, num_p_feat = 10):
        # Indices is (width, height)
        for idx in prange(len(indices)):
            feat_index = indices[idx]
            num = points_in_feat_map[feat_index[1],feat_index[0]]
            if num < num_p_feat:
                feat_map[:,:,num,feat_index[1],feat_index[0]] = points[idx]
                points_in_feat_map[feat_index[1],feat_index[0]] += 1
                p2feat_idx[idx,0] = num
                p2feat_idx[idx,1] = feat_index[1]
                p2feat_idx[idx,2] = feat_index[0]
        return feat_map, p2feat_idx

class Point2FeatMapV2(caffe.Layer):
    def setup(self, bottom, top):
        param = eval(self.param_str)
        # (1,4,100,100,80)
        self.feat_map_size = param['feat_map_size']
        self.point_cloud_range = np.array(param['point_cloud_range'])
        try: self.use_depth = param['use_depth']
        except: self.use_depth = False
        try: self.use_score = param['use_score']
        except: self.use_score = False
        try: self.use_points = param['use_points']
        except: self.use_points = False
        self.thresh = param['thresh']
        self.num_feat = self.feat_map_size[1]
        self.num_points = self.feat_map_size[2]
        self.feat_h = self.feat_map_size[3]
        self.feat_w = self.feat_map_size[4]
        self.feat_map_size = np.array(self.feat_map_size)
        top[0].reshape(1, self.num_feat*self.num_points, self.feat_h, self.feat_w)
        # top[0].reshape(1, self.num_feat, self.num_points, self.feat_h*self.feat_w) #leo added to (1,c,n,h*w)
        # if self.num_feat != 4 and self.num_feat != 5:
        #     print("[Error] Feature number other than 4 and 5 is not yet implemented")
        #     raise NotImplementedError
    def reshape(self, bottom, top):
        pass
    def forward(self, bottom, top):
        points = bottom[0].data[...].squeeze()
        point_xy = points[:,:2]
        #score = bottom[1].data[...].squeeze()
        if not self.use_depth:
            points = points[:,:3]
        if self.use_score:
            points = np.concatenate((points, score.reshape(-1,1)), axis = -1)
        if len(bottom) > 1:
            extra_feat = bottom[1].data[...].squeeze().transpose()
            self.extra_feat_shape = extra_feat.shape
            points = np.concatenate((points, extra_feat), axis = -1)
        if not self.use_points:
            points = points[:,3:]
        self.p2feat_idx = np.zeros((points.shape[0], 3), dtype=np.int_)
        #points = points[score>self.thresh,:]
        #point_xy = point_xy[score>self.thresh,:]
        # p2feat_idx = self.p2feat_idx#[score>self.thresh,:]
        # Calculate grid size of feature map
        # voxel size of [w, h]
        voxel_size = (self.point_cloud_range[3:5]-self.point_cloud_range[:2])/np.array([self.feat_w, self.feat_h])
        # create a feature map of cooresponding shape
        feat_map = np.zeros((1, self.num_feat, self.num_points, self.feat_h, self.feat_w), dtype=np.float32)
        points_in_feat_map = np.zeros((self.feat_h, self.feat_w), dtype=np.int_)
        #point to voxel indices (num, h, w)
        offset = np.array(self.point_cloud_range[:2])
        # Indices (w, h)
        indices = np.floor((point_xy-offset)/voxel_size).astype(np.int_)
        # remove points and indices that are out put range
        feat_map, p2feat_idx=self.to_feat_map(points, feat_map, indices, points_in_feat_map,
                                                self.p2feat_idx, self.num_points)
        # self.p2feat_idx[score>self.thresh,:] = p2feat_idx
        self.p2feat_idx = p2feat_idx
        feat_map = feat_map.reshape(1, -1, self.feat_h, self.feat_w)
        # feat_map = feat_map.reshape(1, self.num_feat, self.num_points, self.feat_h*self.feat_w) #leo added to (1,c,n,h*w)
        top[0].data[...] = feat_map
    def backward(self, top, propagate_down, bottom):
        diff = top[0].diff.reshape(1,self.num_feat,self.num_points,self.feat_h,
                                                            self.feat_w)
        #backward = np.zeros((1,1,1,self.p2feat_idx.shape[0]))
        mask = (self.p2feat_idx > 0).any(-1)
        indices = self.p2feat_idx[mask]
        diff = diff[:,:,indices[:,0],indices[:,1],indices[:,2]].squeeze().transpose()
        if len(bottom) > 1:
            # backward_extra = np.zeros((1,self.extra_feat_shape[1],1,self.extra_feat_shape[0])) #old
            # OPTIMIZE: get rid of two expand_dims
            extra_feat_backward = diff[:,-self.extra_feat_shape[1]:].transpose()
            extra_feat_backward = np.expand_dims(extra_feat_backward,0)
            extra_feat_backward = np.expand_dims(extra_feat_backward,2)
            # backward_extra[..., mask] = extra_feat_backward #old
            # bottom[1].diff[...] = backward_extra #old

            #####################Test new backward##############################
            bottom[1].diff[...] = 0
            bottom[1].diff[..., mask] = extra_feat_backward
            #####################Test new backward##############################

            if self.use_score:
                pass
        else:
            if self.use_score:
                pass
    @staticmethod
    @njit#(nopython=True)#, parallel=True)
    def to_feat_map(points, feat_map, indices, points_in_feat_map, p2feat_idx, num_p_feat = 10):
        # Indices is (width, height)
        for idx in prange(len(indices)):
            feat_index = indices[idx]
            num = points_in_feat_map[feat_index[1],feat_index[0]]
            if num < num_p_feat:
                feat_map[:,:,num,feat_index[1],feat_index[0]] = points[idx]
                points_in_feat_map[feat_index[1],feat_index[0]] += 1
                p2feat_idx[idx,0] = num
                p2feat_idx[idx,1] = feat_index[1]
                p2feat_idx[idx,2] = feat_index[0]
        return feat_map, p2feat_idx

class Point2FeatMapV4(caffe.Layer):
    def setup(self, bottom, top):
        param = eval(self.param_str)
        # (1,4,100,100,80)
        self.feat_map_size = param['feat_map_size']
        self.point_cloud_range = np.array(param['point_cloud_range'])
        try: self.use_depth = param['use_depth']
        except: self.use_depth = False
        try: self.use_score = param['use_score']
        except: self.use_score = False
        try: self.use_points = param['use_points']
        except: self.use_points = False
        self.thresh = param['thresh']
        self.num_feat = self.feat_map_size[1]
        self.num_points = self.feat_map_size[2]
        self.feat_h = self.feat_map_size[3]
        self.feat_w = self.feat_map_size[4]
        self.feat_map_size = np.array(self.feat_map_size)
        self.batch_size = bottom[1].data.shape[0]
        top[0].reshape(self.batch_size, self.num_feat*self.num_points, self.feat_h, self.feat_w)
        # top[0].reshape(1, self.num_feat, self.num_points, self.feat_h*self.feat_w) #leo added to (1,c,n,h*w)
        # if self.num_feat != 4 and self.num_feat != 5:
        #     print("[Error] Feature number other than 4 and 5 is not yet implemented")
        #     raise NotImplementedError
    def reshape(self, bottom, top):
        pass
    def forward(self, bottom, top):
        points = bottom[0].data[...]
        point_xy = points[:,:,:2]
        #score = bottom[1].data[...].squeeze()
        if not self.use_depth:
            points = points[:,:,:3]
        if self.use_score:
            points = np.concatenate((points, score.reshape(-1,1)), axis = -1)
        if len(bottom) > 1:

            extra_feat = bottom[1].data[...].squeeze(2).transpose(0,2,1)
            self.extra_feat_shape = extra_feat.shape
            points = np.concatenate((points, extra_feat), axis = -1)
        if not self.use_points:
            points = points[:,:,3:]
        self.p2feat_idx = np.zeros((self.batch_size,points.shape[1], 3), dtype=np.int_)
        #points = points[score>self.thresh,:]
        #point_xy = point_xy[score>self.thresh,:]
        # p2feat_idx = self.p2feat_idx#[score>self.thresh,:]
        # Calculate grid size of feature map
        # voxel size of [w, h]
        voxel_size = (self.point_cloud_range[3:5]-self.point_cloud_range[:2])/np.array([self.feat_w, self.feat_h])
        # create a feature map of cooresponding shape
        feat_map = np.zeros((self.batch_size, self.num_feat, self.num_points, self.feat_h, self.feat_w), dtype=np.float32)
        points_in_feat_map = np.zeros((self.batch_size, self.feat_h, self.feat_w), dtype=np.int_)
        #point to voxel indices (num, h, w)
        offset = np.array(self.point_cloud_range[:2])
        # Indices (w, h)
        indices = np.floor((point_xy-offset)/voxel_size).astype(np.int_)
        # remove points and indices that are out put range
        feat_map, p2feat_idx=self.to_feat_map(points, feat_map, indices, points_in_feat_map,
                                                self.p2feat_idx, self.num_points)
        # self.p2feat_idx[score>self.thresh,:] = p2feat_idx
        self.p2feat_idx = p2feat_idx
        feat_map = feat_map.reshape(self.batch_size, -1, self.feat_h, self.feat_w)
        # feat_map = feat_map.reshape(1, self.num_feat, self.num_points, self.feat_h*self.feat_w) #leo added to (1,c,n,h*w)
        top[0].data[...] = feat_map
    def backward(self, top, propagate_down, bottom):
        diff = top[0].diff.reshape(self.batch_size,self.num_feat,self.num_points,self.feat_h,
                                                            self.feat_w)
        bottom[1].diff[...] = 0
        for batch in range(self.batch_size):
            #backward = np.zeros((1,1,1,self.p2feat_idx.shape[0]))
            mask = (self.p2feat_idx[batch,...] > 0).any(-1)
            indices = self.p2feat_idx[batch, mask]
            diff_ = diff[batch,:,indices[:,0],indices[:,1],indices[:,2]].squeeze().transpose()
            if len(bottom) > 1:
                # backward_extra = np.zeros((1,self.extra_feat_shape[1],1,self.extra_feat_shape[0])) #old
                # OPTIMIZE: get rid of two expand_dims
                extra_feat_backward = diff_[:,-self.extra_feat_shape[1]:].transpose()
                # extra_feat_backward = np.expand_dims(extra_feat_backward,0)
                # print("extra_feat_shape", extra_feat_backward.shape)
                extra_feat_backward = np.expand_dims(extra_feat_backward,-1)
                # backward_extra[..., mask] = extra_feat_backward #old
                # bottom[1].diff[...] = backward_extra #old

                #####################Test new backward##############################
                bottom[1].diff[batch,:,:,mask] = extra_feat_backward
                #####################Test new backward##############################

                if self.use_score:
                    continue
            else:
                if self.use_score:
                    continue
    #@njit#(nopython=True)#, parallel=True)
    @staticmethod
    @njit
    def to_feat_map(points, feat_map, indices, points_in_feat_map, p2feat_idx, num_p_feat = 10):
        # Indices is (width, height)
        for batch in prange(indices.shape[0]):
            for idx in prange(indices.shape[1]):
                feat_index = indices[batch,idx]
                num = points_in_feat_map[batch,feat_index[1],feat_index[0]]
                if num < num_p_feat:
                    feat_map[batch,:,num,feat_index[1],feat_index[0]] = points[batch,idx]
                    points_in_feat_map[batch,feat_index[1],feat_index[0]] += 1
                    p2feat_idx[batch,idx,0] = num
                    p2feat_idx[batch,idx,1] = feat_index[1]
                    p2feat_idx[batch,idx,2] = feat_index[0]
        return feat_map, p2feat_idx

class Point2Voxel3D(caffe.Layer):
    def setup(self, bottom, top):
        param = eval(self.param_str)
        self.extra_feat_shape = bottom[0].data.shape
        self.p2voxel_idx_shape = bottom[1].data.shape
        self.max_voxels = param['max_voxels']
        self.points_per_voxel = param['points_per_voxel']
        top[0].reshape(1, self.points_per_voxel*self.extra_feat_shape[1], 1, self.max_voxels)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        extra_feat = bottom[0].data[...]
        p2voxel_idx = bottom[1].data[...].astype(np.int_)
        voxels = np.zeros((1, self.extra_feat_shape[1], self.points_per_voxel, self.max_voxels))
        num = p2voxel_idx[:,:,0].squeeze()
        voxel_idx = p2voxel_idx[:,:,1].squeeze()
        point_idx = p2voxel_idx[:,:,2].squeeze()
        voxels[:,:,num,voxel_idx] = extra_feat[...,point_idx].squeeze()
        voxels = np.expand_dims(voxels.reshape(1,-1,self.max_voxels), 2)
        top[0].reshape(1, self.points_per_voxel*self.extra_feat_shape[1], 1, self.max_voxels)
        top[0].data[...] = voxels

    def backward(self, top, propagate_down, bottom):
        diff = top[0].diff.reshape(1, self.extra_feat_shape[1], self.points_per_voxel, self.max_voxels)
        p2voxel_idx = bottom[1].data[...].astype(np.int_)
        num = p2voxel_idx[:,:,0].squeeze()
        voxel_idx = p2voxel_idx[:,:,1].squeeze()
        point_idx = p2voxel_idx[:,:,2].squeeze()
        diff = diff[:, :, num, voxel_idx]
        backward = np.zeros(bottom[0].data.shape)
        backward[..., point_idx] = np.expand_dims(diff, 2)
        bottom[0].diff[...] = backward

class SegWeight(caffe.Layer):
    def setup(self, bottom, top):
        labels = bottom[0].data
        seg_weights = self.prepare_loss_weights(labels)
        top[0].reshape(*seg_weights.shape)

    def reshape(self, bottom, top):
        pass
    def forward(self, bottom, top):
        labels = bottom[0].data
        seg_weights = self.prepare_loss_weights(labels)
        top[0].data[...] = seg_weights
    def prepare_loss_weights(self,
                            labels,
                            pos_cls_weight=1.0,
                            neg_cls_weight=1.0,
                            dtype="float32"):

        positives = labels > 0
        negatives = labels == 0
        negative_cls_weights = negatives.astype(dtype) * neg_cls_weight
        posetive_cls_weights = positives.astype(dtype) * pos_cls_weight
        seg_weights = negative_cls_weights + posetive_cls_weights
        reg_weights = positives.astype(dtype)

        pos_normalizer = np.sum(positives, 1, keepdims=True).astype(dtype)
        seg_weights /= np.clip(pos_normalizer, a_min=1.0, a_max=None) #(1, 107136)

        return seg_weights
    def backward(self, top, propagate_down, bottom):
        pass

class PrepareLossWeight(caffe.Layer):
    def setup(self, bottom, top):
        labels = bottom[0].data
        cls_weights, reg_weights, cared = self.prepare_loss_weights(labels)

        top[0].reshape(*cared.shape)
        top[1].reshape(*reg_weights.shape) #reg_outside_weights
        top[2].reshape(*cls_weights.shape)
    def reshape(self, bottom, top):
        pass
    def forward(self, bottom, top):
        labels = bottom[0].data
        cls_weights, reg_weights, cared = self.prepare_loss_weights(labels)
        top[0].data[...] = cared
        top[1].data[...] = reg_weights #reg_outside_weights
        top[2].data[...] = cls_weights
    def prepare_loss_weights(self,
                            labels,
                            pos_cls_weight=1.0, # TODO: pass params here
                            neg_cls_weight=1.0,
                            loss_norm_type=LossNormType.NormByNumPositives,
                            dtype="float32"):
        """get cls_weights and reg_weights from labels.
        """
        cared = labels >= 0
        # print("label ", np.unique(labels, return_counts=True))
        # cared: [N, num_anchors]
        positives = labels > 0
        negatives = labels == 0
        negative_cls_weights = negatives.astype(dtype) * neg_cls_weight
        posetive_cls_weights = positives.astype(dtype) * pos_cls_weight #(1, 107136)
        cls_weights = negative_cls_weights + posetive_cls_weights
        reg_weights = positives.astype(dtype)

        if loss_norm_type == LossNormType.NormByNumExamples:
            num_examples = cared.astype(dtype).sum(1, keepdims=True)
            num_examples = np.clip(num_examples, a_min=1.0, a_max=None)
            cls_weights /= num_examples
            bbox_normalizer = np.sum(positives, 1, keepdims=True).astype(dtype)
            reg_weights /= np.clip(bbox_normalizer, a_min=1.0, a_max=None)

        elif loss_norm_type == LossNormType.NormByNumPositives:  # for focal loss
            pos_normalizer = np.sum(positives, 1, keepdims=True).astype(dtype)
            reg_weights /= np.clip(pos_normalizer, a_min=1.0, a_max=None) #(1, 107136)
            cls_weights /= np.clip(pos_normalizer, a_min=1.0, a_max=None) #(1, 107136)


        elif loss_norm_type == LossNormType.NormByNumPosNeg:
            pos_neg = np.stack([positives, negatives], a_min=-1).astype(dtype)
            normalizer = np.sum(pos_neg, 1, keepdims=True)  # [N, 1, 2]
            cls_normalizer = np.sum((pos_neg * normalizer),-1)  # [N, M]
            cls_normalizer = np.clip(cls_normalizer, a_min=1.0, a_max=None)
            # cls_normalizer will be pos_or_neg_weight/num_pos_or_neg
            normalizer = np.clip(normalizer, a_min=1.0, a_max=None)
            reg_weights /= normalizer[:, 0:1, 0]
            cls_weights /= cls_normalizer

        else:
            raise ValueError(
                "unknown loss norm type. available: {list(LossNormType)}")

        return cls_weights, reg_weights, cared
    def backward(self, top, propagate_down, bottom):
        pass

#For Point-Wise model
class PrepareLossWeightV2(caffe.Layer):
    def setup(self, bottom, top):
        labels = bottom[0].data
        cls_weights, reg_weights = self.prepare_loss_weights(labels)

        top[0].reshape(*reg_weights.shape) #reg_outside_weights
        top[1].reshape(*cls_weights.shape)
    def reshape(self, bottom, top):
        pass
    def forward(self, bottom, top):
        labels = bottom[0].data
        cls_weights, reg_weights = self.prepare_loss_weights(labels)
        top[0].data[...] = reg_weights #reg_outside_weights
        top[1].data[...] = cls_weights
    def prepare_loss_weights(self,
                            labels,
                            pos_cls_weight=1.0,
                            neg_cls_weight=1.0,
                            loss_norm_type=LossNormType.NormByNumPositives,
                            dtype="float32"):

        # print("label ", np.unique(labels, return_counts=True))
        positives = labels > 0
        negatives = labels == 0
        negative_cls_weights = negatives.astype(dtype) * neg_cls_weight
        posetive_cls_weights = positives.astype(dtype) * pos_cls_weight #(1, 107136)
        cls_weights = negative_cls_weights + posetive_cls_weights
        reg_weights = positives.astype(dtype)

        if loss_norm_type == LossNormType.NormByNumExamples:
            num_examples = cared.astype(dtype).sum(1, keepdims=True)
            num_examples = np.clip(num_examples, a_min=1.0, a_max=None)
            cls_weights /= num_examples
            bbox_normalizer = np.sum(positives, 1, keepdims=True).astype(dtype)
            reg_weights /= np.clip(bbox_normalizer, a_min=1.0, a_max=None)

        elif loss_norm_type == LossNormType.NormByNumPositives:  # for focal loss
            pos_normalizer = np.sum(positives, 1, keepdims=True).astype(dtype)
            reg_weights /= np.clip(pos_normalizer, a_min=1.0, a_max=None) #(1, 107136)
            cls_weights /= np.clip(pos_normalizer, a_min=1.0, a_max=None) #(1, 107136)


        elif loss_norm_type == LossNormType.NormByNumPosNeg:
            pos_neg = np.stack([positives, negatives], a_min=-1).astype(dtype)
            normalizer = np.sum(pos_neg, 1, keepdims=True)  # [N, 1, 2]
            cls_normalizer = np.sum((pos_neg * normalizer),-1)  # [N, M]
            cls_normalizer = np.clip(cls_normalizer, a_min=1.0, a_max=None)
            # cls_normalizer will be pos_or_neg_weight/num_pos_or_neg
            normalizer = np.clip(normalizer, a_min=1.0, a_max=None)
            reg_weights /= normalizer[:, 0:1, 0]
            cls_weights /= cls_normalizer

        else:
            raise ValueError(
                "unknown loss norm type. available: {list(LossNormType)}")
        return cls_weights, reg_weights
    def backward(self, top, propagate_down, bottom):
        pass

class LabelEncode(caffe.Layer):
    def setup(self, bottom, top):

        labels = bottom[0].data
        cared = bottom[1].data

        cls_targets = labels * cared # (1, 107136)
        cls_targets = cls_targets.astype(int)

        self.num_class = 1
        one_hot_targets = np.eye(self.num_class+1)[cls_targets]   #One_hot label -- make sure one hot class is <num_class+1>
        one_hot_targets = one_hot_targets[..., 1:]

        top[0].reshape(*one_hot_targets.shape) #reshape to caffe pattern
    def reshape(self, bottom, top):
        pass
    def forward(self, bottom, top):

        labels = bottom[0].data # (1, 107136)
        cared = bottom[1].data

        cls_targets = labels * cared
        cls_targets = cls_targets.astype(int)

        one_hot_targets = np.eye(self.num_class+1)[cls_targets]   #One_hot label -- make sure one hot class is <num_class+1>
        one_hot_targets = one_hot_targets[..., 1:]

        top[0].data[...] = one_hot_targets
    def backward(self, top, propagate_down, bottom):
        pass

#For Point-Wise model
class LabelEncodeV2(caffe.Layer):
    def setup(self, bottom, top):

        labels = bottom[0].data
        labels = labels.astype(int)
        labels = np.expand_dims(labels,-1)
        top[0].reshape(*labels.shape) #reshape to caffe pattern

    def reshape(self, bottom, top):
        pass
    def forward(self, bottom, top):

        labels = bottom[0].data # (1, 107136)
        labels = labels.astype(int)
        labels = np.expand_dims(labels,-1)
        top[0].data[...] = labels

    def backward(self, top, propagate_down, bottom):
        pass

class WeightFocalLoss(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.gamma = int(params['focusing_parameter'])
        self.alpha = params['alpha']
        self.batch_size = bottom[0].data.shape[0]

    def reshape(self, bottom, top):
        # check input dimensions match
        # if bottom[0].num != bottom[1].num:
        #     raise Exception("Infered scores and labels must have the same dimension.")
        top[0].reshape(1)
    def forward(self, bottom, top):
        self._p = bottom[0].data
        self.label = bottom[1].data
        self.cls_weights = bottom[2].data
        self.cls_weights = np.expand_dims(self.cls_weights,-1)

        log1p = np.log1p(np.exp(-np.abs(self._p))) #logits

        self._p_t =  1 / (1 + np.exp(-self._p)) # Compute sigmoid activations

        self.first = (1-self.label) * (1-self.alpha) + self.label * self.alpha

        self.second = (1-self.label) * ((self._p_t) ** self.gamma) + self.label * ((1 - self._p_t) ** self.gamma)

        self.sigmoid_cross_entropy = (1-self.label) * (log1p + np.clip(self._p, a_min=0, a_max=None)) + \
                                    self.label * (log1p - np.clip(self._p, a_min=None, a_max=0))

        logprobs = ((1-self.label) * self.first * self.second * self.sigmoid_cross_entropy) + \
                    (self.label * self.first * self.second * self.sigmoid_cross_entropy)

        top[0].data[...] = np.sum(logprobs*self.cls_weights) / self.batch_size

    def backward(self, top, propagate_down, bottom):

        dev_log1p = np.sign(self._p) * (1 / (np.exp(np.abs(self._p))+1))  # might fix divided by 0 x/|x| bug

        self.dev_sigmoid_cross_entropy =  (1-self.label) * (dev_log1p - np.where(self._p<=0, 0, 1))  + \
                                            self.label * (dev_log1p + np.where(self._p>=0, 0, 1))

        delta = (1-self.label) *  (self.first * self.second * (self.gamma * (1-self._p_t) * self.sigmoid_cross_entropy - self.dev_sigmoid_cross_entropy)) + \
            self.label * (-self.first * self.second * (self.gamma * self._p_t * self.sigmoid_cross_entropy + self.dev_sigmoid_cross_entropy))

        bottom[0].diff[...] = delta * self.cls_weights / self.batch_size

class WeightedSmoothL1Loss(caffe.Layer):
    def setup(self, bottom, top):
        self.sigma = 3
        self.encode_rad_error_by_sin = True
        self.batch_size = bottom[0].data.shape[0]
    def reshape(self, bottom, top):
        # check input dimensions match
        # if bottom[0].num != bottom[1].num:
        #     raise Exception("Infered scores and labels must have the same dimension.")
        top[0].reshape(1)
    def forward(self, bottom, top):
        box_preds = bottom[0].data
        reg_targets = bottom[1].data
        self.reg_weights = bottom[2].data
        self.reg_weights = np.expand_dims(self.reg_weights,-1)

        self.diff = box_preds - reg_targets

        #use sin_difference rad to sin
        if self.encode_rad_error_by_sin:
            diff_rot = self.diff[...,-1:].copy() #copy rotation without add sin
            self.sin_diff = np.sin(diff_rot)
            self.cos_diff = np.cos(diff_rot)
            self.diff[...,-1] = np.sin(self.diff[...,-1]) #use sin_difference

        self.abs_diff = np.abs(self.diff)
        #change from less than to less or equal
        self.cond = self.abs_diff <= (1/(self.sigma**2))
        loss = np.where(self.cond, 0.5 * self.sigma**2 * self.abs_diff**2,
                                    self.abs_diff - 0.5/self.sigma**2)

        reg_loss = loss * self.reg_weights

        top[0].data[...] = np.sum(reg_loss) / self.batch_size # * 2
    def backward(self, top, propagate_down, bottom):

        if self.encode_rad_error_by_sin:

            delta = np.where(self.cond[...,:-1], (self.sigma**2) * self.diff[...,:-1], np.sign(self.diff[...,:-1]))

            delta_rotation = np.where(self.cond[...,-1:], (self.sigma**2) * self.sin_diff * self.cos_diff,
                                        np.sign(self.sin_diff) * self.cos_diff) #if sign(0) is gonna be 0!

            delta = np.concatenate([delta, delta_rotation], axis=-1)

        else:
            delta = np.where(self.cond, (self.sigma**2) * self.diff, np.sign(self.diff))
        bottom[0].diff[...] = delta * self.reg_weights / self.batch_size# * 2

class FocalLoss(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.gamma = int(params['focusing_parameter'])
        self.alpha = params['alpha']
    def reshape(self, bottom, top):
        # check input dimensions match
        # if bottom[0].num != bottom[1].num:
        #     raise Exception("Infered scores and labels must have the same dimension.")
        top[0].reshape(1)
    def forward(self, bottom, top):
        self._p = bottom[0].data
        self.label = bottom[1].data

        log1p = np.log1p(np.exp(-np.abs(self._p))) #logits

        self._p_t =  1 / (1 + np.exp(-self._p)) # Compute sigmoid activations

        self.first = (1-self.label) * (1-self.alpha) + self.label * self.alpha

        self.second = (1-self.label) * (self._p_t ** self.gamma) + self.label * ((1 - self._p_t) ** self.gamma)

        self.sigmoid_cross_entropy = (1-self.label) * (log1p + np.clip(self._p, a_min=0, a_max=None)) + \
                                    self.label * (log1p - np.clip(self._p, a_min=None, a_max=0))

        logprobs = ((1-self.label) * self.first * self.second * self.sigmoid_cross_entropy) + \
                    (self.label * self.first * self.second * self.sigmoid_cross_entropy)

        top[0].data[...] = np.mean(logprobs)

    def backward(self, top, propagate_down, bottom):

        dev_log1p = np.sign(self._p) * (1 / (np.exp(np.abs(self._p))+1))  # might fix divided by 0 x/|x| bug

        self.dev_sigmoid_cross_entropy =  (1-self.label) * (dev_log1p - np.where(self._p<=0, 0, 1))  + \
                                            self.label * (dev_log1p + np.where(self._p>=0, 0, 1))

        delta = (1-self.label) *  (self.first * self.second * (self.gamma * (1-self._p_t) * self.sigmoid_cross_entropy - self.dev_sigmoid_cross_entropy)) + \
            self.label * (-self.first * self.second * (self.gamma * self._p_t * self.sigmoid_cross_entropy + self.dev_sigmoid_cross_entropy))

        bottom[0].diff[...] = delta

class DiceLoss(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.belta = params['belta'] #0.5
        self.alpha = params['alpha'] #0.5
        self.eps = 1e-5

    def reshape(self, bottom, top):
        top[0].reshape(1)
    def forward(self, bottom, top):
        self._p = bottom[0].data
        self.label = bottom[1].data

        self.tp = self._p * self.label

        self.fn = (1- self._p ) * self.label

        self.fp = self._p * (1 - self.label)

        self.union = self.tp + self.alpha * self.fn + self.belta * self.fp

        logprobs = (np.sum(self.tp) + self.eps) / (np.sum(self.union) + self.eps)

        top[0].data[...] = 1 - logprobs

    def backward(self, top, propagate_down, bottom):

        delta =  self.alpha * np.square(self.label) / (np.square(self.union) + self.eps)

        bottom[0].diff[...] = delta

#for v-net paper
class DiceLossV2(caffe.Layer):
    def setup(self, bottom, top):
        self.eps = 1e-5
        self.smooth = 1

    def reshape(self, bottom, top):
        top[0].reshape(1)
    def forward(self, bottom, top):
        self._p = bottom[0].data
        self.label = bottom[1].data

        self.inter = np.sum(self._p * self.label)
        self.union = np.sum(self._p + self.label)

        logprobs = (2 * self.inter + self.smooth) / (self.union + self.smooth)

        top[0].data[...] = logprobs

    def backward(self, top, propagate_down, bottom):

        delta = (self.label * (self.union) - 2 * self._p * (self.inter)) / (np.square(self.union) + self.eps)

        bottom[0].diff[...] = 2 * delta

class DiceLossV3(caffe.Layer):
    def setup(self, bottom, top):
        # params = eval(self.param_str)
        # self.belta = params['belta'] #0.5
        # self.alpha = params['alpha'] #0.5
        self.eps = 1e-5
        self.smooth = 1

    def reshape(self, bottom, top):
        top[0].reshape(1)
    def forward(self, bottom, top):
        self._p = bottom[0].data
        self.label = bottom[1].data

        self.tp = self._p * self.label
        self.union = self._p + self.label

        logprobs = (2 * np.sum(self.tp) + self.smooth) / (np.sum(self.union) + self.smooth)

        top[0].data[...] = logprobs

    def backward(self, top, propagate_down, bottom):

        delta =  2 * np.square(self.label) / (np.square(self.union) + self.eps)

        bottom[0].diff[...] = delta

class IoUSegLoss(caffe.Layer):
    def setup(self, bottom, top):
        # params = eval(self.param_str)
        # self.belta = params['belta'] #0.5
        # self.alpha = params['alpha'] #0.5
        self.eps = 1e-5

    def reshape(self, bottom, top):
        top[0].reshape(1)
    def forward(self, bottom, top):
        self._p = bottom[0].data
        self.label = bottom[1].data

        self.inter = self._p * self.label
        self.union = self._p + self.label - self.inter
        self.iou = self.inter/self.union

        logprobs = (np.sum(self.inter) + self.eps) / (np.sum(self.union) + self.eps)

        top[0].data[...] = 1 - logprobs

    def backward(self, top, propagate_down, bottom):

        delta = np.where(self.label>0, -1/(self.union + self.eps),  self.inter/(np.square(self.union)+ self.eps))

        bottom[0].diff[...] = delta

class DiceFocalLoss(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.gamma = int(params['focusing_parameter']) #2
        self.alpha = params['alpha'] #0.25
        self.dice_belta = params['dice_belta'] #0.5
        self.dice_alpha = params['dice_alpha'] #0.5
        self.lamda = params['lamda'] #trade off between focal and dice loss # 0.1, 0.5 , 1


    def reshape(self, bottom, top):
        # check input dimensions match
        # if bottom[0].num != bottom[1].num:
        #     raise Exception("Infered scores and labels must have the same dimension.")
        top[0].reshape(1)
    def forward(self, bottom, top):
        self._p = bottom[0].data
        self.label = bottom[1].data
        self.c = len(np.unique(self.label)) #no background

        ####################################Focal loss##########################
        self._p_t =  1 / (1 + np.exp(-self._p)) # Compute sigmoid activations

        self.first = (1-self.label) * (1-self.alpha) + self.label * self.alpha

        self.second = (1-self.label) * ((self._p_t) ** self.gamma) + self.label * ((1 - self._p_t) ** self.gamma)

        log1p = np.log1p(np.exp(-np.abs(self._p)))

        self.sigmoid_cross_entropy = (1-self.label) * (log1p + np.clip(self._p, a_min=0, a_max=None)) + \
                                    self.label * (log1p - np.clip(self._p, a_min=None, a_max=0))

        focal = ((1-self.label) * self.first * self.second * self.sigmoid_cross_entropy) + \
                    (self.label * self.first * self.second * self.sigmoid_cross_entropy)

        focal = np.mean(focal)

        ########################################Dice############################

        self.tp = np.sum(self._p * self.label)

        self.fn = np.sum((1- self._p ) * self.label)

        self.fp = np.sum(self._p * (1 - self.label))

        self.union = self.tp + self.alpha * self.fn + self.belta * self.fp

        dice = self.tp / (self.union + self.eps )

        top[0].data[...] = self.c - dice - self.lamda * focal #average fl

    def backward(self, top, propagate_down, bottom):

        dev_log1p = np.sign(self._p) * (1 / (np.exp(np.abs(self._p))+1))  # might fix divided by 0 x/|x| bug

        self.dev_sigmoid_cross_entropy =  (1-self.label) * (dev_log1p - np.where(self._p<=0, 0, 1))  + \
                                            self.label * (dev_log1p + np.where(self._p>=0, 0, 1))

        focal_delta = (1-self.label) *  (self.first * self.second * (self.gamma * (1-self._p_t) * self.sigmoid_cross_entropy - self.dev_sigmoid_cross_entropy)) + \
            self.label * (-self.first * self.second * (self.gamma * self._p_t * self.sigmoid_cross_entropy + self.dev_sigmoid_cross_entropy))

        ########################################Dice############################
        dev_tp = np.sum(self.label)

        dev_fn = np.sum(-self.label)

        dev_fp = np.sum(1-self.label)

        dice_delta =  (self.tp * (dev_tp + self.alpha * dev_fn + self.belta * dev_fp) - dev_tp * self.union) / ((self.union)**2 + self.eps)

        delta  = -(dice_delta + self.lamda * focal_delta)

        bottom[0].diff[...] = delta

class IoULoss(caffe.Layer):
    def setup(self, bottom, top):
        # params = eval(self.param_str)
        self.eps = 1e-5

    def reshape(self, bottom, top):
        top[0].reshape(1)
    def forward(self, bottom, top):

        pred = bottom[0].data
        gt_box = bottom[1].data
        self.points_label = bottom[2].data
        self.reg_weights = bottom[3].data
        self.reg_weights = np.expand_dims(self.reg_weights,-1)
        points = bottom[4].data[...,:3]

        pred = pred * self.points_label #if label==0 do not count iou

        self.pred_up = pred[..., 5:6]
        self.pred_down = pred[..., 2:3]
        self.pred_fwd = pred[..., 3:4]
        self.pred_bwd = pred[..., 0:1]
        self.pred_right = pred[..., 4:5]
        self.pred_left = pred[..., 1:2]

        self.gt_up = gt_box[..., 5:6]
        self.gt_down = gt_box[..., 2:3]
        self.gt_fwd = gt_box[..., 3:4]
        self.gt_bwd = gt_box[..., 0:1]
        self.gt_right = gt_box[..., 4:5]
        self.gt_left = gt_box[..., 1:2]

        pred_min_points = points - pred[..., :3]
        pred_max_points = points + pred[..., 3:-1]

        gt_min_points = points - gt_box[..., :3]
        gt_max_points = points + gt_box[..., 3:-1]

        pred_area = np.abs((self.pred_up + self.pred_down) * (self.pred_fwd + self.pred_bwd) * (self.pred_right + self.pred_left))
        # pred_area = np.prod(pred_max_points - pred_min_points, axis = -1)

        gt_area = (self.gt_up + self.gt_down) * (self.gt_fwd + self.gt_bwd) * (self.gt_right + self.gt_left)

        # self.inter_h = np.minimum(self.pred_up, self.gt_up) + np.minimum(self.pred_down, self.gt_down)
        # self.inter_w = np.minimum(self.pred_fwd, self.gt_fwd) + np.minimum(self.pred_bwd, self.gt_bwd)
        # self.inter_l = np.minimum(self.pred_right, self.gt_right) + np.minimum(self.pred_left, self.gt_left)

        h_pred_max = np.maximum(pred_max_points[..., 2:], pred_min_points[..., 2:])
        h_pred_min = np.minimum(pred_max_points[..., 2:], pred_min_points[..., 2:])

        w_pred_max = np.maximum(pred_max_points[..., 0:1], pred_min_points[..., 0:1])
        w_pred_min = np.minimum(pred_max_points[..., 0:1], pred_min_points[..., 0:1])

        l_pred_max = np.maximum(pred_max_points[..., 1:2], pred_min_points[..., 1:2])
        l_pred_min = np.minimum(pred_max_points[..., 1:2], pred_min_points[..., 1:2])

        self.inter_h = np.minimum(h_pred_max, gt_max_points[..., 2:]) - np.maximum(h_pred_min, gt_min_points[..., 2:])
        self.inter_w = np.minimum(w_pred_max, gt_max_points[..., 0:1]) - np.maximum(w_pred_min, gt_min_points[..., 0:1])
        self.inter_l = np.minimum(l_pred_max, gt_max_points[..., 1:2]) - np.maximum(l_pred_min, gt_min_points[..., 1:2])

        self.inter_h = np.clip(self.inter_h, a_min=0, a_max=None)
        self.inter_w = np.clip(self.inter_w, a_min=0, a_max=None)
        self.inter_l = np.clip(self.inter_l, a_min=0, a_max=None)

        # self.inter_h = np.minimum(pred_max_points[..., 2:], gt_max_points[..., 2:]) - np.maximum(pred_min_points[..., 2:], gt_min_points[..., 2:])
        # self.inter_w = np.minimum(pred_max_points[..., 0:1], gt_max_points[..., 0:1]) - np.maximum(pred_min_points[..., 0:1], gt_min_points[..., 0:1])
        # self.inter_l = np.minimum(pred_max_points[..., 1:2], gt_max_points[..., 1:2]) - np.maximum(pred_min_points[..., 1:2], gt_min_points[..., 1:2])

        # self.inter = np.clip(self.inter_h, a_min=0, a_max=None) * np.clip(self.inter_w, a_min=0, a_max=None) * np.clip(self.inter_l, a_min=0, a_max=None)
        self.inter = self.inter_h * self.inter_w * self.inter_l
        self.union = pred_area + gt_area - self.inter

        iou = (self.inter + self.eps) / (self.union + self.eps) #* self.points_label #if label==0 do not count iou
        # print("iou", np.unique(iou<=0, return_counts=True))
        # print("iou less than 0", iou[iou<=0])
        # print("self.inter <= 0", self.inter[iou<=0])
        # print("self.union less than 0", self.union[iou<=0])
        # print("pred_area less than 0", pred_area[iou<=0])
        # print("gt_area less than 0", gt_area[iou<=0])
        logprobs = -np.log(iou)

        top[0].data[...] = np.sum(logprobs * self.reg_weights)

    def backward(self, top, propagate_down, bottom):

        dev_h = (self.pred_left * self.pred_fwd) + (self.pred_left * self.pred_bwd) + (self.pred_right * self.pred_fwd) + (self.pred_right * self.pred_bwd)
        dev_w = (self.pred_left * self.pred_up) + (self.pred_left * self.pred_down) + (self.pred_right * self.pred_up) + (self.pred_right * self.pred_down)
        dev_l = (self.pred_up * self.pred_fwd) + (self.pred_up * self.pred_bwd) + (self.pred_down * self.pred_fwd) + (self.pred_down * self.pred_bwd)

        dev_iou_h = self.inter_w * self.inter_l
        dev_iou_w = self.inter_h * self.inter_l
        dev_iou_l = self.inter_w * self.inter_h

        # dev_iou_up = np.where(self.pred_up < self.gt_up, dev_iou_h, 0)
        # dev_iou_down = np.where(self.pred_down < self.gt_down, dev_iou_h, 0)
        # dev_iou_fwd = np.where(self.pred_fwd < self.gt_fwd, dev_iou_w, 0)
        # dev_iou_bwd = np.where(self.pred_bwd < self.gt_bwd, dev_iou_w, 0)
        # dev_iou_right = np.where(self.pred_right < self.gt_right, dev_iou_l, 0)
        # dev_iou_left = np.where(self.pred_left < self.gt_left, dev_iou_l, 0)

        cond_h = (self.pred_up < self.gt_up) + (self.pred_down < self.gt_down) # or condition
        cond_w = (self.pred_fwd < self.gt_fwd) + (self.pred_bwd < self.gt_bwd)
        cond_l = (self.pred_right < self.gt_right) + (self.pred_left < self.gt_left)

        dev_iou_h = np.where(cond_h, dev_iou_h, 0)
        dev_iou_w = np.where(cond_w, dev_iou_w, 0)
        dev_iou_l = np.where(cond_l, dev_iou_l, 0)


        second_term = (self.union + self.inter+ self.eps) / (self.union * self.inter + self.eps)
        first_term  =  1/(self.union + self.eps)

        # delta_up =  first_term * dev_h -  second_term * dev_iou_up
        # delta_down = first_term * dev_h - second_term * dev_iou_down
        # delta_fwd = first_term * dev_w - second_term * dev_iou_fwd
        # delta_bwd = first_term * dev_w - second_term * dev_iou_bwd
        # delta_right = first_term * dev_l - second_term * dev_iou_right
        # delta_left = first_term * dev_l - second_term * dev_iou_left

        delta_h =  first_term * dev_h -  second_term * dev_iou_h
        delta_w = first_term * dev_w - second_term * dev_iou_w
        delta_l = first_term * dev_l - second_term * dev_iou_l

        # delta = delta_up + delta_down + delta_fwd + delta_bwd + delta_right + delta_left

        delta = 2*delta_h + 2*delta_w + 2*delta_l

        bottom[0].diff[...] = delta * self.reg_weights

        # print("IoULoss backward", np.mean(delta * self.reg_weights))

class IoULossV2(caffe.Layer):
    def setup(self, bottom, top):
        self.eps = 1e-5
        self.sigma = 3
    def reshape(self, bottom, top):
        top[0].reshape(1)

    def forward(self, bottom, top):
        pred = bottom[0].data
        gt_box = bottom[1].data
        self.points_label = bottom[2].data
        self.reg_weights = bottom[3].data
        self.reg_weights = np.expand_dims(self.reg_weights,-1)
        # points = bottom[4].data[...,:3]

        pred = pred * self.points_label #if label==0 do not count iou
        # pred = np.where(pred<0, 0, pred) #ReLU

        self.pred_up = pred[..., 5:6]
        self.pred_down = pred[..., 2:3]
        self.pred_fwd = pred[..., 3:4]
        self.pred_bwd = pred[..., 0:1]
        self.pred_right = pred[..., 4:5]
        self.pred_left = pred[..., 1:2]
        self.pred_rot = pred[..., 6:]

        self.gt_up = gt_box[..., 5:6]
        self.gt_down = gt_box[..., 2:3]
        self.gt_fwd = gt_box[..., 3:4]
        self.gt_bwd = gt_box[..., 0:1]
        self.gt_right = gt_box[..., 4:5]
        self.gt_left = gt_box[..., 1:2]
        self.gt_rot = pred[..., 6:]

        self.diff = self.pred_rot - self.gt_rot
        self.abs_diff = np.abs(self.diff)
        self.cond = self.abs_diff <= (1/(self.sigma**2))
        rot_loss = np.where(self.cond, 0.5 * self.sigma**2 * self.abs_diff**2,
                                    self.abs_diff - 0.5/self.sigma**2)


        pred_area = (self.pred_up + self.pred_down) * (self.pred_fwd + self.pred_bwd) * (self.pred_right + self.pred_left)
        gt_area = (self.gt_up + self.gt_down) * (self.gt_fwd + self.gt_bwd) * (self.gt_right + self.gt_left)

        self.inter_h = np.minimum(self.pred_up, self.gt_up) + np.minimum(self.pred_down, self.gt_down)
        self.inter_w = np.minimum(self.pred_fwd, self.gt_fwd) + np.minimum(self.pred_bwd, self.gt_bwd)
        self.inter_l = np.minimum(self.pred_right, self.gt_right) + np.minimum(self.pred_left, self.gt_left)

        self.inter = self.inter_h * self.inter_w * self.inter_l
        self.union = pred_area + gt_area - self.inter

        iou = (self.inter + self.eps) / (self.union + self.eps) #* self.points_label #if label==0 do not count iou

        logprobs = -np.log(iou) + rot_loss

        top[0].data[...] = np.sum(logprobs * self.reg_weights)

    def backward(self, top, propagate_down, bottom):

        dev_h = (self.pred_left * self.pred_fwd) + (self.pred_left * self.pred_bwd) + (self.pred_right * self.pred_fwd) + (self.pred_right * self.pred_bwd)
        dev_w = (self.pred_left * self.pred_up) + (self.pred_left * self.pred_down) + (self.pred_right * self.pred_up) + (self.pred_right * self.pred_down)
        dev_l = (self.pred_up * self.pred_fwd) + (self.pred_up * self.pred_bwd) + (self.pred_down * self.pred_fwd) + (self.pred_down * self.pred_bwd)

        cond_h = (self.pred_up < self.gt_up) + (self.pred_down < self.gt_down) # or condition
        cond_w = (self.pred_fwd < self.gt_fwd) + (self.pred_bwd < self.gt_bwd)
        cond_l = (self.pred_right < self.gt_right) + (self.pred_left < self.gt_left)

        dev_iou_h = np.where(cond_h, self.inter_w * self.inter_l, 0)
        dev_iou_w = np.where(cond_w, self.inter_h * self.inter_l, 0)
        dev_iou_l = np.where(cond_l, self.inter_w * self.inter_h, 0)

        second_term = (self.union + self.inter) / (self.union * self.inter + self.eps)
        first_term  =  1/(self.union + self.eps)

        delta_h =  first_term * dev_h -  second_term * dev_iou_h
        delta_w = first_term * dev_w - second_term * dev_iou_w
        delta_l = first_term * dev_l - second_term * dev_iou_l

        # start_time = timeit.default_timer()

        rot_delta = np.where(self.cond, (self.sigma**2) * self.diff, np.sign(self.diff))
        delta = np.concatenate((delta_w, delta_l, delta_h), axis=-1)
        delta = np.repeat(delta, 2, axis=-1)
        delta = np.concatenate((delta,rotate), axis=-1)
        #
        # end_time = timeit.default_timer()
        # print('np.repeat forwards ran for {}s'.format((end_time-start_time)/60))

        bottom[0].diff[...] = delta * self.reg_weights

class IoULossV3(caffe.Layer):
    def setup(self, bottom, top):
        self.eps = 1e-5
        self.smooth = 1
    def reshape(self, bottom, top):
        top[0].reshape(1)

    def forward(self, bottom, top):
        pred = bottom[0].data
        gt_box = bottom[1].data
        self.points_label = bottom[2].data
        self.reg_weights = bottom[3].data
        self.reg_weights = np.expand_dims(self.reg_weights,-1)
        points = bottom[4].data[...,:3]

        pred = pred * self.points_label #if label==0 do not count iou
        # print("label", np.unique(self.points_label, return_index=True))
        # pred = np.where(pred<=0, 0, pred) #ReLU
        # print("pred", np.unique(self.points_label>0, return_index=True))

        self.pred_up = pred[..., 5:6]
        self.pred_down = pred[..., 2:3]
        self.pred_fwd = pred[..., 3:4]
        self.pred_bwd = pred[..., 0:1]
        self.pred_right = pred[..., 4:5]
        self.pred_left = pred[..., 1:2]

        self.gt_up = gt_box[..., 5:6]
        self.gt_down = gt_box[..., 2:3]
        self.gt_fwd = gt_box[..., 3:4]
        self.gt_bwd = gt_box[..., 0:1]
        self.gt_right = gt_box[..., 4:5]
        self.gt_left = gt_box[..., 1:2]

        pred_area = (self.pred_fwd + self.pred_bwd) * (self.pred_right + self.pred_left)
        # print("pred_area", pred_area[pred_area>4])
        gt_area = (self.gt_fwd + self.gt_bwd) * (self.gt_right + self.gt_left)
        # print("gt_area", gt_area[gt_area>0.8])

        # self.inter_h = np.minimum(self.pred_up, self.gt_up) + np.minimum(self.pred_down, self.gt_down)
        self.inter_w = np.minimum(self.pred_fwd, self.gt_fwd) + np.minimum(self.pred_bwd, self.gt_bwd)
        self.inter_l = np.minimum(self.pred_right, self.gt_right) + np.minimum(self.pred_left, self.gt_left)

        self.inter = self.inter_w * self.inter_l

        # print("self.inter > 0.4", self.inter[self.inter>0.4])

        self.union = pred_area + gt_area - self.inter

        iou = (self.inter + self.eps) / (self.union + self.eps) #* self.points_label #if label==0 do not count iou

        logprobs = -np.log(iou)

        top[0].data[...] = np.sum(logprobs * self.reg_weights)

    def backward(self, top, propagate_down, bottom):

        # dev_h = (self.pred_left * self.pred_fwd) + (self.pred_left * self.pred_bwd) + (self.pred_right * self.pred_fwd) + (self.pred_right * self.pred_bwd)
        dev_w = self.pred_left + self.pred_right
        dev_l = self.pred_fwd  + self.pred_bwd

        # dev_iou_h = self.inter_w * self.inter_l
        # dev_iou_w = self.inter_l
        # dev_iou_l = self.inter_w

        # cond_h = (self.pred_up < self.gt_up) + (self.pred_down < self.gt_down) # or condition
        cond_w = (self.pred_fwd < self.gt_fwd) + (self.pred_bwd < self.gt_bwd)
        cond_l = (self.pred_right < self.gt_right) + (self.pred_left < self.gt_left)

        # dev_iou_h = np.where(cond_h, dev_iou_h, 0)
        dev_iou_w = np.where(cond_w, self.inter_l, 0)
        dev_iou_l = np.where(cond_l, self.inter_w, 0)


        second_term = (self.union + self.inter) / (self.union * self.inter + self.eps)
        first_term  =  1/(self.union + self.eps)

        delta = np.zeros(shape=(1,9000,1))
        # delta_h =  first_term * dev_h -  second_term * dev_iou_h
        delta_w = first_term * dev_w - second_term * dev_iou_w # df, db
        delta_l = first_term * dev_l - second_term * dev_iou_l # dr, dl

        delta[..., 0:1] = delta_w #b
        delta[..., 1:2] = delta_l #l
        delta[..., 3:4] = delta_w #f
        delta[..., 4:5] = delta_l #r
        # delta = np.concatenate((),axis=-1)

        # delta = delta_w + delta_l

        bottom[0].diff[...] = delta * self.reg_weights

class CaLu(caffe.Layer):
    def setup(self, bottom, top):
        input_tensor = bottom[0].data
        top[0].reshape(*input_tensor.shape)
    def reshape(self, bottom, top):
        pass
    def forward(self, bottom, top):
        self.input_tensor = bottom[0].data

        # make positives
        self.t_mask = self.input_tensor < 0
        self.tensor = np.where(self.t_mask, 0, self.input_tensor)

        #activate
        self.tensor = 1 - 1/(1+self.tensor)

        top[0].data[...] = self.tensor

    def backward(self, top, propagate_down, bottom):
        diff = np.where(self.t_mask, 0, 1/np.square((1+self.input_tensor)))
        bottom[0].diff[...] = diff

class CaLuV2(caffe.Layer):
    def setup(self, bottom, top):
        input_tensor = bottom[0].data
        top[0].reshape(*input_tensor.shape)
    def reshape(self, bottom, top):
        pass
    def forward(self, bottom, top):
        self.input_tensor = bottom[0].data

        #activate
        self.tensor = 1 - 1/(1+self.input_tensor)

        top[0].data[...] = self.tensor

    def backward(self, top, propagate_down, bottom):
        diff = 1/np.square((1+self.input_tensor))
        bottom[0].diff[...] = diff

class BCLReshape(caffe.Layer):
    def setup(self, bottom, top):
        top_prev = bottom[0].data
        top_prev, top_lattice = self.reshape_func(top_prev)
        top[0].reshape(*top_prev.shape)
        top[1].reshape(*top_lattice.shape)
    def reshape(self, bottom, top):
        pass
    def forward(self, bottom, top):
        top_prev = bottom[0].data
        top_prev, top_lattice = self.reshape_func(top_prev)
        top[0].reshape(*top_prev.shape) #top_prev
        top[0].data[...] = top_prev
        top[1].reshape(*top_lattice.shape) #top_lattice
        top[1].data[...] = top_lattice
    def backward(self, top, propagate_down, bottom):
        pass
    def reshape_func(self, top_prev):
        top_prev = top_prev.transpose(0,2,1) #(1,N,C) -> (1,C,N)
        top_prev = np.expand_dims(top_prev,2) #(1,C,N) -> (1,C,,1,N)
        top_lattice = top_prev[:, :3, ...]
        return top_prev, top_lattice

class BCLReshapeV2(caffe.Layer):
    def setup(self, bottom, top):
        top_prev = bottom[0].data
        coords = bottom[1].data
        top_prev, top_lattice = self.reshape_func(top_prev, coords)
        top[0].reshape(*top_prev.shape)
        top[1].reshape(*top_lattice.shape)
    def reshape(self, bottom, top):
        pass
    def forward(self, bottom, top):
        top_prev = bottom[0].data
        coords = bottom[1].data
        top_prev, top_lattice = self.reshape_func(top_prev, coords)
        top[0].reshape(*top_prev.shape) #top_prev
        top[0].data[...] = top_prev
        top[1].reshape(*top_lattice.shape) #top_lattice
        top[1].data[...] = top_lattice
    def backward(self, top, propagate_down, bottom):
        pass
    def reshape_func(self, top_prev, coords):
        top_prev = top_prev.transpose(1,2,0) #(N,1,4) -> (1,4,N)
        top_prev = np.expand_dims(top_prev,2) #(1,4,N) -> (1,4,,1,N)
        coords = coords[:,1:][:,::-1].transpose() #coors in reverse order bzyx (V, C) -> (C,V)
        coords = np.expand_dims(coords,0) #(C,V)-> (1,C,V)
        coords = np.expand_dims(coords,2) #(1,C,V)-> (1,C,1,V)
        return top_prev, coords

class BCLReshapeV4(caffe.Layer):
    def setup(self, bottom, top):
        top_prev = bottom[0].data
        coords = bottom[1].data
        top_prev, top_lattice = self.reshape_func(top_prev, coords)
        top[0].reshape(*top_prev.shape)
        top[1].reshape(*top_lattice.shape)
    def reshape(self, bottom, top):
        pass
    def forward(self, bottom, top):
        top_prev = bottom[0].data
        coords = bottom[1].data
        top_prev, top_lattice = self.reshape_func(top_prev, coords)
        top[0].reshape(*top_prev.shape) #top_prev
        top[0].data[...] = top_prev
        top[1].reshape(*top_lattice.shape) #top_lattice
        top[1].data[...] = top_lattice
    def backward(self, top, propagate_down, bottom):
        pass
    def reshape_func(self, top_prev, coords):
        top_prev = top_prev.transpose(2,1,0) #(V,100,C) -> (C,100,V)
        top_prev = np.expand_dims(top_prev,0) #(C,100,V)-> (1,C,100,V)
        coords = coords[:,2:][:,::-1].transpose() #coors in reverse order bzyx, pillar no need z (V,C)
        coords = np.expand_dims(coords,0) #(C,V)-> (1,C,V)
        coords = np.expand_dims(coords,2) #(1,C,V)-> (1,C,1,V)
        coords = np.repeat(coords, top_prev.shape[-2], 2) #repeat 100
        return top_prev, coords

class BCLReshapeV5(caffe.Layer):
    def setup(self, bottom, top):
        top_prev = bottom[0].data
        coords = bottom[1].data
        top_prev, top_lattice = self.reshape_func(top_prev, coords)
        top[0].reshape(*top_prev.shape)
        top[1].reshape(*top_lattice.shape)
    def reshape(self, bottom, top):
        pass
    def forward(self, bottom, top):
        top_prev = bottom[0].data
        coords = bottom[1].data
        top_prev, top_lattice = self.reshape_func(top_prev, coords)
        top[0].reshape(*top_prev.shape) #top_prev
        top[0].data[...] = top_prev
        top[1].reshape(*top_lattice.shape) #top_lattice
        top[1].data[...] = top_lattice
    def backward(self, top, propagate_down, bottom):
        pass
    def reshape_func(self, top_prev, coords):
        top_prev = top_prev.transpose(2,1,0) #(V,N,C) -> (C,N,V)
        top_prev = np.expand_dims(top_prev,0) #(C,N,V)-> (1,C,N,V)
        coords = coords[:,2:][:,::-1].transpose() #coors in reverse order bzyx, pillar no need z (V,C)
        coords = np.expand_dims(coords,0) #(C,V)-> (1,C,V)
        coords = np.expand_dims(coords,2) #(1,C,V)-> (1,C,1,V)
        return top_prev, coords

class GlobalPooling(caffe.Layer):
    def setup(self, bottom, top):
        pass
    def reshape(self, bottom, top):
        n, c, p, h, w = bottom[0].data.shape
        top[0].reshape(*(n, c, h, w))
    def forward(self, bottom, top):
        n, c, p, h, w = bottom[0].data.shape
        self.max_loc = bottom[0].data.argmax(axis=2)
        top[0].data[...] = bottom[0].data.max(axis=2)
    def backward(self, top, propagate_down, bottom):
        n, c, h, w = top[0].diff.shape
        nn, cc, hh, ww = np.ix_(np.arange(n), np.arange(c), np.arange(h),np.arange(w))
        bottom[0].diff[...] = 0
        bottom[0].diff[nn, cc, self.max_loc, hh, ww] = top[0].diff

class LogLayer(caffe.Layer):
    def setup(self, bottom, top):
        in1 = bottom[0].data
        print("debug print", in1)
        print("debug print", in1.shape)
        top[0].reshape(*in1.shape)
    def reshape(self, bottom, top):
        pass
    def forward(self, bottom, top):
        in1 = bottom[0].data
        print("forward debug print", in1)
        print("forward debug print", in1.shape)
        top[0].reshape(*in1.shape)
        top[0].data[...] = in1
        pass
    def backward(self, top, propagate_down, bottom):
        pass

class ProbRenorm(caffe.Layer):
    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        clipped = bottom[0].data * bottom[1].data
        self.sc = 1.0 / (np.sum(clipped, axis=1, keepdims=True) + 1e-10)
        top[0].data[...] = clipped * self.sc

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = top[0].diff * bottom[1].data * self.sc

class PickAndScale(caffe.Layer):
    def setup(self, bottom, top):
        self.nch_out = len(self.param_str.split('_'))
        self.dims = []
        for f in self.param_str.split('_'):
            if f.find('*') >= 0:
                self.dims.append((int(f[:f.find('*')]), float(f[f.find('*') + 1:])))
            elif f.find('/') >= 0:
                self.dims.append((int(f[:f.find('/')]), 1.0 / float(f[f.find('/') + 1:])))

            else:
                self.dims.append((int(f), 1.0))

    def reshape(self, bottom, top):
        top[0].reshape(bottom[0].data.shape[0], self.nch_out, bottom[0].data.shape[2], bottom[0].data.shape[3])

    def forward(self, bottom, top):
        for i, (j, s) in enumerate(self.dims):
            top[0].data[:, i, :, :] = bottom[0].data[:, j, :, :] * s
    def backward(self, top, propagate_down, bottom):
        pass  # TODO NOT_YET_IMPLEMENTED
