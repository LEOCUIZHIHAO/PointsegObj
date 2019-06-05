
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
        # BCL
        if is_segmentation:
            top[0].reshape(*(1, seg_keep_points_eval, 4)) # for pillar shape should (B,C=9,V,N=100), For second (B,C=1,V,N=5)
        else:
            top[0].reshape(*(bcl_keep_voxels_eval, num_points_per_voxel, 4)) #pillar
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

        """shuffle car seg points"""
        indices = np.arange(seg_labels.shape[1])
        np.random.shuffle(indices)
        seg_points = seg_points[:,indices]
        seg_labels = seg_labels[:,indices]

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
            multi_gpu=False) #True FOR Pillar, False For BCL

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
        top[0].reshape(*voxels.shape)
        top[1].reshape(*coors.shape)
        top[2].reshape(*labels.shape)
        top[3].reshape(*reg_targets.shape)
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
            multi_gpu=False) #True FOR Pillar, False For BCL

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

class WeightFocalLoss(caffe.Layer):
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
        self.cls_weights = bottom[2].data
        self.cls_weights = np.expand_dims(self.cls_weights,-1)

        self._p_t =  1 / (1 + np.exp(-self._p)) # Compute sigmoid activations

        self.first = (1-self.label) * (1-self.alpha) + self.label * self.alpha

        self.second = (1-self.label) * ((self._p_t) ** self.gamma) + self.label * ((1 - self._p_t) ** self.gamma)

        log1p = np.log1p(np.exp(-np.abs(self._p)))

        self.sigmoid_cross_entropy = (1-self.label) * (log1p + np.clip(self._p, a_min=0, a_max=None)) + \
                                    self.label * (log1p - np.clip(self._p, a_min=None, a_max=0))

        logprobs = ((1-self.label) * self.first * self.second * self.sigmoid_cross_entropy) + \
                    (self.label * self.first * self.second * self.sigmoid_cross_entropy)

        top[0].data[...] = np.sum(logprobs*self.cls_weights)
    def backward(self, top, propagate_down, bottom):

        dev_log1p = np.sign(self._p) * (1 / (np.exp(np.abs(self._p))+1))  # might fix divided by 0 x/|x| bug

        self.dev_sigmoid_cross_entropy =  (1-self.label) * (dev_log1p - np.where(self._p<=0, 0, 1))  + \
                                            self.label * (dev_log1p + np.where(self._p>=0, 0, 1))

        delta = (1-self.label) *  (self.first * self.second * (self.gamma * (1-self._p_t) * self.sigmoid_cross_entropy - self.dev_sigmoid_cross_entropy)) + \
            self.label * (-self.first * self.second * (self.gamma * self._p_t * self.sigmoid_cross_entropy + self.dev_sigmoid_cross_entropy))

        bottom[0].diff[...] = delta * self.cls_weights

class WeightedSmoothL1Loss(caffe.Layer):
    def setup(self, bottom, top):
        self.sigma = 3
        self.encode_rad_error_by_sin = True
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

        top[0].data[...] = np.sum(reg_loss) * 2
    def backward(self, top, propagate_down, bottom):

        if self.encode_rad_error_by_sin:

            delta = np.where(self.cond[...,:-1], (self.sigma**2) * self.diff[...,:-1], np.sign(self.diff[...,:-1]))

            delta_rotation = np.where(self.cond[...,-1:], (self.sigma**2) * self.sin_diff * self.cos_diff,
                                        np.sign(self.sin_diff) * self.cos_diff) #if sign(0) is gonna be 0!

            delta = np.concatenate([delta, delta_rotation], axis=-1)

        else:
            delta = np.where(self.cond, (self.sigma**2) * self.diff, np.sign(self.diff))
        bottom[0].diff[...] = delta * self.reg_weights * 2

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

        self._p_t =  1 / (1 + np.exp(-self._p)) # Compute sigmoid activations

        self.first = (1-self.label) * (1-self.alpha) + self.label * self.alpha

        self.second = (1-self.label) * ((self._p_t) ** self.gamma) + self.label * ((1 - self._p_t) ** self.gamma)

        log1p = np.log1p(np.exp(-np.abs(self._p)))

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
        self.thresh = 0.5
        self.eps = 1e-5

    def reshape(self, bottom, top):
        top[0].reshape(1)
    def forward(self, bottom, top):
        self._p = bottom[0].data
        self.label = bottom[1].data

        self._p_t =  1 / (1 + np.exp(-self._p)) # Compute sigmoid activations

        # self._p_t = np.where(self._p_t > self.thresh, 1, 0)

        self.tp = self._p_t * self.label

        self.fn = (1- self._p_t ) * self.label

        self.fp = self._p_t * (1 - self.label)

        logprobs = np.sum(self.tp / (self.tp + self.alpha * self.fn + self.belta * self.fp + self.eps ))

        top[0].data[...] = logprobs

    def backward(self, top, propagate_down, bottom):

        delta = self.label * (self.alpha * self._p_t * (1- self._p_t)) / ((self._p_t + self.alpha * (1 - self._p_t))**2)

        # delta = self.label * (self.alpha + self._p_t) / ((self._p_t + self.alpha * (1 - self._p_t))**2) #with caffe sigmoid

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

        self.tp = np.sum(self._p_t * self.label)

        self.fn = np.sum((1- self._p_t ) * self.label)

        self.fp = np.sum(self._p_t * (1 - self.label))

        dice = self.tp / (self.tp + self.dice_alpha * self.fn + self.dice_belta * self.fp)

        top[0].data[...] = self.c - dice - self.lamda * focal #average fl

    def backward(self, top, propagate_down, bottom):

        dev_log1p = np.sign(self._p) * (1 / (np.exp(np.abs(self._p))+1))  # might fix divided by 0 x/|x| bug

        self.dev_sigmoid_cross_entropy =  (1-self.label) * (dev_log1p - np.where(self._p<=0, 0, 1))  + \
                                            self.label * (dev_log1p + np.where(self._p>=0, 0, 1))

        focal_delta = (1-self.label) *  (self.first * self.second * (self.gamma * (1-self._p_t) * self.sigmoid_cross_entropy - self.dev_sigmoid_cross_entropy)) + \
            self.label * (-self.first * self.second * (self.gamma * self._p_t * self.sigmoid_cross_entropy + self.dev_sigmoid_cross_entropy))

        term_1 = self._p_t + self.dice_alpha * (1 - self._p_t)

        term_2 = self._p_t * (1-self._p_t)

        # dice_delta = self.label * (self.dice_alpha * self._p_t * (1- self._p_t)) / (term_1)**2)

        dice_delta = self.label * (term_2 * term_1 - self._p_t * (1 - self.dice_alpha) * term_2) / ((term_1)**2)

        delta  = focal_delta

        bottom[0].diff[...] = delta

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
        top_prev = top_prev.transpose(0,2,1) #(1,N,4) -> (1,4,N)
        top_prev = np.expand_dims(top_prev,2) #(1,4,N) -> (1,4,,1,N)
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

class BCLReshapeV3(caffe.Layer):
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
        top_prev = top_prev.transpose(1,2,0) #(1,N,4) -> (1,4,N)
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
        top[0].reshape(*bottom[0].data.shape)
    def forward(self, bottom, top):
        n, c, h, w = bottom[0].data.shape
        self.max_loc = bottom[0].data.reshape(n, c, h*w).argmax(axis=2)
        top[0].data[...] = bottom[0].data.max(axis=(2, 3), keepdims=True)
    def backward(self, top, propagate_down, bottom):
        n, c, h, w = top[0].diff.shape
        nn, cc = np.ix_(np.arange(n), np.arange(c))
        bottom[0].diff[...] = 0
        bottom[0].diff.reshape(n, c, -1)[nn, cc, self.max_loc] = top[0].diff.sum(axis=(2, 3))

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
