
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

class LossNormType(Enum):
    NormByNumPositives = "norm_by_num_positives"
    NormByNumExamples = "norm_by_num_examples"
    NormByNumPosNeg = "norm_by_num_pos_neg"

class DataFeature(caffe.Layer):
    def setup(self, bottom, top):
        # BCL
        top[0].reshape(*(1, 4, 1, 12000)) # for pillar shape should (B,C=9,V,N=100), For second (B,C=1,V,N=5)
        # Pillar
        # top[0].reshape(*(1, 9, 2000, 100)) # for pillar shape should (B,C=9,V,N=100), For second (B,C=1,V,N=5)
    def reshape(self, bottom, top):
        pass
    def forward(self, bottom, top):
        pass
    def backward(self, top, propagate_down, bottom):
        pass

class LatticeFeature(caffe.Layer):
    def setup(self, bottom, top):
        # BCL
        top[0].reshape(*(12000,4)) #(V, C=4) # TODO:
        # Pillar
        # top[0].reshape(*(2000,4)) #(V, C=4) # TODO:
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
        anchors_fp_w = 1408 #1408 176
        anchors_fp_h = 1600
        fp_factor = 8
        self.fp_w = int(anchors_fp_w/fp_factor) #1408 176
        self.fp_h = int(anchors_fp_h/fp_factor) #1600 200

        self.phase = params['subset']
        self.keep_voxels = 12000
        self.generate_anchors_cachae = params['anchors_cachae'] #True FOR Pillar, False For BCL
        self.input_cfg, self.eval_input_cfg, self.model_cfg, train_cfg = load_config(model_dir, config_path)
        self.voxel_generator, self.target_assigner = build_network(self.model_cfg)
        self.dataloader = self.load_dataloader(self.input_cfg, self.eval_input_cfg, self.model_cfg)

        for example in self.dataloader:
            voxels = example['voxels']
            coors = example['coordinates']
            num_points = example['num_points']
            labels = example['labels']
            reg_targets =example['reg_targets']
            gt_coords = example['gt_boxes_coords']
            pos_reg_targets =example['pos_reg_targets']
            pos_labels =example['pos_labels']
            break

        self.data_iter = iter(self.dataloader)
        # For pillar
        # voxels= self.VoxelFeatureNet(voxels, coors, num_points) #(V,100,C) -> (B, C, V, N)

        #For new method
        reg_targets, labels = self.GroundTruth2FeatMap(labels, reg_targets, gt_coords, self.fp_w, self.fp_h)
        # reg_targets, labels = self.GroundTruth2FeatMap(pos_labels, pos_reg_targets, gt_coords, self.fp_w, self.fp_h)
        reg_targets = reg_targets.reshape(1,-1, reg_targets.shape[-1])
        labels = labels.reshape(1,-1)

        # For BCL
        voxels = voxels.transpose(2, 1, 0) #(V=fixed,N=1,C=4) -> (C=4, N=1, V=fixed)
        voxels = np.expand_dims(voxels, axis=0) # (C=4, N=1, V=fixed) -> (B=1, C=4, N=1, V=fixed)

        top[0].reshape(*voxels.shape)
        top[1].reshape(*coors.shape)
        top[2].reshape(*labels.shape) #[1 107136]
        top[3].reshape(*reg_targets.shape) #[]

    def reshape(self, bottom, top):
        pass
    def forward(self, bottom, top):
        try:
            example = next(self.data_iter)
        except Exception as e:
            print("\n[info] start a new epoch for {} data\n".format(self.phase))
            self.data_iter = iter(self.dataloader)
            example = next(self.data_iter)

        example.pop("metrics")
        voxels = example['voxels'] #(V,NPints,4)
        coors = example['coordinates'] #(V,4) bzyz
        num_points = example['num_points'] #(V,)
        labels = example['labels']
        reg_targets =example['reg_targets']
        gt_coords = example['gt_boxes_coords']
        pos_reg_targets =example['pos_reg_targets']
        pos_labels =example['pos_labels']

        #for new method
        reg_targets, labels = self.GroundTruth2FeatMap(labels, reg_targets, gt_coords, self.fp_w, self.fp_h)
        # reg_targets, labels = self.GroundTruth2FeatMap(pos_labels, pos_reg_targets, gt_coords, self.fp_w, self.fp_h)
        reg_targets = reg_targets.reshape(1,-1, pos_reg_targets.shape[-1])
        labels = labels.reshape(1,-1)

        # for pillar
        # voxels = self.VoxelFeatureNet(voxels, coors, num_points) #(V,100,C) -> (B, C, V, N)

        # for bcl
        voxels = voxels.transpose(2, 1, 0) #(V=fixed,N=1,C=4) -> (C=4, N=1, V=fixed)
        voxels = np.expand_dims(voxels, axis=0) # (C=4, N=1, V=fixed) -> (B=1, C=4, N=1, V=fixed)

        top[0].reshape(*voxels.shape)
        top[1].reshape(*coors.shape)
        top[2].reshape(*labels.shape) #[1 107136]
        top[3].reshape(*reg_targets.shape) #[]
        top[0].data[...] = voxels
        top[1].data[...] = coors
        top[2].data[...] = labels
        top[3].data[...] = reg_targets
        #print("[debug] train img idx : ", example["metadata"])

    def backward(self, top, propagate_down, bottom):
        pass
    def load_dataloader(self, input_cfg, eval_input_cfg, model_cfg):
        dataset = input_reader_builder.build(
            input_cfg,
            model_cfg,
            training=True,
            voxel_generator=self.voxel_generator,
            target_assigner=self.target_assigner,
            multi_gpu=False,
            generate_anchors_cachae=self.generate_anchors_cachae) #True FOR Pillar, False For BCL

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
    def VoxelFeatureNet(self, voxels, coors, num_points):
        point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
        voxel_size = [0.16, 0.16, 4]
        vx = voxel_size[0]
        vy = voxel_size[1]
        x_offset = vx / 2 + point_cloud_range[0]
        y_offset = vy / 2 + point_cloud_range[1]
        points_mean = np.sum(voxels[:, :, :3], axis=1, keepdims=True) / num_points.reshape(-1,1,1)
        f_cluster = voxels[:, :, :3] - points_mean

        f_center = np.zeros_like(voxels[:, :, :2]) # huge debug
        f_center[:, :, 0] = voxels[:, :, 0] - (np.expand_dims(coors[:, 3].astype(float), axis=1) * vx + x_offset)
        f_center[:, :, 1] = voxels[:, :, 1] - (np.expand_dims(coors[:, 2].astype(float), axis=1) * vy + y_offset)

        features_ls = [voxels, f_cluster, f_center]
        features = np.concatenate(features_ls, axis=-1) #[num_voxles, points_num, features]

        points_per_voxels = features.shape[1]
        mask = get_paddings_indicator_np(num_points, points_per_voxels, axis=0)
        mask = np.expand_dims(mask, axis=-1)
        features *= mask

        #(voxel, npoint, channel) -> (channel, voxels, npoints) -> (batch=1, channel=64, voxels, npoints=1)
        features = np.expand_dims(features, axis=0)
        features = features.transpose(0,3,1,2)
        return features
    def __GroundTruth2FeatMap(self, labels, reg_targets, gt_coords, fp_w, fp_h):
        gt_coords = gt_coords.squeeze()
        reg_targets = reg_targets.squeeze()
        nchannels = reg_targets.shape[-1]
        canvas = np.zeros(shape=(fp_w , fp_h, nchannels)).astype(int)  #(7, 176, 200) #reg_head = 7
        # label_canvas = -1*np.ones(shape=(fp_w, fp_h)).astype(int)  #(7, 176, 200) #reg_head = 7
        label_canvas = np.zeros(shape=(fp_w, fp_h)).astype(int)  #(7, 176, 200) #reg_head = 7
        pc_range = [0,-40,-3,70.4,40,1]
        # convert from real space coordinate to feature map index
        w = np.floor((gt_coords[:,0] * fp_w) / 70.4).astype(int)
        # Add half of the fp_h to convert negative y to positive fp_h
        y = np.floor((gt_coords[:,1] * fp_h) / 80 + fp_h/2).astype(int)
        print("reg_targets", reg_targets.shape)
        print("canvas[:,w,y,:]", canvas[w,y,:].shape)
        canvas[w,y,:] = reg_targets
        label_canvas[w,y] = labels
        return canvas, label_canvas
    def GroundTruth2FeatMap(self, labels, reg_targets, gt_coords, fp_w, fp_h):
        num_anchor_per_loc=2
        gt_coords = gt_coords.squeeze()
        reg_targets = reg_targets.squeeze()
        labels = labels.squeeze()
        nchannels = reg_targets.shape[-1]
        reg_targets = reg_targets.reshape(num_anchor_per_loc,-1, nchannels)
        labels = labels.reshape(num_anchor_per_loc, -1)
        canvas = np.zeros(shape=(num_anchor_per_loc, fp_w , fp_h, nchannels)).astype(int)  #(7, 176, 200) #reg_head = 7
        # label_canvas = -1*np.ones(shape=(fp_w, fp_h)).astype(int)  #(7, 176, 200) #reg_head = 7
        # label_canvas = np.zeros(shape=(num_anchor_per_loc, fp_w, fp_h)).astype(int)  #(7, 176, 200) #reg_head = 7
        label_canvas = -1*np.ones(shape=(num_anchor_per_loc, fp_w, fp_h)).astype(int)  #(7, 176, 200) #reg_head = 7
        pc_range = [0,-40,-3,70.4,40,1]
        # convert from real space coordinate to feature map index
        w = np.floor((gt_coords[:,0] * fp_w) / 70.4).astype(int)
        # Add half of the fp_h to convert negative y to positive fp_h
        y = np.floor((gt_coords[:,1] * fp_h) / 80 + fp_h/2).astype(int)
        # print("reg_targets", reg_targets.shape)
        # print("canvas[:,w,y,:]", canvas[:,w,y,:].shape)
        canvas[:,w,y,:] = reg_targets
        # print("label_canvas[:, w,y]", label_canvas[:,w,y].shape)
        # print("labels", labels.shape)
        label_canvas[:,w,y] = labels
        return canvas, label_canvas


class PointPillarsScatter(caffe.Layer):
    def setup(self, bottom, top):
        param = eval(self.param_str)
        output_shape = param['output_shape']
        self.batch_size = 1
        self.ny = output_shape[2]
        self.nx = output_shape[3]
        self.nchannels = output_shape[4]

        voxel_features = bottom[0].data
        voxel_features = np.squeeze(voxel_features) #(1, 64, voxel, 1) -> (64,Voxel)
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
        # bottom[0].diff[...] = np.expand_dims(diff, axis=2) #need match with input features shape
        bottom[0].diff[...] = np.expand_dims(diff, axis=-1) #need match with input features shape
    def ScatterNet(self, voxel_features, coords, nchannels, feature_map_x, feature_map_y):
        batch_canvas = []
        for batch_itt in range(self.batch_size):
            canvas = np.zeros(shape=(nchannels, feature_map_x * feature_map_y)) #(nchannels,-1)
            batch_mask = coords[:, 0] == batch_itt
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 2] * feature_map_x + this_coords[:, 3]
            indices = indices.astype(int)
            voxels = voxel_features[:, batch_mask]
            canvas[:, indices] = voxels
            batch_canvas.append(canvas)

        if len(batch_canvas)>1:
            batch_canvas = np.stack(batch_canvas, 0) # stack is too slow!!!
        else:
            batch_canvas = batch_canvas[0]
        batch_canvas = batch_canvas.reshape(self.batch_size, nchannels, feature_map_y, feature_map_x)
        return batch_canvas, indices

class PointScatter(caffe.Layer):
    def setup(self, bottom, top):
        param = eval(self.param_str)
        output_shape = param['output_shape']
        self.batch_size = 1
        self.ny = output_shape[0]
        self.nx = output_shape[1]
        self.nchannels = output_shape[2]

        voxel_features = bottom[0].data
        voxel_features = np.squeeze(voxel_features) #(1, 64, 1, voxel) -> (64,Voxel)
        coords = bottom[1].data # lattic feature xyz
        coords = np.squeeze(coords)
        batch_canvas, _ = self.ScatterNet(voxel_features, coords, self.nchannels, self.nx, self.ny)
        top[0].reshape(*batch_canvas.shape)
    def reshape(self, bottom, top):
        pass
    def forward(self, bottom, top):
        voxel_features = bottom[0].data #(1,64,-1,1)
        voxel_features = np.squeeze(voxel_features) #(1, 64, -1, 1) -> (64,-1) #(12000, 64)
        coords = bottom[1].data #(3, 12000)
        coords = np.squeeze(coords)
        #start_time = timeit.default_timer()
        batch_canvas, self.indices = self.ScatterNet(voxel_features, coords, self.nchannels, self.nx, self.ny)
        #end_time = timeit.default_timer()
        #print('PointScatter forwards ran for {}s'.format((end_time-start_time)/60))
        top[0].data[...] = batch_canvas
    def backward(self, top, propagate_down, bottom):
        diff = top[0].diff.reshape(self.batch_size, self.nchannels, self.nx * self.ny)[:,:,self.indices]
        bottom[0].diff[...] = np.expand_dims(diff, axis=2) #need match with input features shape
    def ScatterNet(self, voxel_features, coords, nchannels, feature_map_x, feature_map_y):
        canvas = np.zeros(shape=(nchannels, feature_map_x * feature_map_y)) #(nchannels,-1)
        indices = coords[1, :] * feature_map_x + coords[0, :] # y*feature_map_x + x
        indices = indices.astype(int)
        canvas[:, indices] = voxel_features
        canvas = canvas.reshape(self.batch_size, nchannels, feature_map_y, feature_map_x)
        return canvas, indices

    def Voxel3DStack2D(self, voxel_features, coors_3d):
        # print("coords, ", coors_3d.shape)
        # print("voxel_features, ", voxel_features.shape)
        # coords_xy = np.delete(coors_3d, obj=1, axis=1) #delete z column
        voxel_group = npi.group_by(coors_3d) #features mean
        # coors_idx, voxel_features = voxel_group.mode(voxel_features) #features max
        coors_idx, voxel_features = voxel_group.max(voxel_features) #features max

        return voxel_features, coors_idx, voxel_group
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
        # NOTE: 26th may: change from less than to less or equal
        self.cond = self.abs_diff <= (1/(self.sigma**2))
        loss = np.where(self.cond, 0.5 * self.sigma**2 * self.abs_diff**2,
                                    self.abs_diff - 0.5/self.sigma**2)

        reg_loss = loss * self.reg_weights

        top[0].data[...] = np.sum(reg_loss) * 2
    def backward(self, top, propagate_down, bottom):

        if self.encode_rad_error_by_sin:

            delta = np.where(self.cond[...,:-1], (self.sigma**2) * self.diff[...,:-1], np.sign(self.diff[...,:-1]))

            delta_rotation = np.where(self.cond[...,-1:], (self.sigma**2) * self.sin_diff * self.cos_diff,
                                        np.sign(self.sin_diff) * self.cos_diff) #if sign(0) is gonna be 0!!!!!!!!!!!!!!

            delta = np.concatenate([delta, delta_rotation], axis=-1)

        else:
            delta = np.where(self.cond, (self.sigma**2) * self.diff, np.sign(self.diff))
        bottom[0].diff[...] = delta * self.reg_weights * 2

class BCLReshape(caffe.Layer):
    def setup(self, bottom, top):
        in0 = self.reshape_func(bottom[0].data)
        top[0].reshape(*in0.shape)
    def reshape(self, bottom, top):
        pass
    def forward(self, bottom, top):
        in0 = self.reshape_func(bottom[0].data)
        top[0].reshape(*in0.shape)
        top[0].data[...] = in0
    def backward(self, top, propagate_down, bottom):
        pass
    def reshape_func(self, in0):
        _in0 = in0[:,1:][:,::-1].transpose() #coors in reverse order bzyx (V, C) -> (C,V)
        _in0 = np.expand_dims(_in0,0) #(C,V)-> (1,C,V)
        _in0 = np.expand_dims(_in0,2) #(1,C,V)-> (1,C,1,V) C=XYZ
        return _in0

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
