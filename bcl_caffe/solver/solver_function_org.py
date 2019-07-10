import os
import tempfile
import numpy as np
from pathlib import Path
import time
import torch
import caffe
from caffe.proto import caffe_pb2
from tqdm import tqdm

from google.protobuf import text_format
import second.data.kitti_common as kitti
from second.builder import target_assigner_builder, voxel_builder
from second.pytorch.core import box_torch_ops
from second.data.preprocess import merge_second_batch, merge_second_batch_multigpu, PointRandomChoiceV2
from second.protos import pipeline_pb2
from second.pytorch.builder import box_coder_builder, input_reader_builder
from second.pytorch.models.voxel_encoder import get_paddings_indicator_np #for pillar
from second.utils.log_tool import SimpleModelLog
from tools import some_useful_tools as sut
from second.core import box_np_ops
import pickle

import cv2
from second.utils import simplevis

def get_prototxt(solver_proto, save_path=None):
    if save_path:
        f = open(save_path, mode='w+')
    else:
        f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    f.write(str(solver_proto))
    f.close()

    return f.name

class SolverWrapper:
    def __init__(self,  train_net,
                        test_net,
                        pretrain = None,
                        prefix = "pp",
                        model_dir=None,
                        config_path=None,
                        ### Solver Params ###
                        solver_type='ADAM',
                        weight_decay=0.001,
                        lr_policy='step',
                        warmup_step=0,
                        warmup_start_lr=0,
                        lr_ratio=1,
                        end_ratio=1,
                        base_lr=0.002,
                        max_lr=0.002,
                        momentum = 0.9,
                        max_momentum = 0,
                        cycle_steps=1856,
                        gamma=0.8, #0.1 for lr_policy
                        stepsize=100,
                        test_iter=3769,
                        test_interval=50, #set test_interval to 999999999 if not it will auto run validation
                        max_iter=1e5,
                        iter_size=1,
                        snapshot=9999,
                        display=1,
                        random_seed=0,
                        debug_info=False,
                        create_prototxt=True,
                        args=None):
        """Initialize the SolverWrapper."""
        self.test_net = test_net
        self.solver_param = caffe_pb2.SolverParameter()
        self.solver_param.train_net = train_net
        self.solver_param.test_initialization = False


        self.solver_param.display = display
        self.solver_param.warmup_step = warmup_step
        self.solver_param.warmup_start_lr = warmup_start_lr
        self.solver_param.lr_ratio = lr_ratio
        self.solver_param.end_ratio = end_ratio
        self.solver_param.base_lr = base_lr
        self.solver_param.max_lr = max_lr
        self.solver_param.cycle_steps = cycle_steps
        self.solver_param.max_momentum = max_momentum
        self.solver_param.lr_policy = lr_policy  # "fixed" #exp
        self.solver_param.gamma = gamma
        self.solver_param.stepsize = stepsize

        self.solver_param.display = display
        self.solver_param.max_iter = max_iter
        self.solver_param.iter_size = iter_size
        self.solver_param.snapshot = snapshot
        self.solver_param.snapshot_prefix = os.path.join(model_dir, prefix)
        self.solver_param.random_seed = random_seed

        self.solver_param.solver_mode = caffe_pb2.SolverParameter.GPU
        if solver_type is 'SGD':
            print("[Info] SGD Solver >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            self.solver_param.solver_type = caffe_pb2.SolverParameter.SGD
        elif solver_type is 'ADAM':
            print("[Info] ADAM Solver >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            self.solver_param.solver_type = caffe_pb2.SolverParameter.ADAM
        self.solver_param.momentum = momentum
        self.solver_param.momentum2 = 0.999

        self.solver_param.weight_decay = weight_decay
        self.solver_param.debug_info = debug_info

        if create_prototxt:
            solver_prototxt = get_prototxt(self.solver_param, os.path.join(model_dir, 'solver.prototxt'))
            print(solver_prototxt)

        self.solver = caffe.get_solver(solver_prototxt)
        self.test_interval = test_interval

        '''Model config parameter Initialization'''
        self.args = args
        self.model_dir, self.config_path = model_dir, config_path
        _, eval_input_cfg, model_cfg, train_cfg = load_config(self.model_dir, self.config_path)
        voxel_generator, self.target_assigner = build_network(model_cfg)
        self.dataloader, self.eval_dataset = load_dataloader(eval_input_cfg, model_cfg, voxel_generator,
                                        self.target_assigner, args = args)
        self.model_cfg = model_cfg
        # NOTE: Could have problem, if eval no good check here
        self._box_coder=self.target_assigner.box_coder
        classes_cfg = model_cfg.target_assigner.class_settings
        self._num_class = len(classes_cfg)
        self._encode_background_as_zeros = model_cfg.encode_background_as_zeros
        self._nms_class_agnostic=model_cfg.nms_class_agnostic
        self._use_multi_class_nms=[c.use_multi_class_nms for c in classes_cfg]
        self._nms_pre_max_sizes=[c.nms_pre_max_size for c in classes_cfg]
        self._multiclass_nms=all(self._use_multi_class_nms)
        self._use_sigmoid_score=model_cfg.use_sigmoid_score
        self._num_anchor_per_loc=self.target_assigner.num_anchors_per_location

        self._use_rotate_nms=[c.use_rotate_nms for c in classes_cfg]  #False for pillar, True for second
        self._nms_post_max_sizes=[c.nms_post_max_size for c in classes_cfg] #300 for pillar, 100 for second
        self._nms_score_thresholds=[c.nms_score_threshold for c in classes_cfg] # 0.4 in submit, but 0.3 can get better hard performance #pillar use 0.05, second 0.3
        self._nms_iou_thresholds=[c.nms_iou_threshold for c in classes_cfg] ## NOTE: double check #pillar use 0.5, second use 0.01
        self._post_center_range=list(model_cfg.post_center_limit_range) ## NOTE: double check
        self._use_direction_classifier=model_cfg.use_direction_classifier ## NOTE: double check
        path = pretrain["path"]
        weight = pretrain["weight"]
        skip_layer = pretrain["skip_layer"] #list skip layer name
        if path != None and weight != None:
            self.load_pretrained_caffe_weight(path, weight, skip_layer)

        #self.model_logging = log_function(self.model_dir, self.config_path)
        ################################Log#####################################
        self.model_logging = SimpleModelLog(self.model_dir)
        self.model_logging.open()

        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(self.config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)

        self.model_logging.log_text(proto_str + "\n", 0, tag="config")
        self.model_logging.close()
        ########################################################################

        #Log loss
        ########################################################################
        self.log_loss_path = Path(self.model_dir) / f'log_loss.txt'
        ########################################################################

    def load_pretrained_caffe_weight(self, path, weight_path, skip_layer):
        assert isinstance(skip_layer, list) #pass skip list name inlist
        print("### Start loading pretrained caffe weights")
        old_proto_path = os.path.join(path, "train.prototxt")
        old_weight_path = os.path.join(path, weight_path)
        print("### Load old caffe model")
        old_net = caffe.Net(old_proto_path, old_weight_path, caffe.TRAIN)
        print("### Start loading model layers")
        for layer in old_net.params.keys():
            if layer in skip_layer:
                print("### Skipped layer: " + layer)
                continue
            param_length = len(old_net.params[layer])
            print("# Loading layer: " + layer)
            for index in range(param_length):
                try:
                    self.solver.net.params[layer][index].data[...] = old_net.params[layer][index].data[...]
                except Exception as e:
                    print(e)
                    print("!! Cannot load layer: " + layer)
                    continue
        print("### Finish loading pretrained model")

    def eval_model(self):

        self.model_logging.open() #logging

        cur_iter = self.solver.iter
        # if self.args["segmentation"]:
        # self.segmentation_evaluation(cur_iter)
        # else:
        self.object_detection_evaluation(cur_iter)

        self.model_logging.close()

    def train_model(self):
        cur_iter = self.solver.iter
        while cur_iter < self.solver_param.max_iter:
            for i in range(self.test_interval):
                #####For Restrore check
                if cur_iter + i >= self.solver_param.max_iter:
                    break

                self.solver.step(1)

                if (self.solver.iter-1) % self.solver_param.display == 0:
                    with open(self.log_loss_path, "a") as f:
                        lr = self.solver.lr
                        cls_loss = self.solver.net.blobs['cls_loss'].data[...][0]
                        reg_loss = self.solver.net.blobs['reg_loss'].data[...][0]
                        f.write("steps={},".format(self.solver.iter-1))
                        f.write("lr={:.8f},".format(lr))
                        f.write("cls_loss={:.3f},".format(cls_loss))
                        f.write("reg_loss={:.3f}".format(reg_loss))
                        f.write("\n")

            sut.plot_graph(self.log_loss_path, self.model_dir)
            self.eval_model()
            sut.clear_caffemodel(self.model_dir, 8) #KEPP Last 8
            cur_iter += self.test_interval

    def lr_finder(self):
        lr_finder_path = Path(self.model_dir) / f'log_lrf.txt'
        for _ in range(self.solver_param.max_iter):
            self.solver.step(1)

            if (self.solver.iter-1) % self.solver_param.display == 0:
                with open(lr_finder_path, "a") as f:
                    lr = self.solver.lr
                    cls_loss = self.solver.net.blobs['cls_loss'].data[...][0]
                    reg_loss = self.solver.net.blobs['reg_loss'].data[...][0]
                    f.write("steps={},".format(self.solver.iter-1))
                    f.write("lr={:.8f},".format(lr))
                    f.write("cls_loss={:.3f},".format(cls_loss))
                    f.write("reg_loss={:.3f}".format(reg_loss))
                    f.write("\n")

        sut.plot_graph(lr_finder_path, self.model_dir, name='Finder')

    def demo(self):
        print("[Info] Initialize test net\n")
        test_net = caffe.Net(self.test_net, caffe.TEST)
        test_net.share_with(self.solver.net)
        print("[Info] Loaded train net weights \n")
        data_dir = "./debug_tool/experiment/data/2011_09_26_drive_0009_sync/velodyne_points/data"
        point_cloud_files = os.listdir(data_dir)
        point_cloud_files.sort()
        obj_detections = []
        # Voxel generator
        pc_range = self.model_cfg.voxel_generator.point_cloud_range
        class_settings = self.model_cfg.target_assigner.class_settings[0]
        size = class_settings.anchor_generator_range.sizes
        rotations = class_settings.anchor_generator_range.rotations
        anchor_ranges = np.array(class_settings.anchor_generator_range.anchor_ranges)
        voxel_size = np.array(self.model_cfg.voxel_generator.voxel_size)
        out_size_factor = self.model_cfg.middle_feature_extractor.downsample_factor
        point_cloud_range = np.array(pc_range)
        grid_size = (
            point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)
        feature_map_size = grid_size[:2] // out_size_factor
        feature_map_size = [*feature_map_size, 1][::-1]
        for file in tqdm(point_cloud_files):
            file_path = os.path.join(data_dir, file)
            # with open(file_path, "rb") as f:
            #     points = f.read()
            points = np.fromfile(file_path, dtype = np.float32).reshape(-1,4)
            # NOTE: Prior seg preprocessing ###
            points = box_np_ops.remove_out_pc_range_points(points, pc_range)
            # Data sampling
            seg_keep_points = 20000
            points = PointRandomChoiceV2(points, seg_keep_points) #Repeat sample according points distance
            points = np.expand_dims(points, 0)
            ###
            # Anchor Generator
            anchors = box_np_ops.create_anchors_3d_range(feature_map_size, anchor_ranges,
                                                        size, rotations)
            # input
            test_net.blobs['top_prev'].reshape(*points.shape)
            test_net.blobs['top_prev'].data[...] = points
            test_net.forward()

            # segmentation output
            try:
                seg_preds = test_net.blobs['seg_output'].data[...].squeeze()
                points = np.squeeze(points)
                pred_thresh = 0.5
                pd_points = points[seg_preds >= pred_thresh,:]
                with open(os.path.join('./debug_tool/experiment',"pd_points.pkl") , 'ab') as f:
                    pickle.dump(pd_points,f)
            except Exception as e:
                pass

            with open(os.path.join('./debug_tool/experiment',"points.pkl") , 'ab') as f:
                pickle.dump(points,f)

            # Bounding box output
            cls_preds = test_net.blobs['f_cls_preds'].data[...]
            box_preds = test_net.blobs['f_box_preds'].data[...]
            preds_dict = {"box_preds":box_preds, "cls_preds":cls_preds}
            example = {"anchors": np.expand_dims(anchors, 0)}
            example = example_convert_to_torch(example, torch.float32)
            preds_dict = example_convert_to_torch(preds_dict, torch.float32)
            obj_detections += self.predict(example, preds_dict)
            pd_boxes = obj_detections[-1]["box3d_lidar"].cpu().detach().numpy()
            with open(os.path.join('./debug_tool/experiment',"pd_boxes.pkl") , 'ab') as f:
                pickle.dump(pd_boxes,f)

    ############################################################################
    # For object evaluation
    ############################################################################
    def object_detection_evaluation(self, global_step):
        print("[Info] Initialize test net\n")
        test_net = caffe.Net(self.test_net, caffe.TEST)
        test_net.share_with(self.solver.net)
        print("[Info] Loaded train net weights \n")
        data_iter=iter(self.dataloader)
        obj_detections = []
        seg_detections = []
        t = time.time()
        model_dir = str(Path(self.model_dir).resolve())
        model_dir = Path(model_dir)
        result_path = model_dir / 'results'
        result_path_step = result_path / f"step_{global_step}"
        result_path_step.mkdir(parents=True, exist_ok=True)
        for i in tqdm(range(len(data_iter))):
            example = next(data_iter)
            # points = example['seg_points'] # Pointseg
            # voxels = example['voxels']
            # coors = example['coordinates']
            # coors = example['coordinates']
            # num_points = example['num_points']
            # test_net.blobs['top_prev'].reshape(*points.shape)
            # test_net.blobs['top_prev'].data[...] = points
            # test_net.forward()

            # test_net.blobs['top_lat_feats'].reshape(*(voxels.squeeze()).shape)
            # test_net.blobs['top_lat_feats'].data[...] = voxels.squeeze()
            # voxels = voxels.squeeze()
            # with open(os.path.join('./debug',"points.pkl") , 'ab') as f:
            #     pickle.dump(voxels,f)
            # voxels = voxels[cls_out,:]
            # # print("selected voxels", voxels.shape)
            # with open(os.path.join('./debug',"seg_points.pkl") , 'ab') as f:
            #     pickle.dump(voxels,f)
            # NOTE: For voxel seg net
            # seg_points = example['seg_points'] # Pointseg
            # coords = example['coords']
            # coords_center = example['coords_center']
            # p2voxel_idx = example['p2voxel_idx']
            # test_net.blobs['seg_points'].reshape(*seg_points.shape)
            # test_net.blobs['seg_points'].data[...] = seg_points
            # test_net.blobs['coords'].reshape(*coords.shape)
            # test_net.blobs['coords'].data[...] = coords
            # test_net.blobs['p2voxel_idx'].reshape(*p2voxel_idx.shape)
            # test_net.blobs['p2voxel_idx'].data[...] = p2voxel_idx
            ##
            # NOTE: For prior seg
            voxels = example['seg_points']
            test_net.blobs['top_prev'].reshape(*voxels.shape)
            test_net.blobs['top_prev'].data[...] = voxels
            test_net.forward()
            ##
            cls_preds = test_net.blobs['f_cls_preds'].data[...]
            box_preds = test_net.blobs['f_box_preds'].data[...]
            # seg_preds = test_net.blobs['seg_output'].data[...].squeeze()
            # feat_map = test_net.blobs['p2fm'].data[...].squeeze().reshape(5,-1).transpose()
            # feat_map = feat_map[(feat_map != 0).any(-1)]
            # Reverse coordinate for anchor generator
            # anchor generated from generator shape (n_anchors, 7)
            # needed to expand dim for prediction
            # example["anchors"] = np.expand_dims(anchors, 0)
            # preds_dict = {"box_preds":box_preds.reshape(1,-1,7), "cls_preds":cls_preds.reshape(1,-1,1)}
            # example["seg_points"] = voxels
            preds_dict = {"box_preds":box_preds, "cls_preds":cls_preds}
            example = example_convert_to_torch(example, torch.float32)
            preds_dict = example_convert_to_torch(preds_dict, torch.float32)
            obj_detections += self.predict(example, preds_dict)
            # seg_detections += self.seg_predict(np.arange(0.5, 0.75, 0.05), seg_preds, example, result_path_step, vis=False)
            ################ visualization #####################
            pd_boxes = obj_detections[-1]["box3d_lidar"].cpu().detach().numpy()
            with open(os.path.join(result_path_step,"pd_boxes.pkl") , 'ab') as f:
                pickle.dump(pd_boxes,f)

        self.model_logging.log_text(
            f'\nEval at step ---------> {global_step:.2f}:\n', global_step)

        # Object detection evaluation
        result_dict = self.eval_dataset.dataset.evaluation(obj_detections,
                                                str(result_path_step))
        for k, v in result_dict["results"].items():
            self.model_logging.log_text("Evaluation {}".format(k), global_step)
            self.model_logging.log_text(v, global_step)
        self.model_logging.log_metrics(result_dict["detail"], global_step)

        # Class segmentation prediction
        # result_dict = self.total_segmentation_result(seg_detections)
        # for k, v in result_dict["results"].items():
        #     self.model_logging.log_text("Evaluation {}".format(k), global_step)
        #     self.model_logging.log_text(v, global_step)
        # self.model_logging.log_metrics(result_dict["detail"], global_step)

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
        # NOTE: for voxel seg net
        # batch_size = example['coords_center'].shape[0]

        # batch_size = example['seg_points'].shape[0]
        if "metadata" not in example or len(example["metadata"]) == 0:
            meta_list = [None] * batch_size
        else:
            meta_list = example["metadata"]

        batch_anchors = example["anchors"].view(batch_size, -1, example["anchors"].shape[-1])
        # NOTE: for voxel seg net
        # batch_anchors = example["coords_center"].view(batch_size, -1, example["coords_center"].shape[-1])

        # batch_anchors = example["seg_points"].view(batch_size, -1, example["seg_points"].shape[-1])
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
        # NOTE: Original decoding
        batch_box_preds = self._box_coder.decode_torch(batch_box_preds,
                                                       batch_anchors)
        # NOTE: For voxel seg net and point wise prediction
        # batch_box_preds = box_np_ops.fcos_box_decoder_v2_torch(batch_anchors,
        #                                               batch_box_preds)
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
                    print("nms_thres is {} selected {} cars ".format(self._nms_score_thresholds, len(top_scores)))
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
                else:
                    selected = []
                # if selected is not None:
                selected_boxes = box_preds[selected]
                print("IoU_thresh is {} remove {} overlap".format(self._nms_iou_thresholds, (len(box_preds)-len(selected_boxes))))
                #Eval debug
                if "gt_num" in example:
                    eval_idx = example['metadata'][0]['image_idx']
                    eval_obj_num = example['gt_num']
                    detetion_error = eval_obj_num-len(selected_boxes)
                    print("Eval img_{} have {} Object, detected {} Object, error {} ".format(eval_idx, eval_obj_num, len(selected_boxes), detetion_error))

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

    ############################################################################
    # For segmentation evaluation
    ############################################################################
    def segmentation_evaluation(self, global_step):
        print("Initialize test net")
        test_net = caffe.Net(self.test_net, caffe.TEST)
        print("Load train net weights")
        test_net.share_with(self.solver.net)
        _, eval_input_cfg, model_cfg, train_cfg = load_config(self.model_dir, self.config_path)
        voxel_generator, self.target_assigner = build_network(model_cfg)
        ## TODO:
        dataloader, _= load_dataloader(eval_input_cfg, model_cfg,
                                                        voxel_generator,
                                                        self.target_assigner,
                                                        args = self.args)
        data_iter=iter(dataloader)


        model_dir = str(Path(self.model_dir).resolve())
        model_dir = Path(model_dir)
        result_path = model_dir / 'results'
        result_path_step = result_path / f"step_{global_step}"
        result_path_step.mkdir(parents=True, exist_ok=True)

        detections = []
        detections_voc = []
        detections_05 = []
        for i in tqdm(range(len(data_iter))):
            example = next(data_iter)
            points = example['seg_points']
            test_net.blobs['top_prev'].reshape(*points.shape)
            test_net.blobs['top_prev'].data[...] = points
            test_net.forward()

            #seg_cls_pred output shape (1,1,1,16000)
            # seg_cls_pred = test_net.blobs["output"].data[...].squeeze()
            seg_cls_pred = test_net.blobs['seg_output'].data[...].squeeze()
            detections += self.seg_predict(np.arange(0.5, 0.75, 0.05), seg_cls_pred, example, result_path_step, vis=False)
            # detections_voc += self.seg_predict([0.1, 0.3, 0.5, 0.7, 0.9], seg_cls_pred, example, result_path_step, vis=False)
            # detections_05 += self.seg_predict([0.5], seg_cls_pred, example, result_path_step, vis=False)

        result_dict = self.total_segmentation_result(detections)
        # result_dict_voc = self.total_segmentation_result(detections_voc)
        # result_dict_05 = self.total_segmentation_result(detections_05)

        self.model_logging.log_text(
            f'\nEval at step ---------> {global_step:.2f}:\n', global_step)
        for k, v in result_dict["results"].items():
            self.model_logging.log_text("Evaluation {}".format(k), global_step)
            self.model_logging.log_text(v, global_step)
        self.model_logging.log_metrics(result_dict["detail"], global_step)

        # print("\n")
        # for k, v in result_dict_voc["results"].items():
        #     self.model_logging.log_text("Evaluation VOC {}".format(k), global_step)
        #     self.model_logging.log_text(v, global_step)
        # self.model_logging.log_metrics(result_dict_voc["detail"], global_step)
        # print("\n")
        # for k, v in result_dict_05["results"].items():
        #     self.model_logging.log_text("Evaluation 0.5 {}".format(k), global_step)
        #     self.model_logging.log_text(v, global_step)
        # self.model_logging.log_metrics(result_dict_05["detail"], global_step)


    def seg_predict(self, thresh_range, pred, example, result_path_step, vis=False):
        # pred = 1 / (1 + np.exp(-pred)) #sigmoid
        gt = example['seg_labels']
        ############### Params ###############
        eps = 1e-5

        cls_thresh_range = thresh_range
        pos_class = 1 # Car
        list_score = []
        cls_thresh_list = []
        ############### Params ###############

        pred, gt = np.array(pred), np.array(gt)
        gt = np.squeeze(gt)
        labels = np.unique(gt)
        ##################Traverse cls_thresh###################################
        for cls_thresh in cls_thresh_range:
            scores = {}
            _pred = np.where(pred>cls_thresh, 1, 0)

            TPs = np.sum((gt == pos_class) * (_pred == pos_class))
            TNs = np.sum((gt != pos_class) * (_pred != pos_class))
            FPs = np.sum((gt != pos_class) * (_pred == pos_class))
            FNs = np.sum((gt == pos_class) * (_pred != pos_class))
            TargetTotal= np.sum(gt == pos_class)

            scores['accuracy'] = TPs / (TargetTotal + eps)
            scores['class_iou'] = TPs / ((TPs + FNs + FPs) + eps)
            scores['precision'] = TPs / ((TPs + FPs) + eps)

            cls_thresh_list.append(scores)

        ###################Found best cls_thresh################################
        thresh_accuracy=[]
        thresh_class_iou=[]
        thresh_precision=[]
        max_class_iou = 0
        max_class_iou_thresh = 0

        for thresh, cls_list in zip(cls_thresh_range, cls_thresh_list):
            accuracy = cls_list['accuracy']
            class_iou = cls_list['class_iou']
            precision = cls_list['precision']
            thresh_accuracy.append(accuracy)
            thresh_class_iou.append(class_iou)
            thresh_precision.append(precision)

            if class_iou > max_class_iou:
                max_class_iou = class_iou
                max_class_iou_thresh = thresh

        scores['accuracy'] = np.mean(np.array(thresh_accuracy))
        scores['class_iou'] = np.mean(np.array(thresh_class_iou))
        scores['precision'] = np.mean(np.array(thresh_precision))
        scores['best_thresh'] = max_class_iou_thresh #choose the max_thresh for seg

        ############################pred_thresh#################################
        pred_thresh = self._nms_score_thresholds[0]

        points = example['seg_points']
        points = np.squeeze(points)
        pd_points = points[pred >= pred_thresh]

        with open(os.path.join(result_path_step, "gt_points.pkl"), 'ab') as f:
            pickle.dump(pd_points,f)

        if vis:
            image_idx = example['image_idx']
            gt_boxes = example['gt_boxes']
            with open(os.path.join(result_path_step, "image_idx.pkl"), 'ab') as f:
                pickle.dump(image_idx,f)
            with open(os.path.join(result_path_step, "points.pkl"), 'ab') as f:
                pickle.dump(points,f)
            with open(os.path.join(result_path_step, "gt_boxes.pkl"), 'ab') as f:
                pickle.dump(gt_boxes,f)

        list_score.append(scores)

        return list_score

    def total_segmentation_result(self, detections):
        avg_accuracy=[]
        avg_class_iou=[]
        avg_precision=[]
        avg_thresh=[]
        for det in detections:
            avg_accuracy.append(det['accuracy'])
            avg_class_iou.append(det['class_iou'])
            avg_precision.append(det['precision'])
            avg_thresh.append(det['best_thresh'])

        avg_accuracy = np.sum(np.array(avg_accuracy)) / np.sum((np.array(avg_accuracy)!=0)) #divided by none zero no Cars
        avg_class_iou = np.sum(np.array(avg_class_iou)) / np.sum((np.array(avg_class_iou)!=0))  #divided by none zero no Cars
        avg_precision = np.sum(np.array(avg_precision)) / np.sum((np.array(avg_precision)!=0))  #divided by none zero no Cars
        avg_thresh = np.sum(np.array(avg_thresh)) / np.sum((np.array(avg_thresh)!=0))  #divided by none zero no Cars

        print('-------------------- Summary --------------------')

        result_dict = {}
        result_dict['results'] ={"Summary" : 'Threshhold: {:.3f} \n'.format(avg_thresh) + \
                                             'Accuracy: {:.3f} \n'.format(avg_accuracy) + \
                                             'Car IoU: {:.3f} \n'.format(avg_class_iou) + \
                                             'Precision: {:.3f} \n'.format(avg_precision)
                                             }

        result_dict['detail'] = {"Threshold" : avg_thresh,
                                 "Accuracy" : avg_accuracy,
                                 "Car IoU": avg_class_iou,
                                 "Precision": avg_precision,
                                 }
        return result_dict

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

def load_dataloader(eval_input_cfg, model_cfg, voxel_generator, target_assigner, args):
    try: segmentation = args["segmentation"]
    except: segmentation = True
    try: bcl_keep_voxels = args["bcl_keep_voxels_eval"]
    except: bcl_keep_voxels = 6000
    try: seg_keep_points = args["seg_keep_points_eval"]
    except: seg_keep_points = 8000
    try: points_per_voxel = args["points_per_voxel"]
    except: points_per_voxel = 200
    eval_dataset = input_reader_builder.build(
        eval_input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner,
        segmentation=segmentation,
        bcl_keep_voxels=bcl_keep_voxels,
        seg_keep_points=seg_keep_points,
        generate_anchors_cachae=args['anchors_cachae'],
        points_per_voxel=points_per_voxel
        ) #True FOR Pillar, False For BCL

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_input_cfg.batch_size, #eval_input_cfg.batch_size, # only support multi-gpu train
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
        elif k in ["box_preds", "cls_preds", "coords_center"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.float32, device=device).to(dtype)
        else:
            example_torch[k] = v
    return example_torch
