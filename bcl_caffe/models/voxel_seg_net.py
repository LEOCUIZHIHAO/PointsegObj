import caffe
from caffe import layers as L, params as P, to_proto

def conv_bn_relu(n, name, top_prev, ks, nout, stride=1, pad=0, loop=1):

    for idx in range(loop):
        n[str(name)+"_"+str(idx)] = L.Convolution(top_prev, #name = name,
                                            convolution_param=dict(
                                                    kernel_size=ks, stride=stride,
                                                    num_output=nout, pad=pad,
                                                    engine=2,
                                                    weight_filler=dict(type = 'xavier'),
                                                    bias_term = False),
                                                    param=[dict(lr_mult=1)],
                                                    )
        top_prev = n[str(name)+"_"+str(idx)]
        n[str(name)+'_bn_'+str(idx)] = L.BatchNorm(top_prev, batch_norm_param=dict(eps=1e-3, moving_average_fraction=0.99))
        top_prev = n[str(name)+'_bn_'+str(idx)]
        n[str(name)+'_sc_'+str(idx)] = L.Scale(top_prev, scale_param=dict(bias_term=True))
        top_prev = n[str(name)+'_sc_'+str(idx)]
        n[str(name)+'_relu_'+str(idx)] = L.ReLU(top_prev, in_place=True)
        top_prev = n[str(name)+'_relu_'+str(idx)]

    return top_prev

def deconv_bn_relu(n, name, top_prev, ks, nout, stride=1, pad=0):
    n[str(name)] = L.Deconvolution(top_prev, # name = name,
                                            convolution_param=dict(kernel_size=ks, stride=stride,
                                                num_output=nout, pad=pad,
                                                engine=2,
                                                weight_filler=dict(type = 'xavier'),
                                                bias_term = False),
                                                param=[dict(lr_mult=1)],
                                                )
    top_prev = n[str(name)]
    n[str(name)+'_bn'] = L.BatchNorm(top_prev, batch_norm_param=dict(eps=1e-3, moving_average_fraction=0.99))
    top_prev = n[str(name)+'_bn']
    n[str(name)+'_sc'] = L.Scale(top_prev, scale_param=dict(bias_term=True))
    top_prev = n[str(name)+'_sc']
    n[str(name)+'_relu'] = L.ReLU(top_prev, in_place=True)
    top_prev = n[str(name)+'_relu']

    return top_prev

def bcl_bn_relu(n, name, top_prev, top_lat_feats, nout, lattic_scale=None, loop=1, skip=None):

    if skip=='concat':
        skip_params= []

    for idx in range(loop):

        if lattic_scale:

            n[str(name)+"_scale_"+str(idx)] = L.Python(top_lat_feats, python_param=dict(module='bcl_layers',
                                                                    layer='PickAndScale',
                                                                    param_str=lattic_scale[idx]))
            _top_lat_feats = n[str(name)+"_scale_"+str(idx)]


        bltr_weight_filler = dict(type='gaussian', std=float(0.001))
        # bltr_weight_filler = dict(type = 'xavier')
        n[str(name)+"_"+str(idx)] = L.Permutohedral(top_prev, _top_lat_feats, _top_lat_feats,
                                                        ntop=1,
                                                        permutohedral_param=dict(
                                                            num_output=nout[idx],
                                                            group=1,
                                                            neighborhood_size=1,
                                                            bias_term=True,
                                                            norm_type=P.Permutohedral.AFTER,
                                                            offset_type=P.Permutohedral.NONE,
                                                            filter_filler=bltr_weight_filler,
                                                            bias_filler=dict(type='constant',
                                                                             value=0)),
                                                        param=[{'lr_mult': 1, 'decay_mult': 1},
                                                               {'lr_mult': 2, 'decay_mult': 0}])

        top_prev = n[str(name)+"_"+str(idx)]
        n[str(name)+'_bn_'+str(idx)] = L.BatchNorm(top_prev, batch_norm_param=dict(eps=1e-3, moving_average_fraction=0.99))
        top_prev = n[str(name)+'_bn_'+str(idx)]
        n[str(name)+'_sc_'+str(idx)] = L.Scale(top_prev, scale_param=dict(bias_term=True))
        top_prev = n[str(name)+'_sc_'+str(idx)]
        n[str(name)+'_relu_'+str(idx)] = L.ReLU(top_prev, in_place=True)
        top_prev = n[str(name)+'_relu_'+str(idx)]

        if skip=='concat':
            skip_params.append(n[str(name)+'_relu_'+str(idx)])

    if skip=='concat':
        n['concat'] = L.Concat(*skip_params)
        top_prev = n['concat']

    return top_prev

def segmentation(n, seg_points, label, coords, p2voxel_idx, cls_labels,
                                        reg_targets ,dataset_params, phase):
    ############### Params ###############
    num_cls = dataset_params['num_cls']
    box_code_size = dataset_params['box_code_size']
    num_anchor_per_loc = dataset_params['num_anchor_per_loc']

    max_voxels = dataset_params['max_voxels']
    points_per_voxel = dataset_params['points_per_voxel']
    ############### Params ###############

    top_prev, top_lattice= L.Python(seg_points, ntop=2, python_param=dict(module='bcl_layers',layer='BCLReshape'))

    top_prev = conv_bn_relu(n, "conv0_seg", top_prev, 1, 64, stride=1, pad=0, loop=1)

    """
    1. If lattice scale too large the network will really slow and don't have good result
    """
    # #2nd
    # top_prev = bcl_bn_relu(n, 'bcl_seg', top_prev, top_lattice, nout=[64, 64, 128, 128, 128, 64],
    #                       lattic_scale=["0*4_1*4_2*4","0*2_1*2_2*2","0_1_2","0/2_1/2_2/2","0/4_1/4_2/4","0/8_1/8_2/8"], loop=6, skip='concat')
    #
    # #3rd
    top_prev = bcl_bn_relu(n, 'bcl_seg', top_prev, top_lattice, nout=[64, 128, 64],
                          lattic_scale=["0*4_1*4_2*4", "0*2_1*2_2*2", "0_1_2"], loop=3, skip=None)

    # BEST NOW
    # top_prev = bcl_bn_relu(n, 'bcl_seg', top_prev, top_lattice, nout=[64, 128, 128, 128, 64],
                          # lattic_scale=["0*2_1*2_2*2","0_1_2","0/2_1/2_2/2","0/4_1/4_2/4","0/8_1/8_2/8"], loop=5, skip='concat')

    # top_prev = conv_bn_relu(n, "conv0_seg", top_prev, 1, 256, stride=1, pad=0, loop=1)
    # top_prev = conv_bn_relu(n, "conv0_seg", top_prev, 1, 128, stride=1, pad=0, loop=1)
    top_prev = conv_bn_relu(n, "conv1_seg", top_prev, 1, 64, stride=1, pad=0, loop=1)

    # n.seg_preds = L.Convolution(top_prev, name = "seg_head",
    #                      convolution_param=dict(num_output=num_cls,
    #                                             kernel_size=1, stride=1, pad=0,
    #                                             weight_filler=dict(type = 'xavier'),
    #                                             bias_term = True,
    #                                             bias_filler=dict(type='constant', value=0),
    #                                             engine=1,
    #                                             ),
    #                      param=[dict(lr_mult=1), dict(lr_mult=0.1)])
    # Predict class
    # if phase == "train":
    #     seg_preds = L.Permute(n.seg_preds, permute_param=dict(order=[0, 2, 3, 1])) #(B,C=1,H,W) -> (B,H,W,C=1)
    #     seg_preds = L.Reshape(seg_preds, reshape_param=dict(shape=dict(dim=[0, -1, num_cls])))# (B,H,W,C=1)-> (B, -1, 1)
    #
    #     seg_weights = L.Python(label, name = "SegWeight",
    #                            python_param=dict(
    #                                             module='bcl_layers',
    #                                             layer='SegWeight'
    #                                             ))
    #
    #     seg_weights = L.Reshape(seg_weights, reshape_param=dict(shape=dict(dim=[0, -1])))
    #
    #     n.seg_loss = L.Python(seg_preds, label, seg_weights,
    #                      name = "Seg_Loss",
    #                      loss_weight = 1,
    #                      python_param=dict(
    #                      module='bcl_layers',
    #                      layer='FocalLoss'  #WeightFocalLoss, DiceFocalLoss, FocalLoss, DiceLoss
    #                      ),
    #             param_str=str(dict(focusing_parameter=2, alpha=0.25)))

    top_prev = conv_bn_relu(n, "P2VX_Decov", top_prev, 1, 32, stride=1, pad=0, loop=1)
    # n.seg_output = L.Sigmoid(n.seg_preds)
    n.p2vx = L.Python(top_prev, p2voxel_idx, # seg_pred only for rubbish dump
                             name = "Point2Voxel3D",
                             ntop = 1,
                             python_param=dict(
                             module='bcl_layers',
                             layer='Point2Voxel3D'
                             ),
                             param_str=str(dict(max_voxels=max_voxels,
                                                points_per_voxel=points_per_voxel)))

    top_prev = n.p2vx

    top_lattice = L.Permute(coords, name="coords_permute", permute_param=dict(order=[0, 2, 1])) #(B,C=1,H,W) -> (B,H,W,C=1)
    top_lattice = L.Reshape(top_lattice, name="coords_reshape", reshape_param=dict(shape=dict(dim=[0, -1, 1, max_voxels])))# (B,H,W,C=1)-> (B, -1, 1)

    top_prev = conv_bn_relu(n, "conv2_seg_voxel", top_prev, 1, 64, stride=1, pad=0, loop=1)
    top_prev = bcl_bn_relu(n, 'bcl_seg_voxel', top_prev, top_lattice, nout=[64, 128, 128, 64],
                          lattic_scale=["0*8_1*8_2*8", "0*4_1*4_2*4", "0*2_1*2_2*2", "0_1_2"], loop=4, skip='concat')
    top_prev = conv_bn_relu(n, "conv3_seg_voxle", top_prev, 1, 64, stride=1, pad=0, loop=1)

    n.cls_preds = L.Convolution(top_prev, name = "cls_head",
                         convolution_param=dict(num_output=num_anchor_per_loc * num_cls,
                                                kernel_size=1, stride=1, pad=0,
                                                weight_filler=dict(type = 'xavier'),
                                                bias_term = True,
                                                bias_filler=dict(type='constant', value=0),
                                                engine=1,
                                                ),
                         param=[dict(lr_mult=1), dict(lr_mult=1)])

    n.box_preds = L.Convolution(top_prev, name = "reg_head",
                          convolution_param=dict(num_output=num_anchor_per_loc * box_code_size,
                                                 kernel_size=1, stride=1, pad=0,
                                                 weight_filler=dict(type = 'xavier'),
                                                 bias_term = True,
                                                 bias_filler=dict(type='constant', value=0),
                                                 engine=1,
                                                 ),
                          param=[dict(lr_mult=1), dict(lr_mult=1)])

    cls_preds = n.cls_preds
    box_preds = n.box_preds
    box_preds = L.ReLU(box_preds, in_place=True)

    cls_preds = L.Permute(cls_preds, permute_param=dict(order=[0, 2, 3, 1])) #(B,C,H,W) -> (B,H,W,C)
    cls_preds = L.Reshape(cls_preds, reshape_param=dict(shape=dict(dim=[0, -1, 1])))# (B,H,W,C) -> (B, -1, C)

    box_preds = L.Permute(box_preds, permute_param=dict(order=[0, 2, 3, 1])) #(B,C,H,W) -> (B,H,W,C)
    box_preds = L.Reshape(box_preds, reshape_param=dict(shape=dict(dim=[0, -1, box_code_size]))) #(B,H,W,C) -> (B, -1, C)


    if phase == "train":

        n['cared'],n['reg_outside_weights'], n['cls_weights']= L.Python(cls_labels,
                                                                        name = "PrepareLossWeight",
                                                                        ntop = 3,
                                                                        python_param=dict(
                                                                                    module='bcl_layers',
                                                                                    layer='PrepareLossWeight'
                                                                                    ))
        reg_outside_weights, cared, cls_weights = n['reg_outside_weights'], n['cared'], n['cls_weights']

        # Gradients cannot be computed with respect to the label inputs (bottom[1])#
        n['labels_input'] = L.Python(cls_labels, cared, label,
                            name = "Label_Encode",
                            python_param=dict(
                                        module='bcl_layers',
                                        layer='LabelEncode',
                                        ))
        labels_input = n['labels_input']


        n.cls_loss= L.Python(cls_preds, labels_input, cls_weights,
                                name = "FocalLoss",
                                loss_weight = 1,
                                python_param=dict(
                                            module='bcl_layers',
                                            layer='WeightFocalLoss'
                                            ),
                                param_str=str(dict(focusing_parameter=2, alpha=0.25)))

        n.reg_loss= L.Python(box_preds, reg_targets, reg_outside_weights,
                                name = "WeightedSmoothL1Loss",
                                loss_weight = 1,
                                python_param=dict(
                                            module='bcl_layers',
                                            layer='WeightedSmoothL1Loss'
                                            ))

    # Problem
    if phase == "eval":
        n.f_cls_preds = cls_preds
        n.f_box_preds = box_preds

    return n

def seg_object_detection(phase,
            dataset_params=None,
            cfg = None,
            deploy=False,
            create_prototxt=True,
            save_path=None,
            ):

    n = caffe.NetSpec()

    if phase == "train":

        dataset_params_train = dataset_params.copy()
        dataset_params_train['subset'] = phase
        datalayer_train = L.Python(name='data', include=dict(phase=caffe.TRAIN),
                                   ntop= 6, python_param=dict(module='bcl_layers', layer='InputKittiDataV7',
                                                     param_str=repr(dataset_params_train)))
        seg_points, seg_labels, coords, p2voxel_idx, labels, reg_targets = datalayer_train

    elif phase == "eval":
        dataset_params_eval = dataset_params.copy()
        n['seg_points'], n['coords'], n['p2voxel_idx'] = L.Python(
                                name = 'top_pre_input',
                                ntop=3,
                                include=dict(phase=caffe.TEST),
                                python_param=dict(
                                module='bcl_layers',
                                layer='VoxelSegNetInput',
                                param_str=repr(dataset_params_eval)
                                ))
        seg_points = n['seg_points']
        seg_labels = None
        p2voxel_idx = n['p2voxel_idx']
        labels = None
        reg_targets = None
        coords = n['coords']

    n = segmentation(n, seg_points, seg_labels, coords, p2voxel_idx, labels,
                                            reg_targets ,dataset_params, phase)

    # if phase == "eval":
    #     car_points = output

    print(n)
    return n.to_proto()
