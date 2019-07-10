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

def segmentation(n, seg_points, label, cls_labels, reg_targets, dataset_params, phase):
    ############### Params ###############
    num_cls = dataset_params['num_cls']
    box_code_size = dataset_params['box_code_size']
    num_anchor_per_loc = dataset_params['num_anchor_per_loc']

    num_filters = dataset_params['num_filters']
    layer_strides = dataset_params['layer_strides']
    layer_nums = dataset_params['layer_nums']

    num_upsample_filters = dataset_params['num_upsample_filters']
    upsample_strides = dataset_params["upsample_strides"]

    feat_map_size = dataset_params['feat_map_size'] #(b,c,n,h,w)
    point_cloud_range = dataset_params['point_cloud_range']
    seg_thresh = dataset_params['seg_thresh']

    use_depth = dataset_params['use_depth']
    use_score = dataset_params['use_score']
    use_points = dataset_params['use_points']
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
    top_prev = bcl_bn_relu(n, 'bcl_seg', top_prev, top_lattice, nout=[64, 128, 128, 64],
                          lattic_scale=["0*8_1*8_2*8", "0*4_1*4_2*4", "0*2_1*2_2*2", "0_1_2"], loop=4, skip='concat')

    # BEST NOW
    # top_prev = bcl_bn_relu(n, 'bcl_seg', top_prev, top_lattice, nout=[64, 128, 128, 128, 64],
                          # lattic_scale=["0*2_1*2_2*2","0_1_2","0/2_1/2_2/2","0/4_1/4_2/4","0/8_1/8_2/8"], loop=5, skip='concat')

    # top_prev = conv_bn_relu(n, "conv0_seg", top_prev, 1, 256, stride=1, pad=0, loop=1)
    # top_prev = conv_bn_relu(n, "conv0_seg", top_prev, 1, 128, stride=1, pad=0, loop=1)
    top_prev = conv_bn_relu(n, "conv1_seg", top_prev, 1, 64, stride=1, pad=0, loop=1)

    n.seg_preds = L.Convolution(top_prev, name = "seg_head",
                         convolution_param=dict(num_output=num_cls,
                                                kernel_size=1, stride=1, pad=0,
                                                weight_filler=dict(type = 'xavier'),
                                                bias_term = True,
                                                bias_filler=dict(type='constant', value=0),
                                                engine=1,
                                                ),
                         param=[dict(lr_mult=1), dict(lr_mult=0.1)])
    # Predict class
    if phase == "train":
        seg_preds = L.Permute(n.seg_preds, permute_param=dict(order=[0, 2, 3, 1])) #(B,C=1,H,W) -> (B,H,W,C=1)
        seg_preds = L.Reshape(seg_preds, reshape_param=dict(shape=dict(dim=[0, -1, num_cls])))# (B,H,W,C=1)-> (B, -1, 1)

        # seg_weights = L.Python(label, name = "SegWeight",
        #                        python_param=dict(
        #                                         module='bcl_layers',
        #                                         layer='SegWeight'
        #                                         ))
        #
        # seg_weights = L.Reshape(seg_weights, reshape_param=dict(shape=dict(dim=[0, -1])))

        n.seg_loss = L.Python(seg_preds, label, #seg_weights,
                         name = "Seg_Loss",
                         loss_weight = 1,
                         python_param=dict(
                         module='bcl_layers',
                         layer='FocalLoss'  #WeightFocalLoss, DiceFocalLoss, FocalLoss, DiceLoss
                         ),
                param_str=str(dict(focusing_parameter=2, alpha=0.25)))

        # n.accuracy = L.Accuracy(n.seg_preds, label)
    top_prev = conv_bn_relu(n, "P2FM_Decov", top_prev, 1, 32, stride=1, pad=0, loop=1)
    n.seg_output = L.Sigmoid(n.seg_preds)
    n.p2fm = L.Python(seg_points, n.seg_output, top_prev,
                     name = "Point2FeatMap",
                     python_param=dict(
                     module='bcl_layers',
                     layer='Point2FeatMap'
                     ),
                param_str=str(dict(thresh=seg_thresh,
                                   feat_map_size=feat_map_size, #(B,C,N,H,W)
                                   point_cloud_range=point_cloud_range,
                                   use_depth=use_depth,
                                   use_score=use_score,
                                   use_points=use_points)))

    top_prev = n.p2fm

    # top_prev = L.Reshape(top_prev, reshape_param=dict(shape=dict(dim=[0, -1, feat_map_size[3], feat_map_size[4]]))) # (B,H,W,C=1)-> (B, -1, 1)


    top_prev = conv_bn_relu(n, "ini_conv1", top_prev, 3, num_filters[0], stride=layer_strides[0], pad=1, loop=1)
    top_prev = conv_bn_relu(n, "rpn_conv1", top_prev, 3, num_filters[0], stride=1, pad=1, loop=layer_nums[0]) #3
    deconv1 = deconv_bn_relu(n, "rpn_deconv1", top_prev, upsample_strides[0], num_upsample_filters[0], stride=upsample_strides[0], pad=0)

    top_prev = conv_bn_relu(n, "ini_conv2", top_prev, 3, num_filters[1], stride=layer_strides[1], pad=1, loop=1)
    top_prev = conv_bn_relu(n, "rpn_conv2", top_prev, 3, num_filters[1], stride=1, pad=1, loop=layer_nums[1]) #5
    deconv2 = deconv_bn_relu(n, "rpn_deconv2", top_prev, upsample_strides[1], num_upsample_filters[1], stride=upsample_strides[1], pad=0)

    top_prev = conv_bn_relu(n, "ini_conv3", top_prev, 3, num_filters[2], stride=layer_strides[2], pad=1, loop=1)
    top_prev = conv_bn_relu(n, "rpn_conv3", top_prev, 3, num_filters[2], stride=1, pad=1, loop=layer_nums[2]) #5
    deconv3 = deconv_bn_relu(n, "rpn_deconv3", top_prev, upsample_strides[2], num_upsample_filters[2], stride=upsample_strides[2], pad=0)
    n['rpn_out'] = L.Concat(deconv1, deconv2, deconv3)
    top_prev = n['rpn_out']

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

    cls_preds = L.Permute(n.cls_preds, permute_param=dict(order=[0, 2, 3, 1])) #(B,C,H,W) -> (B,H,W,C)
    cls_preds = L.Reshape(cls_preds, reshape_param=dict(shape=dict(dim=[0, -1, 1])))# (B,H,W,C) -> (B, -1, C)

    box_preds = L.Permute(n.box_preds, permute_param=dict(order=[0, 2, 3, 1])) #(B,C,H,W) -> (B,H,W,C)
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
        n['labels_input'] = L.Python(cls_labels, cared,
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
                                   ntop= 4, python_param=dict(module='bcl_layers', layer='InputKittiDataV5',
                                                     param_str=repr(dataset_params_train)))
        seg_points, seg_labels, cls_labels, reg_targets = datalayer_train

    elif phase == "eval":
        dataset_params_eval = dataset_params.copy()
        n['top_prev'] = L.Python(
                                name = 'top_pre_input',
                                ntop=1,
                                include=dict(phase=caffe.TEST),
                                python_param=dict(
                                module='bcl_layers',
                                layer='DataFeature',
                                param_str=repr(dataset_params_eval)
                                ))
        top_prev = n['top_prev']
        seg_points = top_prev
        seg_labels = None
        cls_labels = None
        reg_targets = None

    n = segmentation(n, seg_points, seg_labels, cls_labels, reg_targets, dataset_params, phase)

    # if phase == "eval":
    #     car_points = output

    print(n)
    return n.to_proto()
