import os
import re

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import tensorrt as trt
import tf2onnx
from tensorflow import __version__ as TF_VERSION


def maskrcnn_to_onnx(model, model_name, input_spec, kwargs):
    if model.name != 'mask_rcnn_inference':
        raise ValueError('Inference model should be send to maskrcnn_to_onnx function.')

    output_path = f'../weights/{model_name}.onnx'
    _, _ = tf2onnx.convert.from_keras(model=model,
                                      input_signature=input_spec,
                                      output_path=output_path,
                                      **kwargs)
    print(f'Successfully converted from tensorflow to .onnx: {output_path}')


def make_engine_from_onnx(save_path, model_name, fp16_mode, max_batch_size=1):
    """
    Making TensorRT from ONNX.
    Args:
        save_path: str
        model_name: str

    Returns: None

    """

    WSPACE_SIZE = 1024
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    model_path = os.path.join(save_path, f"{model_name}.onnx")
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(explicit_batch) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:

        with open(model_path, 'rb') as model:
            parser.parse(model.read())

        print('Num of detected layers: ', network.num_layers)
        builder.max_batch_size = max_batch_size
        # Size in Mb. 1e6 bytes == 1Mb
        builder.max_workspace_size = int(1e6 * WSPACE_SIZE)
        builder.fp16_mode = fp16_mode
        engine = builder.build_cuda_engine(network)

        if fp16_mode:
            file_prefix = 'trt_fp16_'
        else:
            file_prefix = 'trt_fp32_'

        with open(os.path.join(save_path, file_prefix + f'{model_name}.engine'), "wb") as f:
            f.write(engine.serialize())

    print(f"""[ONNX parser: DONE] TensorRT engine: {os.path.join(save_path,
                                                                 file_prefix + f'{model_name}.engine')}""")


def add_trt_resize_nearest(graph, config, verbose=False):
    """
    Add ResizeNearest_TRT layer
    https://github.com/NVIDIA/TensorRT/tree/master/plugin/resizeNearestPlugin
    Args:
        graph: .onnx-graph
        config: Mask-RCNN config
        verbose: bool

    Returns: modified .onnx-graph

    """
    attrs = {'scale': 2.0}

    if verbose:
        print(f'\nResizeNearest_TRT attributes: {attrs}')

    # Get a dict with Resize__ layers
    upsampling_nodes_dict = {int(x.replace('Resize__', '').replace(':0', '')): x
                             for x in graph.tensors().keys() if re.match(r'Resize__[0-9]+:0', x)}
    upsampling_node_ids = sorted(upsampling_nodes_dict.keys())

    if float(TF_VERSION[:-2]) < 2.4:
        add_list = ['mask_rcnn_inference/tf_op_layer_AddV2/AddV2',
                    'mask_rcnn_inference/tf_op_layer_AddV2_1/AddV2_1',
                    'mask_rcnn_inference/tf_op_layer_AddV2_2/AddV2_2'
                    ]
    else:
        add_list = ['mask_rcnn_inference/tf.__operators__.add/AddV2',
                    'mask_rcnn_inference/tf.__operators__.add_1/AddV2',
                    'mask_rcnn_inference/tf.__operators__.add_2/AddV2'
                    ]

    for upsample_node_id, add_node_name in zip(upsampling_node_ids, add_list):
        upsample_node_name = upsampling_nodes_dict[upsample_node_id]
        print(upsample_node_name)
        node = graph.tensors()[upsample_node_name]
        node_inputs = [node.inputs[0].inputs[0]]

        node.inputs.clear()
        node.outputs.clear()

        upsample_node = gs.Node(name=upsample_node_name,
                                op="ResizeNearest_TRT",
                                attrs=attrs,
                                inputs=node_inputs,
                                outputs=[gs.Variable(name=upsample_node_name, dtype=None, shape=None)]
                                )
        graph.nodes.append(upsample_node)

        _idx = [idx for idx, x in enumerate(graph.nodes) if x.name in add_node_name][0]
        graph.nodes[_idx].inputs.append(upsample_node.outputs[0])

    return graph


@gs.Graph.register()
def add_trt_proposal_layer(self, rpn_class, rpn_bbox, attrs, verbose=False):
    """
    Add ProposalLayer_TRT
    https://github.com/NVIDIA/TensorRT/tree/master/plugin/proposalLayerPlugin
    Args:
        self: onnx graph instance
        rpn_class: [N, anchors, 2, 1]:
                    N       - the batch_size,
                    anchors - total number of anchors
                    2       - 2 classes of objectness - foreground and background.
        rpn_bbox: [N, anchors, 4, 1]
                  4 refers to the 4 elements of refinement information - [dy, dx, dh, dw].
        attrs:   layers parameters, see full doc. Example:
                 attrs = {'prenms_topk': 1024,
                          'keep_topk': 1000,
                          'iou_threshold': 0.7,
                          'image_size': (3, 512, 512)
                          }
        verbose: bool

    Returns: roi[N, keep_topk, 4]
             keep_topk - the maximum number of detections left after NMS
             4         - coordinates of ROI candidates [y1, x1, y2, x2]

    """

    if verbose:
        print(f'\nProposalLayer_TRT attributes: {attrs}')

    rpn_class.outputs.clear()
    rpn_bbox.outputs.clear()

    return self.layer(op="ProposalLayer_TRT", attrs=attrs,
                      name='proposal_layer_trt',
                      inputs=[rpn_class, rpn_bbox],
                      outputs=[gs.Variable(name='roi',
                                           dtype=np.float32,
                                           shape=(1, attrs['keep_topk'], 4)
                                           )
                               ]
                      )


@gs.Graph.register()
def add_trt_pyramid_roialign(self, inputs_list, pool_size, name, clear_outputs=False, verbose=False):
    """
    Add PyramidROIAlign_TRT
    https://github.com/NVIDIA/TensorRT/tree/master/plugin/pyramidROIAlignPlugin

    Args:
        self: onnx graph instance
        inputs_list: number of input variables, in list:\
                     roi: [N, rois, 4]
                     4x fpn outputs: fpn_p2, fpn_p3, fpn_p4, fpn_p5
        pool_size: pool_size, for Mask-RCNN heads
        name: output name
        clear_outputs: clear outputs or not
        verbose: bool

    Returns: roi_align[N, rois, C, pooled_size, pooled_size]

    """
    attrs = {'pooled_size': pool_size}

    if verbose:
        print(f'\nPyramidROIAlign_TRT attributes: {attrs}')

    if clear_outputs:
        for inp in inputs_list:
            inp.outputs.clear()

    return self.layer(op="PyramidROIAlign_TRT", attrs=attrs,
                      name=name + '_trt',
                      inputs=inputs_list,
                      outputs=[gs.Variable(name=name, dtype=None, shape=None)])


@gs.Graph.register()
def add_trt_detection_layer(self, inputs_list, attrs, verbose=False):
    """
    https://github.com/NVIDIA/TensorRT/tree/master/plugin/detectionLayerPlugin
    Add DetectionLayer_TRT
    Args:
        self: onnx graph instance
        inputs_list: delta_bbox: [N, rois, num_classes*4, 1, 1]
                    score: [N, rois, num_classes, 1, 1]
                    roi:   [N, rois, 4]
        attrs: layer attributes. Example:
            attrs = {'num_classes': 2,
                     'keep_topk': 100,
                     'score_threshold': 0.7,
                     'iou_threshold': 0.3,
                     }
        verbose: bool

    Returns:  mrcnn_detection[N, keep_topk, 6] where:
              N - batch size
              keep_topk - maximum number of detections left after NMS
              '6' - 6 elements of an detection [y1, x1, y2, x2, class_label, score]
    """

    if verbose:
        print(f'\nDetectionLayer_TRT attributes: {attrs}')

    # Do not clear outputs in ProposalLayer_TRT_ROI
    for inp in inputs_list[:2]:
        inp.outputs.clear()

    return self.layer(op="DetectionLayer_TRT", attrs=attrs,
                      name='detection_layer_trt',
                      inputs=inputs_list,
                      outputs=[gs.Variable(name='mrcnn_detection', dtype=np.float32,
                                           shape=(1, attrs['keep_topk'], 6))])


@gs.Graph.register()
def add_trt_special_slice_layer(self, inputs_list, name, verbose=False):
    """
    https://github.com/NVIDIA/TensorRT/tree/master/plugin/specialSlicePlugin
    The SpecialSlice plugin slice the detections of MaskRCNN
    from [y1, x1, y2, x2, class_label, score] to [y1, x1, y2, x2]
    Args:
        self:
        inputs_list:
        name: output tensor name

    Returns: tensor of shape [N, num_det, 4]

    """
    if verbose:
        print('\nAdded SpecialSlice_TRT')
    for inp in inputs_list[:2]:
        inp.outputs.clear()

    return self.layer(op="SpecialSlice_TRT",
                      name='detection_boxes_extraction_trt',
                      inputs=inputs_list,
                      outputs=[gs.Variable(name=name, dtype=None, shape=None)])


@gs.Graph.register()
def add_trt_lrelu_layer(self, inputs_list, attrs, name, verbose=False):
    """
    https://github.com/NVIDIA/TensorRT/tree/master/plugin/leakyReluPlugin
    TensorRT wrapper for Leaky ReLU (PReLU)
    Args:
        self:
        inputs_list:
        attrs:   {'negSlope': 0.3}, 0.3 - tensorflow default
        node_name: original node name
        verbose:

    Returns:

    """
    if verbose:
        print('\nAdded LReLU_TRT')
    for inp in inputs_list[:2]:
        inp.outputs.clear()

    return self.layer(op="LReLU_TRT",
                      name=name + '_TRT',
                      attrs=attrs,
                      inputs=inputs_list,
                      outputs=[gs.Variable(name=f"{name}:0", dtype=None, shape=None)])


@gs.Graph.register()
def onnx_reshape_layer(self, input_tensor, reshape, name, info_shape):
    """
    Make reshape layer in onnx model.
    Args:
        self: onnx graph instance
        input_tensor: tensor in onnx graph
        reshape:      reshape value, numpy.array
        name:         reshape node name
        info_shape:   info about shape, list or tuple

    Returns: reshape layer

    """
    input_tensor.outputs.clear()

    return self.layer(op="Reshape",
                      name=name + '_node',
                      inputs=[input_tensor,
                              gs.Constant(name=f'const_{name}', values=reshape)],
                      outputs=[gs.Variable(name=name, dtype=np.float32, shape=info_shape)])


@gs.Graph.register()
def onnx_transpose_layer(self, input_tensor, perm, name):
    """

    Args:
        self: onnx graph instance
        input_tensor: input tensor to be transposed
        perm:         new axis order with respect to old order
        name:         output variable name

    Returns: transpose layer

    """
    input_tensor.outputs.clear()

    attrs = {'perm': perm}

    return self.layer(op="Transpose", attrs=attrs,
                      name=name + '_node',
                      inputs=[input_tensor],
                      outputs=[gs.Variable(name=name, dtype=np.float32, shape=None)])


@gs.Graph.register()
def onnx_zero_pad(self, name, input_tensor, pad):
    """
    Args:
        self:
        input_tensor:
        output_tensor:
        pad:

    Returns: zero-pad layer

    """
    return self.layer(op="Pad",
                      name=name + '_node',
                      inputs=[input_tensor, gs.Constant(name='zero_pad_values', values=pad)],
                      outputs=[gs.Variable(name=name, dtype=np.float32, shape=None)])


def add_lrelu_nodes(graph, alpha=0.3, verbose=True):

    lrelu_nodes = find_all_nodes_by_pattern(graph, '(.*)/LeakyRelu')
    for node_id, node in enumerate(lrelu_nodes):
        input_tensor_to_relu = node.inputs[0]
        next_node = node.outputs[0].outputs[0]
        node.inputs.clear()
        node.outputs.clear()
        graph.add_trt_lrelu_layer(inputs_list=[input_tensor_to_relu],
                                  attrs={"negSlope": alpha},
                                  name=node.name,
                                  verbose=verbose)
        next_node_id = node_idx_by_name(graph=graph, node_name=next_node.name, verbose=False)
        graph.nodes[next_node_id].inputs.pop(0)
        graph.nodes[next_node_id].inputs.insert(0, graph.tensors()[f'{node.name}:0'])
        if verbose:
            print(f'Leaky ReLU tensor output: {node.name}:0')

        old_node_id = node_idx_by_name(graph=graph, node_name=node.name, verbose=False)
        graph.nodes.pop(old_node_id)

    return graph


def node_idx_by_name(graph, node_name, mode='equal', verbose=True):
    _modes = ['equal', 'part']
    if mode not in _modes:
        raise ValueError(f'node_idx_by_name accepts modes: {_modes}')

    if mode == 'equal':
        node_id = [idx for idx, x in enumerate(graph.nodes) if x.name == node_name]
        if verbose:
            print('Node name: ', node_name)
            print('Nodes: ', node_id)
    else:
        node_id = [idx for idx, x in enumerate(graph.nodes) if re.match(node_name, x.name)]
        print('Node name: ', node_name)
        print('Nodes: ', node_id)
    if len(node_id) != 1:
        raise ValueError(f'Only one node should be associated with a passed name: {node_name}.' +
                         f'Found: {node_id}: {[graph.nodes[i].name for i in node_id]}')
    node_id = node_id[0]
    return node_id


def find_tensor_by_pattern(graph, pattern, verbose=False):
    result = [k for k in graph.tensors().keys() if re.match(pattern, k)]
    if verbose:
        print(result)
    assert len(result) == 1
    return result[0]


def find_all_nodes_by_pattern(graph, pattern, verbose=False):
    nodes_list = [n for n in graph.nodes if re.match(pattern, n.name)]
    if verbose:
        print(nodes_list)
    return nodes_list


def modify_onnx_model(model_path, config, output_names=None, verbose=False):
    """
    Modify Mask-RCNN onnx model for TensorRT optimization.
    Args:
        model_path:   ONNX model path, str
        config:       Mask-RCNN config, dict
        verbose:      Additional info about graph modification, bool
        output_names: Optional.

                      Default outputs:
                      ['mrcnn_detection', 'mrcnn_mask']

                      Possible output names according to the original graph:
                      ['mrcnn_detection',
                      'mask_rcnn_inference/fpnclf_mrcnn_class/activation_([0-9]+)/Softmax:0',
                      'fpnclf_mrcnn_bbox_reshape',
                      'mrcnn_mask',
                      'roi',
                      'concat_rpn_bbox',
                      'concat_rpn_class'
                      ]

                      Other tensors in the modified graph are also possible to be an output.
                      Note, that for each additional output tensor data type must be set.
    Returns: None

    """

    # Load graph
    graph = gs.import_onnx(onnx.load(model_path))

    # Print info about initial graph inputs and outputs
    print(f'\nInitial graph inputs: {graph.inputs}')
    print(f'\nInitial graph outputs: {graph.outputs}')

    graph.inputs = [graph.tensors()['input_image']]
    graph.outputs.clear()

    #  Remove tensorflow layers: DetectionLayer, ProposalLayer by cleaning its tensors inputs and outputs
    variables = []
    for layer_name in ['mrcnn_detection', 'mask_rcnn_inference/roi/',
                       'roi_align_classifier/', 'roi_align_mask/'
                       ]:
        variables.extend([x for x in list(graph.tensors().keys()) if layer_name in x])
    for v in variables:
        try:
            graph.tensors()[v].inputs.clear()
            graph.tensors()[v].outputs.clear()
        except:
            if verbose:
                print(f'Already cleared: {v}')

    # Add LReLU_TRT nodes if necessary
    graph = add_lrelu_nodes(graph)

    # Add ResizeNearest_TRT
    graph = add_trt_resize_nearest(graph, config=config, verbose=verbose)

    # Add ProposalLayer_TRT
    graph.onnx_reshape_layer(input_tensor=graph.tensors()['concat_rpn_class'],
                             reshape=np.array([1, -1, 2, 1]),
                             name='concat_rpn_class_expanded',
                             info_shape=[1, -1, 2, 1]
                             )
    graph.onnx_reshape_layer(input_tensor=graph.tensors()['concat_rpn_bbox'],
                             reshape=np.array([1, -1, 4, 1]),
                             name='concat_rpn_bbox_expanded',
                             info_shape=[1, -1, 4, 1]
                             )
    graph.add_trt_proposal_layer(rpn_class=graph.tensors()['concat_rpn_class_expanded'],
                                 rpn_bbox=graph.tensors()['concat_rpn_bbox_expanded'],
                                 attrs={'prenms_topk': 1024,
                                        'keep_topk': config['post_nms_rois_inference'],
                                        'iou_threshold': config['rpn_nms_threshold'],
                                        'image_size': (config['image_shape'][2], *config['image_shape'][:2])
                                        },
                                 verbose=verbose
                                 )
    # Prepare PyramidROIAlign_TRT and fpn_classifier_graph
    graph.add_trt_pyramid_roialign(inputs_list=[graph.tensors()['roi'],
                                                graph.tensors()['mask_rcnn_inference/fpn_p2/BiasAdd:0'],
                                                graph.tensors()['mask_rcnn_inference/fpn_p3/BiasAdd:0'],
                                                graph.tensors()['mask_rcnn_inference/fpn_p4/BiasAdd:0'],
                                                graph.tensors()['mask_rcnn_inference/fpn_p5/BiasAdd:0'],
                                                ],
                                   pool_size=config['pool_size'],
                                   name='roi_align_classifier',
                                   clear_outputs=False,
                                   verbose=verbose
                                   )
    graph.onnx_transpose_layer(input_tensor=graph.tensors()['roi_align_classifier'],
                               perm=np.array([0, 1, 3, 4, 2]),
                               name='roi_align_classifier_transpose'
                               )

    # Connect roi_align_classifier_transpose output to fpn_classifier_graph
    node_id = node_idx_by_name(graph=graph, node_name='mask_rcnn_inference/mrcnn_class_conv1/Reshape')
    new_inputs = [graph.tensors()['roi_align_classifier_transpose']]
    new_inputs.extend(graph.nodes[node_id].inputs)
    graph.nodes[node_id].inputs.clear()
    graph.nodes[node_id].inputs.extend(new_inputs)

    # Connect shapes to roi_align_classifier
    node_id = node_idx_by_name(graph=graph, node_name='mask_rcnn_inference/mrcnn_class_conv1/Shape')
    graph.nodes[node_id].inputs.extend([graph.tensors()['roi_align_classifier_transpose']])

    # Connect roi_align_classifier output to DetectionLayer_TRT node
    _tensor_names = list(graph.tensors().keys())
    if 'mask_rcnn_inference/fpnclf_mrcnn_bbox_fc/dense_1/BiasAdd:0' in _tensor_names:
        _input_tensor_fpnclf = 'mask_rcnn_inference/fpnclf_mrcnn_bbox_fc/dense_1/BiasAdd:0'
    else:
        _input_tensor_fpnclf = 'mask_rcnn_inference/fpnclf_mrcnn_bbox_fc/dense_1/MatMul:0'
    graph.onnx_reshape_layer(input_tensor=graph.tensors()[_input_tensor_fpnclf],
                             reshape=np.array([1, config['post_nms_rois_inference'], 4 * config['num_classes'], 1, 1]),
                             name='mrcnn_bbox_reshaped',
                             info_shape=[1, config['post_nms_rois_inference'], 4 * config['num_classes'], 1, 1]
                             )

    fpnclf_pattern = r'mask_rcnn_inference/fpnclf_mrcnn_class/activation_([0-9]+)/Softmax:0'
    fpnclf_act = [k for k in graph.tensors().keys() if re.match(fpnclf_pattern, k)][0]
    graph.onnx_reshape_layer(
        input_tensor=graph.tensors()[fpnclf_act],
        reshape=np.array([1, config['post_nms_rois_inference'], config['num_classes'], 1, 1]),
        name='mrcnn_class_reshaped',
        info_shape=[1, config['post_nms_rois_inference'], config['num_classes'], 1, 1]
    )
    # Set data type to one of the original Mask-RCNN output
    graph.tensors()[fpnclf_act].dtype = np.float32

    # Add DetectionLayer_TRT
    graph.add_trt_detection_layer(
        inputs_list=[graph.tensors()['mrcnn_bbox_reshaped'],
                     graph.tensors()['mrcnn_class_reshaped'],
                     graph.tensors()['roi'],
                     ],
        attrs={'num_classes': config['num_classes'],
               'keep_topk': config['detection_max_instances'],
               'score_threshold': config['detection_min_confidence'],
               'iou_threshold': config['detection_nms_threshold']
               },
        verbose=verbose
    )

    # Add SpecialSlice_TRT
    graph.add_trt_special_slice_layer(inputs_list=[graph.tensors()['mrcnn_detection']],
                                      name='mrcnn_detection_boxes',
                                      verbose=verbose
                                      )

    # Add PyramidROIAlign_TRT for fpn_mask_graph
    graph.add_trt_pyramid_roialign(inputs_list=[graph.tensors()['mrcnn_detection_boxes'],
                                                graph.tensors()['mask_rcnn_inference/fpn_p2/BiasAdd:0'],
                                                graph.tensors()['mask_rcnn_inference/fpn_p3/BiasAdd:0'],
                                                graph.tensors()['mask_rcnn_inference/fpn_p4/BiasAdd:0'],
                                                graph.tensors()['mask_rcnn_inference/fpn_p5/BiasAdd:0'],
                                                ],
                                   pool_size=config['mask_pool_size'],
                                   name='roi_align_mask',
                                   clear_outputs=False,
                                   verbose=verbose
                                   )
    graph.onnx_transpose_layer(input_tensor=graph.tensors()['roi_align_mask'],
                               perm=np.array([0, 1, 3, 4, 2]),
                               name='roi_align_mask_transpose'
                               )

    # Connect PyramidROIAlign_TRT to fpn_mask_graph
    node_id = node_idx_by_name(graph=graph, node_name='mask_rcnn_inference/mrcnn_mask_conv1/Reshape')
    new_inputs = [graph.tensors()['roi_align_mask_transpose']]
    new_inputs.extend(graph.nodes[node_id].inputs)
    graph.nodes[node_id].inputs.clear()
    graph.nodes[node_id].inputs.extend(new_inputs)

    # Save for debugging fpnclf_mrcnn_bbox_reshape
    node_id = node_idx_by_name(graph=graph, node_name='mask_rcnn_inference/fpnclf_mrcnn_bbox_reshape/Reshape')
    graph.nodes[node_id].inputs.insert(0,
                                       graph.tensors()[_input_tensor_fpnclf])

    # Remove reshape nodes for fpnclf_mrcnn_class
    remove_nodes = ['mask_rcnn_inference/fpnclf_mrcnn_class/Shape',
                    'mask_rcnn_inference/fpnclf_mrcnn_class/Shape__',
                    'mask_rcnn_inference/fpnclf_mrcnn_class/Reshape_1/shape_Concat__',
                    'mask_rcnn_inference/fpnclf_mrcnn_class/Reshape_1__',
                    'mask_rcnn_inference/fpnclf_mrcnn_class/Reshape_1']

    rmv_node_ids = [idx for idx, x in enumerate(graph.nodes) if x.name in remove_nodes]
    for idx in rmv_node_ids:
        graph.nodes[idx].inputs.clear()
        graph.nodes[idx].outputs.clear()
        if verbose:
            print(f'Removed node: {graph.nodes[idx]}')

    # Zero Pad fix for ResNet, SE-ResNet - backbones
    if 'resnet' in config['backbone'] or 'seresnet' in config['backbone']:
        _bbone_name = config['backbone']

        if f'mask_rcnn_inference/backbone_{_bbone_name}/relu0/LeakyRelu:0' in graph.tensors().keys():
            tensor_name = f'mask_rcnn_inference/backbone_{_bbone_name}/relu0/LeakyRelu:0'
        else:
            tensor_name = f'mask_rcnn_inference/backbone_{_bbone_name}/relu0/Relu:0'

        graph.onnx_zero_pad(name='zero_padding',
                            input_tensor=graph.tensors()[tensor_name],
                            pad=np.array([0, 1, 1, 0]))
        # Clear old padding node
        node_id = node_idx_by_name(graph=graph,
                                   node_name=f'mask_rcnn_inference/backbone_{_bbone_name}/zero_padding2d_1/Pad')
        graph.nodes[node_id].inputs.clear()
        graph.nodes[node_id].outputs.clear()
        # Connect new padding node
        node_id = node_idx_by_name(graph=graph,
                                   node_name=f'mask_rcnn_inference/backbone_{_bbone_name}/pooling0/MaxPool')
        graph.nodes[node_id].inputs = [graph.tensors()['zero_padding']]

    # Zero Pad fix for MobileNet backbones
    if 'mobilenet' in config['backbone']:
        pad_nodes = {'mobilenet': ['mask_rcnn_inference/backbone_mobilenet/conv_pad_2/Pad',
                                   'mask_rcnn_inference/backbone_mobilenet/conv_pad_4/Pad',
                                   'mask_rcnn_inference/backbone_mobilenet/conv_pad_6/Pad',
                                   'mask_rcnn_inference/backbone_mobilenet/conv_pad_12/Pad',
                                   ],
                     'mobilenetv2': ['mask_rcnn_inference/backbone_mobilenetv2/block_1_pad/Pad',
                                     'mask_rcnn_inference/backbone_mobilenetv2/block_3_pad/Pad',
                                     'mask_rcnn_inference/backbone_mobilenetv2/block_6_pad/Pad',
                                     'mask_rcnn_inference/backbone_mobilenetv2/block_13_pad/Pad',
                                     ]
                     }[config['backbone']]
        for node_name in pad_nodes:
            node_id = node_idx_by_name(graph=graph, node_name=node_name)
            graph.nodes[node_id].inputs = [graph.nodes[node_id].inputs[0],
                                           gs.Constant(name='zero_pad_values', values=np.array([0, 1, 1, 0]))]

    # Reconnect fpn classifier nodes
    for node_name in ['mask_rcnn_inference/mrcnn_class_conv1/Shape',
                      'mask_rcnn_inference/mrcnn_class_conv1/Reshape_1',
                      'mask_rcnn_inference/mrcnn_class_bn1/Reshape',
                      'mask_rcnn_inference/mrcnn_class_bn1/Reshape_1',
                      'mask_rcnn_inference/mrcnn_class_conv2/Shape',
                      'mask_rcnn_inference/mrcnn_class_conv2/Reshape_1',
                      'mask_rcnn_inference/mrcnn_class_bn2/Reshape',
                      'mask_rcnn_inference/mrcnn_class_bn2/Reshape_1', ]:
        node_id = node_idx_by_name(graph=graph, node_name=node_name)
        graph.nodes[node_id].inputs.clear()
        graph.nodes[node_id].outputs.clear()

    special_nodes = []
    pattern_list = [r'(\w+)/mrcnn_class_bn1/batch_normalization/FusedBatchNormV3__([0-9]+):0',
                    r'(\w+)/mrcnn_class_conv2/conv2d_([0-9]+)/BiasAdd__([0-9]+):0',
                    r'(\w+)/mrcnn_class_bn2/batch_normalization_1/FusedBatchNormV3__([0-9]+):0'
                    ]
    for pattern in pattern_list:
        special_nodes.extend([x[:-2] for x in graph.tensors().keys() if re.match(pattern, x)])

    node_pairs = [[special_nodes[0],
                   find_tensor_by_pattern(graph, r'mask_rcnn_inference/mrcnn_class_conv1/conv2d(_?)([0-9]+)?/BiasAdd:0')
                   ],
                  [r'mask_rcnn_inference/fpnclf_relu_act1/(\w+)?Relu',
                   r'mask_rcnn_inference/mrcnn_class_bn1/batch_normalization/FusedBatchNormV3:0'
                   ],
                  [special_nodes[1],
                   find_tensor_by_pattern(graph, r'mask_rcnn_inference/fpnclf_relu_act1/(\w+)?Relu:0')
                   ],
                  [special_nodes[2],
                   find_tensor_by_pattern(graph, 'mask_rcnn_inference/mrcnn_class_conv2/conv2d(_?)([0-9]+)?/BiasAdd:0')

                   ],
                  [r'mask_rcnn_inference/fpnclf_relu_act2/(\w+)?Relu',
                   r'mask_rcnn_inference/mrcnn_class_bn2/batch_normalization_1/FusedBatchNormV3:0'
                   ]
                  ]

    for pair in node_pairs:
        node_name, tensor_name = pair
        node_id = node_idx_by_name(graph=graph, node_name=node_name, mode='part', verbose=verbose)

        if len(graph.nodes[node_id].inputs) == 1:
            graph.nodes[node_id].inputs = [graph.tensors()[tensor_name]]
        else:
            graph.nodes[node_id].inputs = [graph.tensors()[tensor_name], graph.nodes[node_id].inputs[1]]

    """
    Outputs according to the original graph:
    
    graph.tensors()['mrcnn_detection'],
    graph.tensors()['mask_rcnn_inference/fpnclf_mrcnn_class/activation_([0-9]+)/Softmax:0'],
    graph.tensors()['fpnclf_mrcnn_bbox_reshape'],
    graph.tensors()['mrcnn_mask'],
    graph.tensors()['roi'],
    graph.tensors()['concat_rpn_bbox'],
    graph.tensors()['concat_rpn_class'], 
    """

    if output_names:
        graph.outputs = [graph.tensors()[x] for x in output_names]
    else:
        graph.outputs = [graph.tensors()['mrcnn_detection'], graph.tensors()['mrcnn_mask']]

    graph.cleanup()
    graph.toposort()

    if verbose:
        # Print info about initial graph inputs and outputs
        print(f'\nModified graph inputs: {graph.inputs}')
        print(f'\nModified graph outputs: {graph.outputs}')

    new_model_path = model_path.replace('.onnx', '_trt_mod.onnx')
    onnx.save(gs.export_onnx(graph), new_model_path)
    print(f'\nModel {model_path}  was successfully modified for TensorRT optimization: {new_model_path}')
