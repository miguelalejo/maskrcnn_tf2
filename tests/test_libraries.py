def test_tensorflow():
    import tensorflow as tf
    assert tf.__version__ in ('2.2.0', '2.3.4', '2.4.3', '2.5.1')


def test_image_classifiers():
    import classification_models as clf_models
    assert clf_models.__version__ == '1.0.0'


def test_keras():
    import keras
    assert keras.__version__ == '2.4.3'


def test_numpy():
    import numpy as np
    assert np.__version__ in ('1.18.5', '1.19.2')


def test_tqdm():
    import tqdm
    assert tqdm.__version__ == '4.46.1'


def test_pycocotools():
    from pycocotools import coco
    assert coco.__version__ == '2.0'


def test_tf2onnx():
    import tf2onnx
    assert tf2onnx.__version__ == '1.8.5'


def test_onnx():
    import onnx
    assert onnx.__version__ == '1.8.1'


def test_onnxruntime():
    import onnxruntime
    assert onnxruntime.__version__ == '1.6.0'


def test_scipy():
    import scipy
    assert scipy.__version__ == '1.4.1'


def test_albumentations():
    import albumentations
    assert albumentations.__version__ == '0.4.5'


def test_efficientnet():
    import efficientnet
    assert efficientnet.__version__ == '1.1.1'


def test_matplotlib():
    import matplotlib
    assert matplotlib.__version__ == '3.2.2'


def test_watermark():
    import watermark
    assert watermark.__version__ == '2.2.0'


def test_onnx_graphsurgeon():
    try:
        import onnx_graphsurgeon
    except:
        raise ModuleNotFoundError('onnx-graphsurgeon can be manually installed from TensorRT folder.')
    assert onnx_graphsurgeon.__version__ in ('0.2.6', '0.3.11')


def test_tensorrt():
    try:
        import tensorrt as trt
    except:
        raise ModuleNotFoundError('tensorrt python wrapper can be manually installed from TensorRT folder.')
    assert trt.__version__ == '7.2.3.4'
