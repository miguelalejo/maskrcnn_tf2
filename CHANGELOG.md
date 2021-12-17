# Change Log
All notable changes to the maskrcnn with tensorflow 2 will be documented in this file.

 
## [Unreleased] - 2021-09-25

### Added
- ResNet's backbones got `resnet_leaky_relu` option.
- Mask and classifier heads also got `mask_head_leaky_relu` and `cls_head_leaky_relu` options.
- Extended ResNets and EfficientNets backbones: added ResNet152, EfficientNet [B4, B5, B6, B7] backbones.
- ResNeXt backbones [50, 101].
- SE-ResNet backbones [18, 34, 50, 101, 152].
- SE-ResNeXt backbones [50, 101].
- SE-Net backbones [154].
 
### Changed
- New tensorboard log folder format: `[maskrcnn_<BACKBONE_NAME>_<YYYY-MM-DD_hh-mm-ss>]`
- `logs` folder now is generated outside `src`
- Updated `README.md`
 
### Fixed

* Normalization issue. Now normalization can be specified via model config in `src/common/config.py`. 
* ONNX ZeroPadding for TensorRT engines.

### Known issues

- There is a drop in performance after ONNX modification for TRT, also NaNs happens in TensorRT model output.
  The issue is under research.


- TensorRT: `../rtSafe/cublas/cublasLtWrapper.cpp (279) - Assertion Error in getCublasLtHeuristic: 0 (cublasStatus == CUBLAS_STATUS_SUCCESS)`
   
    * Possible cause: cuda and TensorRT versions mismatch;
    * Possible workaround if error still exists - remove cublasLt from tacticSources:
       * fp32:
          `trtexec --onnx=<PATH_TO_ONNX_GRAPH> --saveEngine=<PATH_TO_TRT_ENGINE> --tacticSources=-cublasLt,+cublas --workspace=<WORKSPACE_SIZE> --verbose`
       * fp16:
           `trtexec --onnx=<PATH_TO_ONNX_GRAPH> --saveEngine=<PATH_TO_TRT_ENGINE> --tacticSources=-cublasLt,+cublas --fp16 --workspace=<WORKSPACE_SIZE> --verbose`
    

- Tensorlfow v2.5: AttributeError: module 'keras.utils' has no attribute 'get_file'
       
   * Possible cause: the influence of keras-nightly automatically suggested with tensorflow installation.
   * Possible workaround:
         
     Open `__init__.py` in `<ANACONDA_PATH>/envs/tf2.5/lib/python3.8/site-packages/classification_models`
         
     Modify file:
     ```python
     import keras_applications as ka
     from .__version__ import __version__
     import tensorflow.keras.utils as utils
         
     def get_submodules_from_kwargs(kwargs):
         backend = kwargs.get('backend', ka._KERAS_BACKEND)
         layers = kwargs.get('layers', ka._KERAS_LAYERS)
         models = kwargs.get('models', ka._KERAS_MODELS)
         return backend, layers, models, utils
     ```
   
- `senet154`, `efficientnetb5`, `efficientnetb6`, `efficientnetb7` backbones are not tested enough for now because of the 
  high GPU memory consumption. Thus, the model with `senet154` backbone may be not supported for ONNX graph modification.

## [Unreleased] - 2021-12-17

### Fixed

* `steps_per_epoch` calculation in  `src/preprocess/preprocess.py`. 
   [Related issue](https://github.com/alexander-pv/maskrcnn_tf2/issues/9).