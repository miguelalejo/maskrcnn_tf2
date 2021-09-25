#!/usr/bin/env python3
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
This script runs an identity model with ONNX-Runtime and TensorRT,
then compares outputs.
"""

import sys
from polygraphy import constants as POLYGRAPHY_CONST
POLYGRAPHY_CONST.INTERNAL_CORRECTNESS_CHECKS = True
from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnx
from polygraphy.backend.trt import EngineFromNetwork, NetworkFromOnnxPath, TrtRunner
from polygraphy.comparator import Comparator


def main(onnx_model, trt_prepared_onnx_model):
    """

    Args:
        onnx_model:
        trt_prepared_onnx_model:

    Returns: None

    """
    # The OnnxrtRunner requires an ONNX-RT session.
    # We can use the SessionFromOnnx lazy loader to construct one easily:
    build_onnxrt_session = SessionFromOnnx(onnx_model)

    # The TrtRunner requires a TensorRT engine.
    # To create one from the ONNX model, we can chain a couple lazy loaders together:
    build_engine = EngineFromNetwork(NetworkFromOnnxPath(trt_prepared_onnx_model))

    runners = [
        TrtRunner(build_engine),
        OnnxrtRunner(build_onnxrt_session),
    ]

    # `Comparator.run()` will run each runner separately using synthetic input data and
    #   return a `RunResults` instance. See `polygraphy/comparator/struct.py` for details.
    #
    # TIP: To use custom input data, you can set the `data_loader` parameter in `Comparator.run()``
    #   to a generator or iterable that yields `Dict[str, np.ndarray]`.
    run_results = Comparator.run(runners)

    # `Comparator.compare_accuracy()` checks that outputs match between runners.
    assert bool(Comparator.compare_accuracy(run_results))

    # We can use `RunResults.save()` method to save the inference results to a JSON file.
    # This can be useful if you want to generate and compare results separately.
    run_results.save("inference_results.json")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Pass paths to ONNX models:\n\tArgument 1: onnx_model\n\tArgument 2: onnx_trt_prepared_model')
        sys.exit(1)
    else:
        main(onnx_model=sys.argv[1], trt_prepared_onnx_model=sys.argv[2])
        sys.exit(0)
