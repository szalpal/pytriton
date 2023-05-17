#!/usr/bin/env python3
# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import torch

from pytriton.decorators import batch
from pytriton.model_config import DynamicBatcher, ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types

from model_inference import SegmentationPyTorch

MAX_BATCH_SIZE = 32


@pipeline_def(batch_size=MAX_BATCH_SIZE, num_threads=4, device_id=0, prefetch_queue_depth=1)
def dali_preprocessing_pipe():
    encoded = fn.external_source(device='cpu', name='encoded')  # Input from PyTriton is always on CPU.
    decoded = fn.decoders.image(encoded, device='mixed', output_type=types.RGB)
    preprocessed = fn.resize(decoded, resize_x=224, resize_y=224)
    preprocessed = fn.crop_mirror_normalize(preprocessed,
                                            dtype=types.FLOAT,
                                            output_layout='CHW',
                                            crop=(224, 224),
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    return decoded, preprocessed


@pipeline_def(batch_size=MAX_BATCH_SIZE, num_threads=4, device_id=0, prefetch_queue_depth=1)
def dali_postprocessing_pipe(class_idx=0, prob_threshold=0.6):
    image = fn.external_source(device='gpu', name='image', layout='HWC')
    image = fn.transpose(image, perm=(2, 0, 1))
    width = fn.cast(fn.external_source(device='cpu', name='width'), dtype=types.FLOAT)
    height = fn.cast(fn.external_source(device='cpu', name='height'), dtype=types.FLOAT)
    prob = fn.external_source(device='gpu', name='probabilities', layout='CHW')
    prob = fn.resize(prob, resize_x=width, resize_y=height, interp_type=types.DALIInterpType.INTERP_NN)
    prob = fn.cast(prob > prob_threshold, dtype=types.UINT8)
    prob = fn.stack(prob[class_idx], prob[class_idx], prob[class_idx])
    return fn.transpose(image * prob, perm=(1, 2, 0))


preprocessing_pipe = dali_preprocessing_pipe()
preprocessing_pipe.build()
postprocessing_pipe = dali_postprocessing_pipe()
postprocessing_pipe.build()


def dali_tensorlist_to_torch_tensor(tensorlist_gpu):
    import cupy as cp
    dali_t = tensorlist_gpu.as_tensor()
    cp_t = cp.asarray(dali_t)
    torch_t = torch.tensor(cp_t)
    return torch_t.cuda()


def preprocess(images):
    preprocessing_pipe.feed_input("encoded", images)
    imgs, preprocessed = preprocessing_pipe.run()
    return imgs, dali_tensorlist_to_torch_tensor(preprocessed)


def postprocess(images, probabilities):
    postprocessing_pipe.feed_input("image", images, layout='HWC')
    postprocessing_pipe.feed_input("probabilities", probabilities, layout='CHW')
    image_sizes = np.transpose(np.array(images.shape()))
    postprocessing_pipe.feed_input("height", image_sizes[0])
    postprocessing_pipe.feed_input("width", image_sizes[1])
    img, = postprocessing_pipe.run()
    return img


segmentation = SegmentationPyTorch(
    seg_class_name="__background__",
    device_id=0,
)


@batch
def _infer_fn(**enc):
    enc = enc["image"]

    image, input = preprocess(enc)
    prob = segmentation(input)
    out = postprocess(image, prob)

    return {
        "original": image.as_cpu().as_array(),
        "segmented": out.as_cpu().as_array(),
    }


def main():
    with Triton(config=TritonConfig(log_verbose=1)) as triton:
        triton.bind(
            model_name="ResNet",
            infer_func=_infer_fn,
            inputs=[
                Tensor(name="image", dtype=np.uint8, shape=(-1,)),  # Encoded image
            ],
            outputs=[
                Tensor(name="original", dtype=np.uint8, shape=(-1, -1, -1)),
                Tensor(name="segmented", dtype=np.uint8, shape=(-1, -1, -1)),
            ],
            config=ModelConfig(
                max_batch_size=MAX_BATCH_SIZE,
                batcher=DynamicBatcher(max_queue_delay_microseconds=5000),
            ),
        )
        triton.serve()


if __name__ == "__main__":
    main()
