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

MAX_BATCH_SIZE = 5
SEQUENCE_LENGTH = 4


@pipeline_def(batch_size=MAX_BATCH_SIZE, num_threads=4, device_id=0, prefetch_queue_depth=1)
def dali_preprocessing_pipe():
    decoded = fn.experimental.inputs.video(name='encoded', sequence_length=4, device='mixed')
    preprocessed = fn.resize(decoded, resize_x=224, resize_y=224)
    preprocessed = fn.crop_mirror_normalize(preprocessed,
                                            dtype=types.FLOAT,
                                            output_layout='FCHW',
                                            crop=(224, 224),
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    return decoded, preprocessed


@pipeline_def(batch_size=MAX_BATCH_SIZE * SEQUENCE_LENGTH, num_threads=4, device_id=0, prefetch_queue_depth=1)
def dali_postprocessing_pipe(class_idx=0, prob_threshold=0.6):
    image = fn.external_source(device='gpu', name='image', layout='HWC')
    width = fn.cast(fn.external_source(device='cpu', name='width'), dtype=types.FLOAT)
    height = fn.cast(fn.external_source(device='cpu', name='height'), dtype=types.FLOAT)
    prob = fn.external_source(device='gpu', name='probabilities', layout='CHW')
    prob = fn.expand_dims(prob[class_idx], axes=[2], new_axis_names='C')
    prob = fn.resize(prob, resize_x=width, resize_y=height, interp_type=types.DALIInterpType.INTERP_NN)
    mask = fn.cast(prob > prob_threshold, dtype=types.UINT8)
    return image * mask


preprocessing_pipe = dali_preprocessing_pipe()
preprocessing_pipe.build()
postprocessing_pipe = dali_postprocessing_pipe()
postprocessing_pipe.build()


def preprocess(images):
    preprocessing_pipe.feed_input("encoded", images)
    imgs, preprocessed = preprocessing_pipe.run()
    return torch.as_tensor(imgs), torch.as_tensor(preprocessed)


def postprocess(images, probabilities):
    postprocessing_pipe.feed_input("image", images, layout='HWC')
    postprocessing_pipe.feed_input("probabilities", probabilities, layout='CHW')
    postprocessing_pipe.feed_input("height", np.full(images.shape[0], images.shape[1]))
    postprocessing_pipe.feed_input("width", np.full(images.shape[0], images.shape[2]))
    img, = postprocessing_pipe.run()
    return img


segmentation = SegmentationPyTorch(
    seg_class_name="__background__",
    device_id=0,
)


@batch
def _infer_fn(**enc):
    enc = enc["video"]

    image, input = preprocess(enc)

    input = input.reshape(-1, *input.shape[-3:])  # NFCHW to NCHW (flattening first two dimensions)
    image = image.reshape(-1, *image.shape[-3:])  # NFCHW to NCHW (flattening first two dimensions)

    prob = segmentation(input)
    out = postprocess(image, prob)

    return {
        "original": image.cpu().numpy(),
        "segmented": out.as_cpu().as_array(),
    }


def main():
    with Triton(config=TritonConfig(log_verbose=1)) as triton:
        triton.bind(
            model_name="ResNet",
            infer_func=_infer_fn,
            inputs=[
                Tensor(name="video", dtype=np.uint8, shape=(-1,)),  # Encoded image
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
