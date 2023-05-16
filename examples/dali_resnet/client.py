#!/usr/bin/env python3
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

import argparse
import io
import logging
import numpy as np
import base64
from PIL import Image
from pytriton.client import ModelClient

logger = logging.getLogger("examples.dali_resnet101_pytorch.client")

IMAGE_PATHS = [
    'dog-1461239_1280.jpg',
    # 'padlock-406986_640.jpg',
]


def array_from_list(arrays):
    """
    Convert list of ndarrays to single ndarray with ndims+=1. Pad if necessary.
    """
    lengths = [arr.shape[0] for arr in arrays]
    max_len = max(lengths)
    arrays = [np.pad(arr, (0, max_len - arr.shape[0])) for arr in arrays]
    for arr in arrays:
        assert arr.shape == arrays[0].shape, "Arrays must have the same shape"
    return np.stack(arrays)


def ndarray_from_pil(pil_image):
    """
    Convert PIL image to ndarray. The ndarray will contain decoded image.
    """
    bytes = io.BytesIO()
    pil_image.save(bytes, format="JPEG")
    return np.frombuffer(bytes.getbuffer(), dtype=np.uint8)


def to_ndarray(list_of_images):
    """
    Convert list of PIL images to ndarray.
    """
    return array_from_list([ndarray_from_pil(img) for img in list_of_images])


def load_images(img_paths):
    return array_from_list([np.fromfile(f, dtype=np.uint8) for f in img_paths])

def _decode_image_from_base64(msg):
    msg = base64.b64decode(msg)
    buffer = io.BytesIO(msg)
    image = Image.open(buffer)
    return image


def infer_model(input, args):
    batch_size = input.shape[0]
    with ModelClient(args.url, "ResNet", init_timeout_s=args.init_timeout_s) as client:
        result_data = client.infer_batch(input)
        import ipdb; ipdb.set_trace()
        original = result_data['original']
        segmented = result_data['segmented']

        if args.dump_images:
            for i, (orig, segm) in enumerate(zip(original, segmented)):
                import cv2
                cv2.imwrite(f"orig{i}.jpg", orig)
                cv2.imwrite(f"segm{i}.jpg", np.transpose(segm, (1, 2, 0)))

        logger.info("Processing finished.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="localhost",
        help=(
            "Url to Triton server (ex. grpc://localhost:8001)."
            "HTTP protocol with default port is used if parameter is not provided"
        ),
        required=False,
    )
    parser.add_argument(
        "--init-timeout-s",
        type=float,
        default=600.0,
        help="Server and model ready state timeout in seconds.",
        required=False,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--dump-images",
        action="store_true",
        default=False,
        help="If True, the client will save processed images to disk. Requires cv2 module.",
        required=False,
    )
    parser.add_argument(
        "--image-paths",
        nargs='+',
        default=None,
        help="Paths of the images to process.",
        required=False,
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

    infer_model(load_images(IMAGE_PATHS), args)


if __name__ == "__main__":
    main()
