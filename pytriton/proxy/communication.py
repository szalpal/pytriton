# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
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
"""Communication utility module.

It is used for interaction between model and proxy_backend.
"""

import atexit
import ctypes
import ctypes.util
import dataclasses
import fcntl
import gc
import json
import logging
import math
import multiprocessing.managers
import multiprocessing.popen_spawn_posix
import multiprocessing.shared_memory
import pathlib
import signal
import struct
import threading
import time
import uuid
import weakref
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

LOGGER = logging.getLogger(__name__)


# copy from
# https://github.com/triton-inference-server/python_backend/blob/main/src/resources/triton_python_backend_utils.py


def _serialize_byte_tensor(tensor) -> bytes:
    """Serializes a bytes tensor into a flat numpy array of length prepended bytes.

    The numpy array should use dtype of np.object_. For np.bytes_,
    numpy will remove trailing zeros at the end of byte sequence and because
    of this it should be avoided.

    Args:
        input_tensor: The bytes tensor to serialize.

    Returns:
    serialized array as bytes buffer.

    Raises:
        UnicodeEncodeErrors: raised when try to cast to string of non-bytes items fails
    """
    if tensor.size == 0:
        return b""

    # If the input is a tensor of string/bytes objects, then must flatten those
    # into a 1-dimensional array containing the 4-byte byte size followed by the
    # actual element bytes. All elements are concatenated together in "C" order.
    assert (tensor.dtype == np.object_) or (tensor.dtype.type == np.bytes_)
    flattened_ls = []
    total_len = 0
    for obj in np.nditer(tensor, flags=["refs_ok"], order="C"):
        # If directly passing bytes to BYTES type,
        # don't convert it to str as Python will encode the
        # bytes which may distort the meaning
        if tensor.dtype == np.object_ and type(obj.item()) != bytes:
            s = str(obj.item()).encode("utf-8")
        else:
            s = obj.item()
        item_len = len(s)
        flattened_ls.append(struct.pack("<I", item_len))
        flattened_ls.append(s)
        total_len += struct.calcsize("<I") + item_len
    flattened_ls.insert(0, struct.pack("<I", total_len))
    flattened = b"".join(flattened_ls)
    return flattened


# copy from
# https://github.com/triton-inference-server/python_backend/blob/main/src/resources/triton_python_backend_utils.py
def _deserialize_bytes_tensor(encoded_tensor, dtype, order: Literal["C", "F"] = "C") -> np.ndarray:
    """Deserializes an encoded bytes tensor into an numpy array of dtype of python objects.

    Args:
        encoded_tensor : The encoded bytes tensor where each element has its length in
        first 4 bytes followed by the content
        dtype: The dtype of the numpy array to deserialize to.
        order: The order of the numpy array to deserialize to.

    Returns:
    The 1-D numpy array of type object containing the deserialized bytes in 'C' order.
    """
    strs = []
    offset = 0
    val_buf = encoded_tensor
    val_len = struct.unpack_from("<I", val_buf, offset)[0] + 4
    offset += 4
    while offset < val_len:
        item_length = struct.unpack_from("<I", val_buf, offset)[0]
        offset += 4
        item = struct.unpack_from(f"<{item_length}s", val_buf, offset)[0]
        offset += item_length
        strs.append(item)
    return np.array(strs, dtype=dtype, order=order)


_MAX_DTYPE_DESCR = 16  # up to 16 chars in dtype descr; |S2147483647 (2^31-1) with margin
_PARTIAL_HEADER_FORMAT = f"<{_MAX_DTYPE_DESCR}scH"


def _pack_header(shape: Tuple[int, ...], dtype: np.dtype, order: Literal["C", "F"] = "C") -> bytes:
    header_format = _PARTIAL_HEADER_FORMAT + "Q" * len(shape)
    dtype_descr = np.lib.format.dtype_to_descr(dtype)
    assert (
        len(dtype_descr) <= _MAX_DTYPE_DESCR
    ), f"dtype descr is too long; dtype_descr={dtype_descr} max={_MAX_DTYPE_DESCR}"
    return struct.pack(header_format, dtype_descr.encode("utf-8"), order.encode("ascii"), len(shape), *shape)


def _unpack_header(header: bytes) -> Tuple[Tuple[int, ...], np.dtype, Literal["C", "F"]]:
    shape_offset = struct.calcsize(_PARTIAL_HEADER_FORMAT)
    dtype_descr, order, ndim = struct.unpack_from(_PARTIAL_HEADER_FORMAT, header, offset=0)
    shape = struct.unpack_from("Q" * ndim, header, offset=shape_offset)
    dtype = np.lib.format.descr_to_dtype(dtype_descr.decode("utf-8").rstrip("\x00"))
    order = order.decode("ascii")
    return shape, dtype, order


def serialize_numpy_with_struct_header(tensor: np.ndarray) -> List[Union[bytes, memoryview]]:
    """Serialize numpy array to list of bytes and memoryviews.

    Args:
        tensor: numpy array to serialize

    Returns:
        List of data frames in form of bytes and memoryviews
    """
    if tensor.dtype.hasobject:
        data = _serialize_byte_tensor(tensor.ravel())
        order = "C"  # as _serialize_byte_tensor returns C-ordered array
    else:
        if not tensor.data.contiguous:
            tensor = np.ascontiguousarray(tensor)
        data = tensor.data
        order = "C" if tensor.flags.c_contiguous else "F"

    header = _pack_header(tensor.shape, tensor.dtype, order)
    frames = [header, data]
    return frames


def deserialize_numpy_with_struct_header(frames: List[Union[bytes, memoryview]]) -> np.ndarray:
    """Deserialize numpy array from list of bytes and memoryviews.

    Args:
        frames: List of data frames in form of bytes and memoryviews

    Returns:
        numpy array
    """
    header, data = frames
    shape, dtype, order = _unpack_header(header)
    if dtype.hasobject:
        tensor = _deserialize_bytes_tensor(data, dtype).reshape(shape)
    else:
        tensor = np.ndarray(shape, dtype=dtype, buffer=data, order=order)
    return tensor


def calc_serialized_size_of_numpy_with_struct_header(tensor: np.ndarray) -> List[int]:
    """Calculate size of serialized numpy array.

    Args:
        tensor: numpy array to serialize

    Returns:
        List of sizes of data frames
    """
    header_size = struct.calcsize(_PARTIAL_HEADER_FORMAT) + struct.calcsize("Q") * len(tensor.shape)
    if tensor.dtype.hasobject:
        items_sizes = []
        order = "C" if tensor.flags.c_contiguous else "F"
        for obj in np.nditer(tensor, flags=["refs_ok"], order=order):
            if tensor.dtype == np.object_ and type(obj.item()) != bytes:
                s = str(obj.item()).encode("utf-8")
            else:
                s = obj.item()
            items_sizes.append(len(s))

        # total_size + for size of each item + each item
        data_size = struct.calcsize("<I") + struct.calcsize("<I") * len(items_sizes) + sum(items_sizes)
    else:
        data_size = tensor.nbytes

    return [header_size, data_size]


@dataclasses.dataclass
class MetaRequestResponse:
    """Data class for storing input/output data and parameters."""

    idx: int
    data: Optional[Dict[str, str]] = None
    parameters: Optional[Dict[str, Union[str, int, bool]]] = None
    eos: Optional[bool] = None


@dataclasses.dataclass
class InferenceHandlerRequests:
    """Object transferred from proxy backend to callback handler containing input data."""

    requests: List[MetaRequestResponse]

    @classmethod
    def from_bytes(cls, content: bytes) -> "InferenceHandlerRequests":
        """Reconstruct InferenceHandlerRequests object from bytes.

        Args:
            content: bytes to parse
        """
        requests = json.loads(content)
        return cls(
            requests=[
                MetaRequestResponse(
                    idx=request.get("idx"),
                    data=request.get("data", {}),
                    parameters=request.get("parameters"),
                )
                for request in requests["requests"]
            ]
        )

    def as_bytes(self) -> bytes:
        """Serializes InferenceHandlerRequests object to bytes."""
        requests = {
            "requests": [
                {
                    "idx": request.idx,
                    "data": request.data,
                    "parameters": request.parameters,
                }
                for request in self.requests
            ]
        }
        return json.dumps(requests).encode("utf-8")


@dataclasses.dataclass
class InferenceHandlerResponses:
    """Object transferred from callback handler containing output data."""

    responses: Optional[List[MetaRequestResponse]] = None
    error: Optional[str] = None

    @classmethod
    def from_bytes(cls, content: bytes) -> "InferenceHandlerResponses":
        """Reconstruct InferenceHandlerResponses object from bytes.

        Args:
            content: bytes to parse
        """
        responses = json.loads(content)
        return cls(
            responses=[
                MetaRequestResponse(idx=response.get("idx"), data=response.get("data", {}), eos=response.get("eos"))
                for response in responses.get("responses", [])
            ],
            error=responses.get("error"),
        )

    def as_bytes(self) -> bytes:
        """Serializes InferenceHandlerResponses object to bytes."""
        result = {"error": self.error}
        if self.responses:
            result["responses"] = [
                {"idx": response.idx, "data": response.data, "eos": response.eos} for response in self.responses
            ]
        return json.dumps(result).encode("utf-8")


@dataclasses.dataclass
class BlockDescriptor:
    """Descriptor of block in shared memory."""

    shm_name: str
    offset: int
    size: Optional[int] = None

    def __post_init__(self):
        """Initialize other attributes."""
        self.id = f"{self.shm_name}:{self.offset}"

    @classmethod
    def from_id(cls, tensor_id: str):
        """Create BlockDescriptor from dict."""
        shm_name, offset = tensor_id.split(":")
        return cls(shm_name, int(offset))


class _SharedMemorySegment:
    def __init__(self, size):
        self.shared_memory = multiprocessing.shared_memory.SharedMemory(create=True, size=size)
        multiprocessing.util.debug(f"Created {self.shared_memory.name} of size {self.shared_memory.size}")
        self.used_blocks: List[BlockDescriptor] = []
        self.used_blocks_lock = threading.RLock()
        self.free_blocks = [BlockDescriptor(self.shared_memory.name, offset=0, size=size)]
        self.max_free_block_size = size

    def _update_free_blocks(self):
        total_size = self.shared_memory.size
        free_blocks = []
        offset = 0

        with self.used_blocks_lock:
            # find holes between used blocks
            for used_block in self.used_blocks:
                if used_block.offset > offset:
                    free_blocks.append(
                        BlockDescriptor(self.shared_memory.name, offset=offset, size=used_block.offset - offset)
                    )
                offset = used_block.offset + used_block.size
        # if tail is free
        if offset < total_size:
            free_blocks.append(BlockDescriptor(self.shared_memory.name, offset=offset, size=total_size - offset))

        self.free_blocks = free_blocks
        self.max_free_block_size = max(block.size for block in self.free_blocks) if self.free_blocks else 0

    def __contains__(self, block_id: str) -> bool:
        with self.used_blocks_lock:
            return any(block_id == block.id for block in self.used_blocks)  # pytype: disable=attribute-error

    def __getitem__(self, block_id: str) -> BlockDescriptor:
        with self.used_blocks_lock:
            for block in self.used_blocks:
                if block.id == block_id:  # pytype: disable=attribute-error
                    return block
        raise KeyError(f"Block with id {block_id} not found in segment {self.shared_memory.name}")

    def allocate(self, offset, byte_size):
        block = BlockDescriptor(self.shared_memory.name, offset=offset, size=byte_size)
        with self.used_blocks_lock:
            self.used_blocks.append(block)
            self.used_blocks.sort(key=lambda block: block.offset)
            self._update_free_blocks()
        return block

    def release(self, block: BlockDescriptor):
        with self.used_blocks_lock:
            self.used_blocks.remove(block)
            self._update_free_blocks()


class _DataBlocksServer:
    _instance = None
    _cnt = 0
    _minimal_segment_size = 4096  # 4KB

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # WAR: for some reason, the __init__ is called on each create of proxy object
        if self._cnt == 1:
            return
        self._cnt += 1
        self._id = uuid.uuid4()  # to verify that it is singleton across processes
        self._segments = []
        self._segments_lock = threading.RLock()
        atexit.register(self.close)

    def get_free_blocks(self, bytes_sizes: Sequence[int]) -> Sequence[str]:
        tensors_ids = []
        with self._segments_lock:
            for byte_size in bytes_sizes:
                for segment in self._segments:
                    if segment.max_free_block_size >= byte_size:
                        for free_block in segment.free_blocks:
                            if free_block.size >= byte_size:
                                block = self._allocate_block(segment, free_block.offset, byte_size)
                                tensors_ids.append(block.id)  # pytype: disable=attribute-error
                                break
                    else:
                        continue  # If no suitable block was found, try the next segment
                    break  # If a suitable block was found, don't try any more segments
                else:  # If no suitable block was found in any segment
                    new_segment_size = int(
                        max(self._minimal_segment_size, math.pow(2, math.ceil(math.log2(byte_size))))
                    )
                    block = self._allocate_block(
                        self._create_new_segment(new_segment_size), offset=0, byte_size=byte_size
                    )
                    tensors_ids.append(block.id)  # pytype: disable=attribute-error
        return tensors_ids

    def release_block(self, block_id: str):
        with self._segments_lock:
            for segment in self._segments:
                try:
                    block = segment[block_id]
                    segment.release(block)
                    return
                except KeyError:
                    pass
        raise KeyError(f"Block with id {block_id} not found in server")

    def _allocate_block(self, segment: _SharedMemorySegment, offset: int, byte_size: int) -> BlockDescriptor:
        return segment.allocate(offset, byte_size)

    def _create_new_segment(self, segment_size):
        segment = _SharedMemorySegment(segment_size)
        self._segments.append(segment)
        return segment

    def _get_debug_status(self):
        return {
            "server_id": str(self._id),
            "host_pid": multiprocessing.current_process().pid,
            "segments": [
                {
                    "shared_memory": segment.shared_memory.name,
                    "used_blocks": [str(block) for block in segment.used_blocks],
                }
                for segment in self._segments
            ],
        }

    def close(self):
        multiprocessing.util.debug(f"Closing server {self._id}")
        with self._segments_lock:
            while self._segments:
                segment = self._segments.pop()
                multiprocessing.util.debug(f"Closing and delete segment {segment.shared_memory.name}")
                segment.shared_memory.close()
                segment.shared_memory.unlink()


class BlocksStoreManager(multiprocessing.managers.BaseManager):
    """Remote block store for storing and retrieving numpy arrays in/from shared memory."""

    @classmethod
    def _run_server(cls, registry, address, authkey, serializer, writer, initializer=None, initargs=()):
        PR_SET_PDEATHSIG = 1  # noqa
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)  # terminate process when parent **thread** dies
        super()._run_server(
            registry, address, authkey, serializer, writer, initializer, initargs
        )  # pytype: disable=attribute-error


class _DataBlocksServerProxy(multiprocessing.managers.BaseProxy):
    def release_block(self, /, *args, **kwargs):
        return self._callmethod("release_block", args, kwargs)

    def get_free_blocks(self, /, *args, **kwargs):
        return self._callmethod("get_free_blocks", args, kwargs)

    def _get_debug_status(self, /, *args, **kwargs):
        return self._callmethod("_get_debug_status", args, kwargs)

    def close(self, /, *args, **kwargs):
        return self._callmethod("close", args, kwargs)


BlocksStoreManager.register("blocks", _DataBlocksServer, proxytype=_DataBlocksServerProxy)


class _FileLock:
    _locks = {}

    def __new__(cls, file_path):
        if file_path not in cls._locks:
            cls._locks[file_path] = super().__new__(cls)
        return cls._locks[file_path]

    def __init__(self, file_path):
        if hasattr(self, "_file_path"):
            return
        self._file_path = pathlib.Path(file_path)
        self._file_lock = None
        self._lock = threading.RLock()
        atexit.register(self._clean)

    def __enter__(self):
        self._file_lock = self._file_path.open("a")
        fcntl.flock(self._file_lock.fileno(), fcntl.LOCK_EX)
        self._lock.acquire()

    def __exit__(self, exc_type, exc_value, traceback):
        fcntl.flock(self._file_lock.fileno(), fcntl.LOCK_UN)
        self._lock.release()

    def _clean(self):
        if self._file_lock is not None:
            self._file_lock.close()
        try:
            self._file_path.unlink(missing_ok=True)
        except OSError as e:
            LOGGER.warning(f"Could not remove lock file {self._file_path}; {e}")


class _Popen(multiprocessing.popen_spawn_posix.Popen):
    def _launch(self, process_obj):
        # Modified version of multiprocessing.popen_spawn_posix.Popen._launch
        import io
        import os
        from multiprocessing import context, resource_tracker, spawn, util

        tracker_fd = resource_tracker.getfd()
        self._fds.append(tracker_fd)  # pytype: disable=attribute-error

        # get prep_data + remove init_main_from* as they are not required for TensorStore process
        prep_data = spawn.get_preparation_data(process_obj._name)
        prep_data.pop("init_main_from_module", None)
        prep_data.pop("init_main_from_path", None)

        fp = io.BytesIO()
        context.set_spawning_popen(self)
        try:
            context.reduction.dump(prep_data, fp)  # pytype: disable=module-attr
            context.reduction.dump(process_obj, fp)  # pytype: disable=module-attr
        finally:
            context.set_spawning_popen(None)

        parent_r = child_w = child_r = parent_w = None
        try:
            parent_r, child_w = os.pipe()
            child_r, parent_w = os.pipe()
            cmd = spawn.get_command_line(tracker_fd=tracker_fd, pipe_handle=child_r)
            self._fds.extend([child_r, child_w])  # pytype: disable=attribute-error
            self.pid = util.spawnv_passfds(
                spawn.get_executable(), cmd, self._fds  # pytype: disable=attribute-error,wrong-arg-types
            )
            self.sentinel = parent_r
            with open(parent_w, "wb", closefd=False) as f:
                f.write(fp.getbuffer())
        finally:
            fds_to_close = []
            for fd in (parent_r, parent_w):
                if fd is not None:
                    fds_to_close.append(fd)
            self.finalizer = util.Finalize(self, util.close_fds, fds_to_close)  # pytype: disable=module-attr

            for fd in (child_r, child_w):
                if fd is not None:
                    os.close(fd)


class _SpawnProcess(multiprocessing.process.BaseProcess):
    _start_method = "spawn"

    @staticmethod
    def _Popen(process_obj):  # noqa N802
        return _Popen(process_obj)


class _SpawnContext(multiprocessing.context.BaseContext):
    _name = "spawn"
    Process = _SpawnProcess


class TensorStore:
    """Tensor store for storing and retrieving numpy arrays in/from shared memory."""

    _SOCKET_EXISTANCE_CHECK_INTERVAL_S = 0.1
    _instances = {}

    def __new__(cls, *args, **kwargs):
        """Create TensorStore object. If object with given address already exists, return it."""
        if args:
            address = args[0]
        elif "address" in kwargs:
            address = kwargs["address"]
        else:
            raise TypeError("TensorStore() missing 1 required positional argument: 'address'")

        address = address.as_posix() if isinstance(address, pathlib.Path) else address

        if address not in cls._instances:
            cls._instances[address] = super().__new__(cls)

        return cls._instances[address]

    def __init__(self, address: Union[str, pathlib.Path], auth_key: Optional[bytes] = None):
        """Initialize TensorStore object.

        Args:
            address: address of data store
            auth_key: authentication key required to setup connection. If not provided, current process authkey will be used
        """
        if not hasattr(self, "_remote_blocks_store_manager"):
            address = address.as_posix() if isinstance(address, pathlib.Path) else address
            self._remote_blocks_store_manager = BlocksStoreManager(address, authkey=auth_key, ctx=_SpawnContext())
            self._remote_blocks_store = None
            self._manager_start_stop_filelock = _FileLock(f"{address}.lock")

            # container for keeping map between tensor_id and numpy array weak ref
            self._handled_blocks: Dict[str, weakref.ReferenceType] = {}
            self._handled_blocks_lock = threading.RLock()

            self._shm_segments: Dict[str, multiprocessing.shared_memory.SharedMemory] = {}
            self._shm_segments_lock = threading.RLock()

            self.serialize = serialize_numpy_with_struct_header
            self.deserialize = deserialize_numpy_with_struct_header
            self._calc_serialized_tensor_size = calc_serialized_size_of_numpy_with_struct_header

    @property
    def address(self) -> str:
        """Return address of remote block store."""
        return self._remote_blocks_store_manager.address

    def start(self):
        """Start remote block store."""
        with self._manager_start_stop_filelock:
            if self._remote_blocks_store is not None:
                raise RuntimeError("Remote block store is already started/connected")

            self._remote_blocks_store_manager.start()
            self._remote_blocks_store = self._remote_blocks_store_manager.blocks()  # pytype: disable=attribute-error

            address = pathlib.Path(self._remote_blocks_store_manager.address)
            self._wait_for_address(address)
            LOGGER.debug(
                f"Started remote block store at {address} (pid={self._remote_blocks_store_manager._process.pid})"  # pytype: disable=attribute-error
            )

    def connect(self, timeout_s: Optional[float] = None):
        """Connect to remote block store."""
        if self._remote_blocks_store is None:
            address = pathlib.Path(self._remote_blocks_store_manager.address)

            self._wait_for_address(address, timeout_s)
            self._remote_blocks_store_manager.connect()
            self._remote_blocks_store = self._remote_blocks_store_manager.blocks()  # pytype: disable=attribute-error
            LOGGER.debug(f"Connected to remote block store at {address})")
        else:
            LOGGER.debug(f"Already connectd to remote block store at {self.address}")

    def _wait_for_address(self, address, timeout_s: Optional[float] = None):
        should_stop_at = time.time() + timeout_s if timeout_s is not None else None
        if timeout_s is not None and self._SOCKET_EXISTANCE_CHECK_INTERVAL_S > timeout_s:
            socket_existance_check_interval = timeout_s
        else:
            socket_existance_check_interval = self._SOCKET_EXISTANCE_CHECK_INTERVAL_S

        while not address.exists():
            if should_stop_at is not None and time.time() >= should_stop_at:
                raise TimeoutError(f"Timeout while waiting for {address} to be created")
            time.sleep(socket_existance_check_interval)

    def _calc_serialized_size(self, tensor: np.ndarray) -> int:
        # frames payload sum + total size + frames sizes
        # assume 2 frames: header with tensor description + data
        return sum(self._calc_serialized_tensor_size(tensor)) + struct.calcsize("<I") + 2 * struct.calcsize("<I")

    def put(self, tensors: Sequence[np.ndarray]) -> Sequence[str]:
        """Append tensor to shared memory buffer.

        Args:
            tensors: numpy arrays to store

        Returns:
            List of ids of stored tensors
        """
        byte_size_of_frames_containers = [self._calc_serialized_size(tensor) for tensor in tensors]
        tensors_ids = self._remote_blocks_store.get_free_blocks(byte_size_of_frames_containers)
        blocks = [BlockDescriptor.from_id(tensor_id) for tensor_id in tensors_ids]

        for tensor, block in zip(tensors, blocks):
            with self._shm_segments_lock:
                shm = self._shm_segments.get(block.shm_name)
                if shm is None:
                    shm = multiprocessing.shared_memory.SharedMemory(block.shm_name, create=False)
                    self._shm_segments[block.shm_name] = shm

            frames = self.serialize(tensor)
            self._copy_frames(frames, shm, block.offset)

        return tensors_ids

    def get(self, tensor_id: str) -> np.ndarray:
        """Get numpy array from tensor store.

        Args:
            tensor_id: id of of tenosr to get

        Returns:
            numpy array
        """
        tensor = None
        # try to handle already handled tensor from weakref
        with self._handled_blocks_lock:
            tensor_ref = self._handled_blocks.get(tensor_id)
            if tensor_ref is not None:
                tensor = tensor_ref()

        if tensor is None:  # if tensor was not handled yet or weakref is already empty
            block = BlockDescriptor.from_id(tensor_id)

            # check if shm segment is already opened
            with self._shm_segments_lock:
                shm = self._shm_segments.get(block.shm_name)

            # if not open it and put into cache
            if shm is None:
                shm = multiprocessing.shared_memory.SharedMemory(block.shm_name, create=False)
                with self._shm_segments_lock:
                    shm = self._shm_segments.setdefault(block.shm_name, shm)  # in meantime other thread could create it

            frames = self._handle_frames(shm, block.offset)
            tensor = self.deserialize(frames)

            # store tensor in weakref to be able to release shared memory when tensor will be garbage collected
            with self._handled_blocks_lock:
                tensor_ref = self._handled_blocks.setdefault(tensor_id, weakref.ref(tensor))
                tensor = tensor_ref()

        return tensor  # pytype: disable=bad-return-type

    def release_block(self, tensor_id: str):
        """Release shared memory block.

        Args:
            tensor_id: id of tensor to release
        """
        LOGGER.debug(f"Releasing shared memory block for tensor {tensor_id}")

        tensor_ref = None
        with self._handled_blocks_lock:
            tensor_ref = self._handled_blocks.pop(tensor_id, None)

        try:
            if tensor_ref is not None:
                self._remote_blocks_store.release_block(tensor_id)
        except OSError:  # thrown when remote process is already closed
            LOGGER.warning(
                f"Failed to release block {tensor_id} on remote process at {self.address}. Probably remote process is already closed"
            )

    def _copy_frames(
        self,
        frames: List[Union[bytes, memoryview]],
        shm: multiprocessing.shared_memory.SharedMemory,
        offset: int,
    ) -> int:
        total_size = struct.calcsize("<I")  # start after total_size; max 4GB for all frames
        for frame in frames:
            if isinstance(frame, bytes):
                frame = memoryview(frame)

            assert frame.contiguous, "Only contiguous arrays are supported"
            struct.pack_into("<I", shm.buf, offset + total_size, frame.nbytes)  # pytype: disable=wrong-arg-types
            total_size += struct.calcsize("<I")
            shm.buf[offset + total_size : offset + total_size + frame.nbytes] = frame.cast("B")

            total_size += frame.nbytes

        struct.pack_into("<I", shm.buf, offset, total_size)  # pytype: disable=wrong-arg-types
        return total_size

    def _handle_frames(self, shm: multiprocessing.shared_memory.SharedMemory, block_offset: int) -> List[memoryview]:
        frames = []
        (total_size,) = struct.unpack_from("<I", shm.buf, block_offset)  # pytype: disable=wrong-arg-types
        offset = struct.calcsize("<I")
        while offset < total_size:
            (frame_size,) = struct.unpack_from("<I", shm.buf, block_offset + offset)  # pytype: disable=wrong-arg-types
            offset += struct.calcsize("<I")
            frame = shm.buf[block_offset + offset : block_offset + offset + frame_size]
            offset += frame_size
            frames.append(frame)
        return frames

    def close(self):
        """Free resources used by TensorStore object."""
        from multiprocessing.resource_tracker import register, unregister

        started_server = hasattr(self._remote_blocks_store_manager, "shutdown")
        LOGGER.debug(f"TensorStore is being closed (started_server={started_server})")

        gc.collect()
        with self._handled_blocks_lock:
            tensors_ids = list(self._handled_blocks)
            for tensor_id in tensors_ids:
                self.release_block(tensor_id)

        with self._shm_segments_lock:
            while self._shm_segments:
                _, shm = self._shm_segments.popitem()
                LOGGER.debug(f"Closing shared memory {shm.name}")
                try:
                    shm.close()
                except Exception as e:
                    LOGGER.warning(f"Failed to close shared memory {shm.name}: {e}")
                finally:
                    if not started_server:
                        register(shm._name, "shared_memory")  # pytype: disable=attribute-error
                        unregister(shm._name, "shared_memory")  # pytype: disable=attribute-error

        if started_server:
            if self._remote_blocks_store is not None:
                LOGGER.debug(f"Releasing all resources on remote process at {self.address}")
                try:
                    self._remote_blocks_store.close()
                except FileNotFoundError:  # thrown when remote process is already closed
                    pass
            self._remote_blocks_store = None
            LOGGER.debug(f"Shutting down side process of data store at {self.address}")
            self._remote_blocks_store_manager.shutdown()
        LOGGER.debug(f"TensorStore at {self.address} closed")
