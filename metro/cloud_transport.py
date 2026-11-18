"""One reusable shared-memory slot for PointCloud2 data (one worker in flight).

Small metadata travels through the process queue. The slot may only be reused
after the previous result arrives or its worker has been stopped.
"""
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from types import SimpleNamespace


@dataclass
class CloudReference:
    name: str
    size: int
    metadata: dict


class CloudSlot:
    def __init__(self, max_bytes=64*1024*1024):
        if max_bytes <= 0:
            raise ValueError('max_cloud_bytes must be positive')
        self.memory = None
        self.max_bytes = max_bytes

    def pack(self, msg):
        data = memoryview(msg.data).cast('B')
        size = len(data)
        if size > self.max_bytes:
            raise ValueError('PointCloud2 exceeds shared-memory input limit')
        if self.memory is None or self.memory.size < max(1, size):
            self.close()
            self.memory = SharedMemory(create=True, size=max(1, size))
        self.memory.buf[:size] = data
        stamp = SimpleNamespace(sec=msg.header.stamp.sec, nanosec=msg.header.stamp.nanosec)
        metadata = dict(header=SimpleNamespace(frame_id=msg.header.frame_id, stamp=stamp),
            fields=[SimpleNamespace(name=f.name, offset=f.offset, datatype=f.datatype,
                                    count=getattr(f, 'count', 1)) for f in msg.fields])
        metadata.update({key: getattr(msg, key) for key in
                         ('height', 'width', 'point_step', 'row_step', 'is_bigendian')})
        return CloudReference(self.memory.name, size, metadata)

    def close(self):
        if self.memory is not None:
            self.memory.close()
            self.memory.unlink()
            self.memory = None


def open_cloud(reference):
    memory = SharedMemory(name=reference.name)
    message = SimpleNamespace(**reference.metadata, data=memory.buf[:reference.size])
    return memory, message
