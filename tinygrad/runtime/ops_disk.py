from __future__ import annotations
import os
import mmap
import io
from typing import Optional
from tinygrad.helpers import OSX
from tinygrad.device import Compiled, Allocator
from multiprocessing import shared_memory

class DiskBuffer:
    def __init__(self, device: DiskDevice, size: int, offset=0):
        self.device, self.size, self.offset = device, size, offset
    def __repr__(self): return f"<DiskBuffer size={self.size} offset={self.offset}>"
    def _buf(self) -> memoryview:
        assert self.device.mem is not None, "DiskBuffer wasn't opened"
        return memoryview(self.device.mem)[self.offset:self.offset + self.size]

MAP_LOCKED, MAP_POPULATE = 0, 0
if not OSX:
    MAP_POPULATE = getattr(mmap, "MAP_POPULATE", 0x008000)

class DiskAllocator(Allocator):
    def __init__(self, device: DiskDevice):
        self.device = device
    def _alloc(self, size: int, options):
        self.device._might_open(size)
        return DiskBuffer(self.device, size)
    def _free(self, buf, options):
        self.device._might_close()
    def as_buffer(self, src: DiskBuffer):
        return src._buf()
    def copyin(self, dest: DiskBuffer, src: memoryview):
        dest._buf()[:] = src
    def copyout(self, dest: memoryview, src: DiskBuffer):
        if OSX and hasattr(self.device, 'fd'):
            with io.FileIO(self.device.fd, "a+b", closefd=False) as fo:
                fo.seek(src.offset)
                fo.readinto(dest)
        else:
            dest[:] = src._buf()
    def offset(self, buf: DiskBuffer, size: int, offset: int):
        return DiskBuffer(buf.device, size, offset)

class DiskDevice(Compiled):
    def __init__(self, device: str):
        self.size: Optional[int] = None
        self.count = 0
        super().__init__(device, DiskAllocator(self), None, None, None)
    def _might_open(self, size):
        self.count += 1
        assert self.size is None or size <= self.size, f"can't reopen Disk tensor with larger size, opened with {self.size}, tried to open with {size}"
        if self.size is not None: return
        filename = self.dname[len("disk:"):]
        self.size = size

        if filename.startswith("shm:"):
            self.shm = shared_memory.SharedMemory(name=filename[4:], create=True, size=self.size)
            self.mem = self.shm.buf
        else:
            try:
                flags = os.O_RDWR | os.O_CREAT
                if not OSX and hasattr(os, 'O_DIRECT'):
                    flags |= os.O_DIRECT
                self.fd = os.open(filename, flags)
            except OSError:
                self.fd = os.open(filename, os.O_RDWR | os.O_CREAT)
            if os.fstat(self.fd).st_size < self.size:
                os.ftruncate(self.fd, self.size)
            self.mem = mmap.mmap(self.fd, self.size, access=mmap.ACCESS_WRITE)
        if (hp := getattr(mmap, "MADV_HUGEPAGE", None)) is not None:
            self.mem.madvise(hp)  # type: ignore
    def _might_close(self):
        self.count -= 1
        if self.count == 0:
            if hasattr(self, 'shm'):
                self.shm.close()
                self.shm.unlink()
            elif hasattr(self, 'fd'):
                os.close(self.fd)
            self.size = None
