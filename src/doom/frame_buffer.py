import numpy


class FrameBuffer:

    def __init__(self, width, height, depth, frames):
        self.width  = width
        self.height = height
        self.depth  = depth
        self.frames = frames

        self.clear()

    def clear(self):
        size = self.width*self.height*self.depth

        self.buffer = []
        for i in range(0, self.frames):
            self.buffer.append(numpy.zeros(size))


    def add_item(self, value):
        for i in range(0, self.frames-1):
            idx = self.frames - 1 - i
            self.buffer[idx] = self.buffer[idx-1].copy()

        self.buffer[0] = value.copy()

    def get(self, idx):
        return self.buffer[idx]
