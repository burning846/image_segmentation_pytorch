import torch
from itertools import product as product
import numpy as np
from math import ceil


class PriorBox(object):
    def __init__(self, min_sizes, steps, clip, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = min_sizes
        self.steps = steps
        self.clip = clip
        self.image_size = image_size
        print(self.steps)
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output

if __name__ == '__main__':
    pior = PriorBox(min_sizes=[[16, 32], [64, 128], [256, 512]], steps=[8, 16, 32], clip=False, image_size=[640, 640])
    output = pior.forward()
    print(output.shape)
    print(output)