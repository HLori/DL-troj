import numpy as np
import torch
from torch import optim

BINARY_SEARCH_STEPS = 1  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 50  # number of iterations to perform gradient descent
# ABORT_EARLY = True       # if we stop improving, abort gradient descent early
LEARNING_RATE = 0.1  # larger values converge faster to less accurate results
INITIAL_CONST = 1  # the initial constant lambda to pick as a first guess
IMG_SIZE = 224
CHANNELS = 3


class PerImgPert:
    def __init__(self, filepath, size, device,
                 learning_rate=LEARNING_RATE,
                 binary_search_steps=BINARY_SEARCH_STEPS, max_iterations=MAX_ITERATIONS,
                 initial_const=INITIAL_CONST):
        image_size, num_channels= IMG_SIZE, CHANNELS
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        # self.ABORT_EARLY = abort_early
        self.initial_const = initial_const
        self.size = size
        self.device = device
        self.regu = 'l1'
        # initial variables
        self.model = torch.load(filepath)
        self.model = torch.nn.DataParallel(self.model)

    def cal_loss(self, output, labs):
        real = (labs*output).sum(dim=1)
        other = torch.max((1-labs)*output - labs*10000, dim=1)
        loss1d = torch.clamp(other.values - real, min=-10)
        # lossm = torch.sum(torch.abs(self.modifier))
        loss1 = torch.sum(loss1d)
        return loss1

    def attack(self, images, labels):
        for outer in range(len(labels)):
            imgs = images[outer]
            labs = labels[outer]
            imgs = imgs.to(self.device)
            imgs.requires_grad = True
            labs = torch.Tensor(labs).to(self.device)
            labs.requires_grad = True

            modifier = torch.zeros_like(imgs,
                                        dtype=torch.float32, device=self.device, requires_grad=True)
            det = torch.ones_like(imgs,
                                  dtype=torch.float32, device=self.device, requires_grad=True)
            opt = optim.Adam([modifier, det], lr=self.LEARNING_RATE)

            for iteration in range(self.MAX_ITERATIONS):
                # perform the attack
                # opt1 and output
                newimg = modifier * det + imgs * (1-modifier)
                output = self.model(newimg)
                loss1 = self.cal_loss(output, labs)
                lossm = torch.sum(torch.abs(modifier))
                loss = loss1 + lossm
                # minimize loss over modifier and assign modifier <-- sign(m) ....
                opt.zero_grad()
                loss.backward()
                opt.step()

                # clamp data
                modifier.data = torch.clamp(modifier.data, min=0, max=1)
                det.data = torch.clamp(det.data, min=0, max=255)

                indices = np.where(np.equal(torch.max(output, dim=1).values.cpu().detach().numpy(),
                                             np.argmax(labs.cpu().detach().numpy(), axis=1)))[0]
                if len(indices) == 0:
                    break

            if outer == 0:
                modifier_all = modifier
                det_all = det
                indices_all = indices
                output_all = output.cpu().detach()
            else:
                modifier_all = torch.cat((modifier_all, modifier), dim=0)
                det_all = torch.cat((det_all, det), dim=0)
                indices_all += indices
                output_all = torch.cat((output_all, output.cpu().detach()), dim=0)
        return modifier_all, det_all, indices, output_all.numpy()
