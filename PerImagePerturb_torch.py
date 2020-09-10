import numpy as np
import torch
from torch import optim
from tqdm import tqdm, trange

BINARY_SEARCH_STEPS = 1  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 101  # number of iterations to perform gradient descent
# ABORT_EARLY = True       # if we stop improving, abort gradient descent early
LEARNING_RATE = 0.1  # larger values converge faster to less accurate results
INITIAL_CONST = 1  # the initial constant lambda to pick as a first guess
IMG_SIZE = 224
CHANNELS = 3
NUM_LABEL = 5


class PerImgPert:
    def __init__(self, filepath, batch_size, device,
                 learning_rate=LEARNING_RATE,
                 binary_search_steps=BINARY_SEARCH_STEPS, max_iterations=MAX_ITERATIONS,
                 initial_const=INITIAL_CONST):
        image_size, num_channels, num_labels = IMG_SIZE, CHANNELS, NUM_LABEL
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        # self.ABORT_EARLY = abort_early
        self.initial_const = initial_const
        self.batch_size = batch_size
        self.device = device
        self.regu = 'l1'
        shape_pert = (batch_size, num_channels, image_size, image_size)

        # initial variables
        self.modifier = torch.zeros(shape_pert, dtype=torch.float32, device=self.device, requires_grad=True)
        self.det = torch.ones(shape_pert, dtype=torch.float32, device=self.device, requires_grad=True)   # clip by (0, 255)
        self.model = torch.load(filepath)
        self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.to(device)

    def cal_loss(self, output, labs):
        real = (labs*output).sum(dim=1)
        other = torch.max((1-labs)*output - labs*10000, dim=1)
        loss1d = torch.clamp(other.values - real, min=-10)
        lossm = torch.sum(torch.abs(self.modifier))
        loss1 = torch.sum(loss1d)
        return loss1, lossm

    def attack(self, imgs, labs):
        imgs = imgs.to(self.device)
        imgs.requires_grad = True
        labs = torch.Tensor(labs).to(self.device)
        labs.requires_grad = True
        for outer_step in range(self.BINARY_SEARCH_STEPS):
            opt1 = optim.Adam([self.modifier], lr=self.LEARNING_RATE)
            opt2 = optim.Adam([self.det], lr=self.LEARNING_RATE)

            for iteration in trange(self.MAX_ITERATIONS, ascii=True, desc="Per-Image iteration"):
                # perform the attack
                # opt1 and output
                newimg = self.modifier * self.det + imgs * (1-self.modifier)
                output = self.model(newimg)

                loss1, lossm = self.cal_loss(output, labs)
                loss = loss1 + lossm

                # minimize loss over modifier and assign modifier <-- sign(m) ....
                opt1.zero_grad()
                loss.backward()
                opt1.step()

                # with torch.no_grad():
                #     # self.modifier = torch.clamp(self.modifier, min=0, max=1)
                #     self.modifier =\
                #         torch.clamp(torch.sign(self.modifier) *
                #                     torch.clamp(torch.abs(self.modifier)-self.LEARNING_RATE/self.initial_const, max=0),
                #                     min=0, max=1)

                # minimize loss over delta
                newimg = self.modifier * self.det + imgs*(1-self.modifier)
                output = self.model(newimg)

                loss1, lossm = self.cal_loss(output, labs)
                loss = loss1 + lossm

                # minimize loss over modifier and assign modifier <-- sign(m) ....
                opt2.zero_grad()
                loss.backward()
                opt2.step()
                # with torch.no_grad():
                #     self.det = torch.clamp(self.det, min=0, max=255)

                if iteration % (self.MAX_ITERATIONS//10) == 0:
                    print("iteration: ", iteration, "total loss: ", loss, "loss m : ", lossm)
                    # print("det: ", self.det, "mask: ", self.modifier)
                if iteration % 50 == 0:
                    print("per-image label: \n", torch.argmax(output, dim=1))

        output_arg1 = torch.max(output, dim=1).values.cpu().detach().numpy()
        indices1 = np.where(np.equal(output_arg1, np.argmax(labs.cpu().detach().numpy(), axis=1)))
        # print("indices:", indices1)
        return self.modifier, self.det, indices1, output.cpu().detach().numpy()

