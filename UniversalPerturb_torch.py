import numpy as np
import torch
from torch import optim
from tqdm import tqdm, trange

BINARY_SEARCH_STEPS = 1  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 101  # number of iterations to perform gradient descent
# ABORT_EARLY = True       # if we stop improving, abort gradient descent early
LEARNING_RATE = 0.1    # larger values converge faster to less accurate results
INITIAL_CONST = 0.01     # the initial constant lambda to pick as a first guess
IMG_SIZE = 224
CHANNELS = 3
NUM_LABEL = 5


class UniversalPert:
    def __init__(self, filepath, batch_size, batch_size2, device,
                 learning_rate=LEARNING_RATE, binary_search_steps=BINARY_SEARCH_STEPS,
                 max_iterations=MAX_ITERATIONS, initial_const=INITIAL_CONST):
        # filepath: model path
        self.image_size = IMG_SIZE
        self.num_channels = CHANNELS
        self.num_labels = NUM_LABEL
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        # self.ABORT_EARLY = abort_early
        self.initial_const = initial_const
        self.batch_size = batch_size
        self.batch_size2 = batch_size2
        self.device = device
        self.shape = (self.batch_size + self.batch_size2, self.num_channels, self.image_size, self.image_size)
        self.shape_pert = (1, self.num_channels, self.image_size, self.image_size)
        self.model = torch.load(filepath)
        self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.to(device)

        self.modifier = torch.zeros(self.shape_pert, dtype=torch.float32, device=self.device, requires_grad=True)
        self.det = torch.ones(self.shape_pert, dtype=torch.float32, device=self.device, requires_grad=True)

    def cal_loss(self, output1, output2, labs, labs2):
        real = (labs*output1).sum(dim=1)
        other = torch.max((1-labs)*output1-labs*10000, dim=1)
        real2 = (labs2*output2).sum(dim=1)
        other2 = torch.max((1-labs2)*output2 - labs2*10000, dim=1)
        loss1d = torch.clamp(real - other.values, min=-10)
        loss1 = torch.sum(loss1d)
        loss1d2 = torch.clamp(other2.values - real2, min=-10)
        loss12 = torch.sum(loss1d2)
        num_val = sum([1 for val in loss1d if val <= 0]) + sum([1 for val in loss1d2 if val <= 0])
        return loss1, loss12, num_val

    def attack(self, imgs1, imgs2, labs, labs2):
        """
        imgs: all images include D_k and D_k-
        labs: labels of D_k-
        labs2: labels of D_k
        """
        batch_size = self.batch_size
        batch_size2 = self.batch_size2
        batch_size_full = batch_size + batch_size2
        upper = 6
        lower = 0
        area_best = 1e8
        l1_area = []
        logits = []
        index_const = -1

        # to device
        imgs1 = imgs1.to(self.device)
        imgs2 = imgs2.to(self.device)
        labs = torch.Tensor(labs)
        labs2 = torch.Tensor(labs2)
        labs = labs.to(self.device)
        labs2 = labs2.to(self.device)
        imgs1.requires_grad = True
        imgs2.requires_grad = True
        labs.requires_grad = True
        labs2.requires_grad = True

        for outer_step in range(self.BINARY_SEARCH_STEPS):
            # initial const increase with outer_step.
            CONST = np.ones(1) * self.initial_const
            if index_const == -1 and outer_step > 0:
                lower = self.initial_const
                self.initial_const = (self.initial_const + upper) / 2
                CONST = np.ones(1) * self.initial_const
            if index_const == 1:
                upper = self.initial_const
                self.initial_const = (self.initial_const + lower) / 2
                index_const = -1
                CONST = np.ones(1) * self.initial_const

            opt1 = optim.Adam([self.modifier], lr=self.LEARNING_RATE)
            opt2 = optim.Adam([self.det], lr=self.LEARNING_RATE)

            ###########################
            #    perform the attack  #
            #########################
            for iteration in trange(self.MAX_ITERATIONS, ascii=True, desc='universal epoch'):
                # update train1, modifier
                newimg1 = self.modifier * self.det + imgs1 * (1 - self.modifier)
                newimg2 = self.modifier * self.det + imgs2 * (1 - self.modifier)
                output1 = self.model(newimg1)
                output2 = self.model(newimg2)

                # calculate loss function
                loss1, loss2, _ = self.cal_loss(output1, output2, labs, labs2)
                lossm = torch.sum(torch.abs(self.modifier))
                loss = loss1 + loss2 + lossm/self.initial_const

                opt1.zero_grad()
                loss.backward()
                opt1.step()

                # update train2, det
                newimg1 = self.modifier * self.det + imgs1 * (1 - self.modifier)
                newimg2 = self.modifier * self.det + imgs2 * (1 - self.modifier)
                output1 = self.model(newimg1)
                output2 = self.model(newimg2)

                loss1, loss2, num_val = self.cal_loss(output1, output2, labs, labs2)
                lossm = torch.sum(torch.abs(self.modifier))
                loss = loss1 + loss2 + lossm

                opt2.zero_grad()
                loss.backward()
                opt2.step()

                # with torch.no_grad():
                #     self.det = torch.clamp(self.det, min=0, max=255)

                output = torch.cat((output1, output2))
                if num_val > batch_size_full * 0.7:
                    index_const = 1

                if iteration % (self.MAX_ITERATIONS//10) == 0:
                    print("iteration:", iteration, "\nLOSS1:\n", loss1, "\nLoss2\n", loss2, "lossL1: ", lossm)
                    # print("det: \n", self., "\nmask: \n", self.modifier)

                if iteration % 50 == 0:
                    print("Universal labels: \n", torch.argmax(output, dim=1))

            print("index_const: ", index_const)
            print("inital_const: ", CONST)

            output = output.cpu().detach().numpy()
            if index_const == 1 or outer_step == 0:
                area_tmp = torch.sum(torch.abs(self.modifier)).cpu().detach().numpy()
                logits.append(output)
                l1_area.append(area_tmp)
                if area_best > area_tmp:
                    best_out = output[:]
                    print("l1 norm of perturbation: ", area_tmp)
                    if area_tmp > 0.5:
                        area_best = area_tmp

        output_arg1 = np.argmax(output[: batch_size], axis=1)
        output_arg2 = np.argmax(output[batch_size:], axis=1)
        indices1 = np.where(np.equal(output_arg1, np.argmax(labs.cpu().detach().numpy(), axis=1)))
        indices2 = np.where(np.not_equal(output_arg2, np.argmax(labs2.cpu().detach().numpy(), axis=1)))

        return area_best, self.det, indices1, indices2, best_out, logits, l1_area













































