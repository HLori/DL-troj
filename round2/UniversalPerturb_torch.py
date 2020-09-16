import numpy as np
import torch
from torch import optim
from torchvision import transforms
# from tqdm import tqdm, trange

BINARY_SEARCH_STEPS = 1  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 101  # number of iterations to perform gradient descent
# ABORT_EARLY = True       # if we stop improving, abort gradient descent early
LEARNING_RATE = 0.01    # larger values converge faster to less accurate results
INITIAL_CONST = 0.01     # the initial constant lambda to pick as a first guess
IMG_SIZE = 224
CHANNELS = 3


class UniversalPert:
    def __init__(self, filepath, batch_size, batch_size2, device,
                 learning_rate=LEARNING_RATE, binary_search_steps=BINARY_SEARCH_STEPS,
                 max_iterations=MAX_ITERATIONS, initial_const=INITIAL_CONST):
        # filepath: model path
        self.image_size = IMG_SIZE
        self.num_channels = CHANNELS
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        # self.ABORT_EARLY = abort_early
        self.initial_const = initial_const
        self.batch_size = batch_size
        self.batch_size2 = batch_size2
        self.shape = (self.batch_size + self.batch_size2, self.num_channels, self.image_size, self.image_size)
        self.shape_pert = (1, self.num_channels, self.image_size, self.image_size)
        self.model = torch.load(filepath, map_location=device)

        self.device = device
        self.model = torch.nn.DataParallel(self.model)
        # self.model = self.model.to(device)

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

    def attack(self, imgs1, imgs2, labs, labs2, save_path=None):
        """
        imgs: all images include D_k and D_k-
        labs: labels of D_k-
        labs2: labels of D_k
        """
        upper = 6
        lower = 0
        area_best = 1e8
        l1_area = []
        logits = []
        index_const = -1
        stop_flag = -1
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
            if index_const == -1 and outer_step > 0:
                lower = self.initial_const
                self.initial_const = (self.initial_const + upper) / 2
            if index_const == 1:
                upper = self.initial_const
                self.initial_const = (self.initial_const + lower) / 2
                index_const = -1

            opt = optim.Adam([self.modifier, self.det], lr=self.LEARNING_RATE)

            ###########################
            #    perform the attack  #
            #########################
            # for iteration in trange(self.MAX_ITERATIONS, ascii=True, desc='universal epoch'):
            for iteration in range(self.MAX_ITERATIONS):
                # calculate output of model with new input
                newimg1 = self.modifier*self.det + imgs1*(1-self.modifier)
                newimg2 = self.modifier*self.det + imgs2*(1-self.modifier)
                output1 = self.model(newimg1)
                output2 = self.model(newimg2)

                # calculate loss function
                loss1, loss2, num_val = self.cal_loss(output1, output2, labs, labs2)
                lossm = torch.sum(torch.abs(self.modifier))
                loss = loss1 + loss2 + lossm*self.initial_const

                # back propagation
                opt.zero_grad()
                loss.backward()
                opt.step()

                # clamp data
                self.modifier.data = torch.clamp(self.modifier.data, max=1, min=0)
                self.det.data = torch.clamp(self.det.data, max=255, min=0)
                self.modifier.requires_grad = True
                self.det.requires_grad = True

                output = torch.cat((output1, output2))
                if num_val > (self.batch_size + self.batch_size2) * 0.7:
                    index_const = 1
                if num_val > (self.batch_size2+self.batch_size)*0.95:
                    stop_flag = 1
                    break

                # if iteration % 100 == 0:
                #     print("Universal labels: \n", torch.argmax(output, dim=1))
                #     print("\niteration:", iteration, "\nLOSS1:",
                #           loss1, "\nLoss2:", loss2, "\nlossL1: ", lossm)
                #     # print("det: \n", self., "\nmask: \n", self.modifier)

            print("Universal labels: \n", torch.argmax(output, dim=1))
            print("\niteration:", iteration, "\nLOSS1:",
                      loss1, "\nLoss2:", loss2, "\nlossL1: ", lossm)
            print("index_const: ", index_const)
            print("finished universal outer step: ", outer_step)
            print("lambda: ", self.initial_const)
            print("-----------------------------------------------------")

            output = output.cpu().detach().numpy()
            if index_const == 1 or outer_step == 0:
                area_tmp = torch.sum(torch.abs(self.modifier)).cpu().detach().numpy()
                logits.append(output)
                l1_area.append(area_tmp)
                if area_best > area_tmp:
                    best_out = output[:]
                    if area_tmp > 0.5:
                        area_best = area_tmp
            if stop_flag == 1:
                break

        output_arg1 = np.argmax(output[: self.batch_size], axis=1)
        output_arg2 = np.argmax(output[self.batch_size:], axis=1)
        indices1 = np.where(np.equal(output_arg1, np.argmax(labs.cpu().detach().numpy(), axis=1)))
        indices2 = np.where(np.not_equal(output_arg2, np.argmax(labs2.cpu().detach().numpy(), axis=1)))

        if save_path is not None:
            save_img = self.modifier*self.det
            save_img = save_img[0].squeeze().detach().cpu()
            save_img = transforms.ToPILImage()(save_img).convert('RGB')
            save_img.save(save_path)

            save_img2 = self.modifier*self.det + imgs2*(1-self.modifier)
            save_img2 = save_img2[0].squeeze().detach().cpu()
            save_img2 = transforms.ToPILImage()(save_img2).convert('RGB')
            save_path = save_path[:-4] + '_full' + save_path[-4:]
            save_img2.save(save_path)

        return area_best, self.det, indices1, indices2, best_out, logits, l1_area














































