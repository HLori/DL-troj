import numpy as np
import torch
from torch import optim
from torchvision import transforms

# from tqdm import tqdm, trange

BINARY_SEARCH_STEPS = 1  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 101  # number of iterations to perform gradient descent
# ABORT_EARLY = True       # if we stop improving, abort gradient descent early
LEARNING_RATE = 0.01  # larger values converge faster to less accurate results
INITIAL_CONST = 0.01  # the initial constant lambda to pick as a first guess
IMG_SIZE = 224
CHANNELS = 3


class UniversalPert:
    def __init__(self, filepath, size1, size2, device,
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
        self.size1 = size1
        self.size2 = size2

        self.shape_pert = (1, self.num_channels, self.image_size, self.image_size)
        self.model = torch.load(filepath)
        self.device = device
        self.model = torch.nn.DataParallel(self.model)
        # self.model = self.model.to(device)

        self.modifier = torch.zeros(self.shape_pert, dtype=torch.float32,
                                    device=self.device, requires_grad=True)
        self.det = torch.ones(self.shape_pert, dtype=torch.float32,
                              device=self.device, requires_grad=True)

    def cal_loss(self, output1, output2, labs, labs2):
        real = (labs * output1).sum(dim=1)
        other = torch.max((1 - labs) * output1 - labs * 10000, dim=1)
        real2 = (labs2 * output2).sum(dim=1)
        other2 = torch.max((1 - labs2) * output2 - labs2 * 10000, dim=1)
        loss1d = torch.clamp(real - other.values, min=-10)
        loss1 = torch.sum(loss1d)
        loss1d2 = torch.clamp(other2.values - real2, min=-10)
        loss12 = torch.sum(loss1d2)
        num_val = sum([1 for val in loss1d if val > 0]) + sum([1 for val in loss1d2 if val > 0])
        return loss1, loss12, num_val

    def attack(self, images1, images2, labels, labels2, save_path=None):
        """
        images: all images include D_k-
        images2:  D_k
        labs: labels of D_k-
        labs2: labels of D_k
        """
        opt = optim.Adam([self.modifier, self.det], lr=self.LEARNING_RATE)

        ###########################
        #    perform the attack  #
        #########################
        for iteration in range(self.MAX_ITERATIONS):
            num_val_all = 0
            for i in range(len(labels)*2):
                idx1 = i % len(self.size1)
                idx2 = i % len(self.size2)
                imgs1 = images1[idx1].to(self.device)
                imgs2 = images2[idx2].to(self.device)
                labs = torch.Tensor(labels[idx1])
                labs2 = torch.Tensor(labels2[idx2])
                labs = labs.to(self.device)
                labs2 = labs2.to(self.device)

                imgs1.requires_grad = True
                imgs2.requires_grad = True
                labs.requires_grad = True
                labs2.requires_grad = True

                # calculate output of model with new input
                newimg1 = self.modifier * self.det + imgs1 * (1 - self.modifier)
                newimg2 = self.modifier * self.det + imgs2 * (1 - self.modifier)
                output1 = self.model(newimg1)
                output2 = self.model(newimg2)
                output = torch.cat((output1, output2), dim=0)

                # calculate loss function
                loss1, loss2, num_val = self.cal_loss(output1, output2, labs, labs2)
                lossm = torch.sum(torch.abs(self.modifier))
                loss = loss1 + loss2 + lossm * self.initial_const
                num_val_all += num_val

                # back propagation
                opt.zero_grad()
                loss.backward()
                opt.step()

                # clamp data
                self.modifier.data = torch.clamp(self.modifier.data, max=1, min=0)
                self.det.data = torch.clamp(self.det.data, max=255, min=0)
                self.modifier.requires_grad = True
                self.det.requires_grad = True

                if i == 0:
                    output_all = output
                    indices1_all = list(np.where(
                        np.equal(np.argmax(output1.cpu().detach().numpy(), axis=1),
                                 np.argmax(labs.cpu().detach().numpy(), axis=1)))[0])
                    indices2_all = list(np.where(
                        np.not_equal(np.argmax(output2.cpu().detach().numpy(), axis=1),
                                     np.argmax(labs2.cpu().detach().numpy(), axis=1)))[0])
                else:
                    output_all = torch.cat((output_all, output), dim=0)
                    # a = np.argmax(output1.cpu().detach().numpy(), axis=1)
                    # print('a-----', a.shape)
                    # b = np.argmax(labs.cpu().detach().numpy(), axis=1)
                    # print('---b', b.shape)
                    # c =  np.equal(a, b)
                    # print('---c', c)
                    # d = np.where(c)
                    # print(d)
                    # d = d[0]
                    # indices1_all += d
                    indices1_all += list(np.where(
                        np.equal(np.argmax(output1.cpu().detach().numpy(), axis=1),
                                 np.argmax(labs.cpu().detach().numpy(), axis=1)))[0])
                    indices2_all += list(np.where(
                        np.not_equal(np.argmax(output2.cpu().detach().numpy(), axis=1),
                                     np.argmax(labs2.cpu().detach().numpy(), axis=1)))[0])

            if (num_val_all < 4 and iteration > 50) or num_val_all < 2:
                print('------break')
                break

        print('Type1 indices: ', indices1_all)
        print('Type2 indices: ', indices2_all)
        print("Universal labels: \n", torch.argmax(output_all, dim=1))
        print("-----------------------------------------------------")

        output = output_all.cpu().detach().numpy()
        if save_path is not None:
            save_img = self.modifier * self.det
            save_img = save_img[0].squeeze().detach().cpu()
            save_img = transforms.ToPILImage()(save_img).convert('RGB')
            save_img.save(save_path)

            save_img2 = self.modifier * self.det + imgs2 * (1 - self.modifier)
            save_img2 = save_img2[0].squeeze().detach().cpu()
            save_img2 = transforms.ToPILImage()(save_img2).convert('RGB')
            save_path = save_path[:-4] + '_full' + save_path[-4:]
            save_img2.save(save_path)

        return self.modifier, self.det, indices1_all, indices2_all, output
