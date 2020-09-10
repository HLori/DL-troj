import numpy as np
import os
import cv2
import torch
from PerImagePerturb_torch import PerImgPert
from UniversalPerturb_torch import UniversalPert

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
test_dir = './id-00000018/example_data'
model_dir = './id-00000018/model.pt'
num_class = 5  # number of labels
REGU = "l1"  # l1 or l2, regularization of mask loss
output_dir = "./results/"
device = "cuda:0"
device = torch.device(device)


def load_data(root_dir):
    """
    images: torch.Tensor
    labels: numpy.array
    len(labels): int
    """
    thres = 20
    files = os.listdir(root_dir)
    flag = False
    count = [0] * num_class
    for filename in files:
        if not filename.endswith('.png'):
            continue
        label = eval(filename[6])
        if count[label] < thres:
            count[label] += 1
        else:
            continue
        img = cv2.imread(os.path.join(root_dir, filename))
        img = img.transpose((2, 0, 1)) / 255.0  # transpose to CHW and convert to [0, 1]
        img = torch.Tensor([img])
        if not flag:
            images = img
            labels = [label]
            flag = True
        else:
            images = torch.cat((images, img), dim=0)
            labels.append(label)
    return images, np.array(labels), len(labels)


data_shuffle, labels_shuffle, batch_size_all = load_data(test_dir)

for jj in range(0, 1):
    data_num_labels = []
    out_save_poison = []
    bestarea_cand_pos = []
    area_cand_pos = []
    out_cand_pos = []
    # pos_mul = []
    pos_single = []

    for targets in range(0, num_class):
        regul = REGU
        # find images belonging to the selected label and images that are not
        indices = np.where(np.not_equal(labels_shuffle, targets))
        indices2 = np.where(np.equal(labels_shuffle, targets))

        batchsize = len(indices[0])
        batchsize2 = len(indices2[0])

        labels_shuffle_new = np.zeros((batch_size_all, num_class))
        labels_shuffle_new[np.arange(batch_size_all), labels_shuffle] = 1

        imgs = data_shuffle[indices]
        labs = labels_shuffle_new[indices]
        imgs2 = data_shuffle[indices2]
        labs2 = labels_shuffle_new[indices2]

        images = torch.cat((imgs, imgs2), axis=0)

        # print current information:
        print("-------------------------------------------------------------")
        print("- current target label: ", targets)
        print("- regularization form: ", regul)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # find the Universal Perturbation
        ######################################
        pert_p, coff_p, ind1, ind2, output_p, log_output, area =\
            UniversalPert(model_dir, batchsize, batchsize2, device).attack(imgs, imgs2, labs, labs2)
        print("Universal Perturbation found in target ", targets)
        print("Type 1 wrong labels: ", labs[ind1])
        print("Type 2 wrong labels: ", labs2[ind2])

        bestarea_cand_pos.append(pert_p)
        out_cand_pos.append(log_output)
        area_cand_pos.append(area)

        # find per-image target perturbation
        #########################################
        m, det, indic, output = PerImgPert(model_dir, batchsize, device).attack(imgs, labs)
        print("Per-image Perturbation found in target ", targets)
        print("Wrong indices: ", indic)
        pos_single.append(output)

        temp_out_poison = []
        for i in range(batchsize):
            temp_single = np.dot(output_p[i], np.transpose(output[i])) / \
                          (np.linalg.norm(output_p[i]) * np.linalg.norm(output[i]))
            temp_out_poison.append(temp_single)
        out_save_poison.append(temp_out_poison)

    best_area = np.amin(bestarea_cand_pos)

    # check whether the model is a Trojan model and the target label
    T = 0.7  # a preset threshold
    target = -1
    for i in range(num_class):
        qt = np.quantile(out_save_poison[i], 0.25)
        print("label ", i, qt)
        if qt > T:
            print('The model is a Trojan model and the target label is: {}'.format(i))
            target = i
    if target == -1:
        print('The model is a clean model')





















