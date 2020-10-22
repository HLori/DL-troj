import numpy as np
import os
import torch
from PerImagePerturb_torch import PerImgPert
from UniversalPerturb_torch import UniversalPert
from utils import load_data, data_split, cal_result
import argparse
import json


def main(model_dir, result_path, test_dir, save_path=None, device='cuda:0', debug=False):
    """
    model_dir <--- model_filepath
    result_path <--- result_filepath
    test_dir <---- examples_dirpath
    """
    # some parameters
    REGU = "l1"  # l1 or l2, regularization of mask loss
    rate = 0.15  # final rate for threshold
    scale = 1.1  # 1.1 * bestarea

    data_thres = 5
    batch = 64  # batch size for input
    data_shuffle, labels_shuffle, batch_size_all, num_class = \
        load_data(test_dir, data_thres, batch)

    if device is not None:
        device = torch.device(device)

    bestarea_list_p = []  # universal: _p
    outputs_list_p = []   # for all classes: _list
    outputs_list = []
    similarities_best = []
    jdict = {}

    # main part
    for targets in range(0, num_class):
        # find images belonging to the selected label and images that are not
        imgs, imgs2, labs, labs2, size1, size2 = \
            data_split(data_shuffle, labels_shuffle, targets, num_class)
        size1_all = sum(size1)

        # print current information:
        print("-------------------------------------------------------------")
        print("- current target label: ", targets)
        print("- regularization form: ", REGU)

        # create path for saving images
        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path_i = os.path.join(save_path, 'target_{}.jpg'.format(targets))
        else:
            save_path_i = None

        # find the Universal Perturbation
        modifier_p, det_p, ind1, ind2, output_p =\
            UniversalPert(model_dir, size1, size2, device)\
                .attack(imgs, imgs2, labs, labs2, save_path_i)

        # find per-image target perturbation
        modifier, det, indic, output = \
            PerImgPert(model_dir, size1, device).attack(imgs, labs)

        print("===========main-information================================================")
        print("Universal Perturbation found in target\n", targets)
        print("Type 1 wrong labels: ", ind1)
        print("Type 2 wrong labels: ", ind2)
        print("Per-image Perturbation found in target\n", targets)
        print("Wrong indices: ", indic)
        print("==============================================================================")

    #     jdict[targets] = {"modifier_p": modifier_p.cpu().detach().numpy().tolist(),
    #              "det_p": det_p.cpu().detach().numpy().tolist(),
    #              "output_p": output_p.tolist(),
    #              # "modifier": modifier.cpu().detach().numpy().tolist(),
    #              # "det": det.cpu().detach().numpy().tolist(),
    #              "output": output.tolist(),
    #              "size_all": batch_size_all}
    # with open(result_path, 'w') as f:
    #     json.dump(jdict, f)
        # bestarea_list_p.append(torch.sum(torch.abs(modifier_p)))
        outputs_list_p.append(output_p)
        outputs_list.append(output)

    # write result
    if not debug:  # write result when proposed
        res = cal_result(outputs_list_p, outputs_list, num_class)
        with open(result_path, 'w') as f:
            f.write("{}".format(res))
    else:  # write for statistics
        with open(result_path, 'w') as f:
            f.write('scale\tlabel\tqt_0.15\tqt_0.25\tqt_0.5\tqt_0.75\n')
            for i in range(num_class):
                qt0 = np.quantile(similarities_best[i], 0.15)
                qt1 = np.quantile(similarities_best[i], 0.25)
                qt2 = np.quantile(similarities_best[i], 0.5)
                qt3 = np.quantile(similarities_best[i], 0.75)
                print("scale: ", scale, "label ", i, qt0, qt1, qt2, qt3)
                f.write("\t{0}\t{1}\t{2}\t{3}\t{4}\n".format(scale, i, qt0, qt1, qt2, qt3))
    print("Model Done")


# for i in range(1042, 1100):
#     model_path = './round2/id-%08d/model.pt' % i
#     result = './result-data/id-%08d.json' % i
#     data_path = './round2/id-%08d/example_data' % i
#     # trigger_path = './reverse_trigger-2/id-%08d' % i
#     main(model_path, result, data_path, save_path=None, device='cuda:0', debug=True)
# main('./round2/id-00001003/model.pt', 'test.txt', './round2/id-00001003/example_data', debug=False)