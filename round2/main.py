import numpy as np
import os
import skimage.io
import torch
from PerImagePerturb_torch import PerImgPert
from UniversalPerturb_torch import UniversalPert
import argparse
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'


def load_data(root_dir, thres=10):
    """
    images: torch.Tensor
    labels: numpy.array
    len(labels): int
    """
    files = os.listdir(root_dir)
    flag = False
    count = {}
    # count = [0] * num_class
    for filename in files:
        if not filename.endswith('.png'):
            continue
        label = eval(filename.split('_')[1])
        count_num = count.get(label, 0)
        if count_num < thres:
            count_num += 1
            count[label] = count_num
        else:
            continue

        img = skimage.io.imread(os.path.join(root_dir, filename))  # read in RGB
        img = img[:, :, [2, 1, 0]]  # convert RGB to BGR

        # perform center crop to what the CNN is expecting 224x224
        h, w, c = img.shape
        dx = int((w - 224) / 2)
        dy = int((w - 224) / 2)
        img = img[dy:dy + 224, dx:dx + 224, :]

        img = np.transpose(img, (2, 0, 1)) / 255.0  # transpose to CHW and convert to [0, 1]
        img = np.expand_dims(img, 0)  # transpose to NCHW
        img = torch.Tensor(img)

        if not flag:
            images = img
            labels = [label]
            flag = True
        else:
            images = torch.cat((images, img), dim=0)
            labels.append(label)
        num_class = len(count)
    return images, np.array(labels), len(labels), num_class


def main(model_dir, result_path, test_dir, save_path=None, device='cuda:0', debug=False):
    """model_dir <--- model_filepath
    result_path <--- result_filepath
    test_dir <---- examples_dirpath
    """
    # some parameters
    REGU = "l1"  # l1 or l2, regularization of mask loss
    rate = 0.25  # final rate for threshold
    scale = 1.1  # 1.1 * bestarea
    scales = [1.1, 1.2, 1.3, 1.05]

    data_thres = 5
    # num_class = 5  # for round0 & round1

    data_shuffle, labels_shuffle, batch_size_all, num_class = load_data(test_dir, data_thres)
    if device is not None:
        device = torch.device(device)

    bestarea_list_p = []  # universal: _p
    outputs_list_p = []   # for all classes: _list
    areas_list_p = []    # not one for a class: s
    output_list = []
    similarities_best = []
    # main part
    for targets in range(0, num_class):
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
        area_p, _, ind1, ind2, bestout_p, outputs_p, areas =\
            UniversalPert(model_dir, batchsize, batchsize2, device).attack(imgs, imgs2, labs, labs2, save_path_i)
        # find per-image target perturbation
        _, __, indic, output = PerImgPert(model_dir, batchsize, device).attack(imgs, labs)

        print("===========main-information================================================")
        print("Universal Perturbation found in target\n", targets)
        print("Type 1 wrong labels: ", ind1)
        print("Type 2 wrong labels: ", ind2)
        print("Per-image Perturbation found in target\n", targets)
        print("Wrong indices: ", indic)
        print("==============================================================================")

        outputs_list_p.append(outputs_p)
        bestarea_list_p.append(area_p)
        areas_list_p.append(areas)
        output_list.append(output)

        # saving similarity of best area perturbation & each per-image
        sims_best = []
        for i in range(batchsize):
            sim_best = np.dot(bestout_p[i], np.transpose(output[i])) / \
                       (np.linalg.norm(bestout_p[i]) * np.linalg.norm(output[i]))
            sims_best.append(sim_best)
        similarities_best.append(sims_best)

    best_area = np.amin(bestarea_list_p)

    def cal_similarities(scale):
        simi_tmp = similarities_best[:]
        # for each class, check the existence of  higher similarity
        for idx in range(num_class):
            middle_similarity_best = np.quantile(similarities_best, 0.5)
            outputs_p = outputs_list_p[idx]
            areas_p = areas_list_p[idx]
            output = output_list[idx]

            # maybe several candidates:
            for k in range(len(areas_p)):
                similarities = []
                area_p = areas_p[k]
                output_p = outputs_p[k]

                # if this area is small enough,
                # then calculate similarity for each data output
                if area_p < scale * best_area:
                    for i in range(len(output)):
                        similarity = np.dot(output_p[i], np.transpose(output[i])) / \
                                     np.linalg.norm(output_p[i] * np.linalg.norm(output[i]))
                        similarities.append(similarity)
                    # if |m| is small with high similarity:
                    if np.quantile(similarities, 0.5) > middle_similarity_best:
                        simi_tmp[idx] = similarities[:]
        return simi_tmp

    # write result
    if not debug:  # write result when proposed
        simi_res = cal_similarities(scale)
        with open(result_path, 'w') as f:
            qt = 0
            for i in range(num_class):
                qt_tmp = np.quantile(simi_res[i], rate)
                qt = max(qt, qt_tmp)
            f.write("{}".format(qt))
    else:  # write for statistics
        with open(result_path, 'w') as f:
            f.write('scale\tlabel\tqt_0.25\tqt_0.5\tqt_0.75\n')
            for s in scales:
                simi_res = cal_similarities(s)
                for i in range(num_class):
                    qt1 = np.quantile(simi_res[i], 0.25)
                    qt2 = np.quantile(simi_res[i], 0.5)
                    qt3 = np.quantile(simi_res[i], 0.75)
                    print("scale: ", s, "label ", i, qt1, qt2, qt3)
                    f.write("\t{0}\t{1}\t{2}\t{3}\t{4}\n".format(s, i, qt1, qt2, qt3))
    print("Model Done")


for i in range(1000, 1100):
    model_path = './round2/id-%08d/model.pt' % i
    result = './results-2/id-%08d.txt' % i
    data_path = './round2/id-%08d/example_data' % i
    trigger_path = './reverse_trigger-2/id-%08d' % i
    main(model_path, result, data_path, trigger_path, device='cuda:0', debug=True)
# main('./round2/id-00001000/model.pt', 'res.txt', './round2/id-00001000/example_data', debug=True)
