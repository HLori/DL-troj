import os
import numpy as np
import skimage.io
import torch


def load_image(file_name):
    """return NCHW, BGR, 3*224*224, Tensor type image"""
    img = skimage.io.imread(file_name)  # read in RGB
    img = img[:, :, [2, 1, 0]]  # convert RGB to BGR

    # perform center crop to what the CNN is expecting 224x224
    h, w, c = img.shape
    dx = int((w - 224) / 2)
    dy = int((w - 224) / 2)
    img = img[dy:dy + 224, dx:dx + 224, :]

    img = np.transpose(img, (2, 0, 1)) / 255.0  # transpose to CHW and convert to [0, 1]
    img = np.expand_dims(img, 0)  # transpose to NCHW
    img = torch.Tensor(img)
    return img


def load_data(root_dir, thres=10, batch=64):
    """
    load images from root_dir
    thres: #images each class
    """
    files = os.listdir(root_dir)
    flag = False  # flag for new batch
    count = {}  # count for images of each label
    images_all = []
    labels_all = []
    batch_count = 0  # count total number

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

        img = load_image(os.path.join(root_dir, filename))

        if batch_count % batch == 0:
            flag = False

        if not flag:  # a new batch
            images = img
            labels = [label]
            flag = True
            images_all.append(images)
            labels_all.append(labels)
        else:
            images = images_all[-1]
            images = torch.cat((images, img), dim=0)
            labels = labels_all[-1]
            labels.append(label)
            images_all[-1] = images
            labels_all[-1] = labels
        batch_count += 1
    num_class = len(count)
    return images_all, labels_all, batch_count, num_class


def data_split(images_all, labels_all, target_class, num_class):
    """
    find images belong and not belong to target class
    :param images_all: list of batch images
    :param labels_all: list of batch labels
    :param target_class: target class
    :param num_class: class number
    :return: imgs for non-target-class, imgs for target-class
    """
    outer_number = len(images_all)
    size1 = []
    size2 = []
    imgs = []
    imgs2 = []
    labs = []
    labs2 = []

    for i in range(outer_number):
        images = images_all[i]
        labels = labels_all[i]

        indices = np.where(np.not_equal(labels, target_class))
        indices2 = np.where(np.equal(labels, target_class))
        labels_new = np.zeros((len(labels), num_class))
        labels_new[np.arange(len(labels)), labels] = 1

        if len(indices[0]) != 0:
            size1.append(len(indices[0]))
            imgs.append(images[indices])
            labs.append(labels_new[indices])

        if len(indices2[0]) != 0:
            size2.append(len(indices2[0]))
            imgs2.append(images[indices2])
            labs2.append(labels_new[indices2])
    return imgs, imgs2, labs, labs2, size1, size2


def get_similarity(a, b, types='cos'):
    similarity = 0
    if types == 'cos':
        similarity = np.dot(a, np.transpose(b))/\
                     (np.linalg.norm(a)*np.linalg.norm(b))
    return similarity


def cal_result(outputs_list_p, outputs_list, num_class):
    similarity_model = []
    qts = []

    # calculate similarities and qt
    for cls in range(num_class):
        cos = []
        outputs_p = outputs_list_p[cls]
        outputs = outputs_list[cls]
        for i in range(outputs.shape[0]):
            s = get_similarity(outputs_p[i], outputs[i])
            cos.append(s)
        qt = np.quantile(cos, 0.25)
        similarity_model.append(cos)
        qts.append(qt)

    # calculate whether the model is trojan
    flag = 0
    for cls in range(num_class):
        qt = qts[cls]
        cos = sorted(similarity_model[cls])
        count = 0
        for s in cos:
            count += 1 if s < min(qt, 0.4) else 0
        if count >= 3:
            flag = 1
    return flag








