import matplotlib.pyplot as plt
import os
import json


result_path = './results-2'

# store a prob_dics
prob_dics = [{}, {}, {}, {}]  # for scale=1.05, 1.1, 1.2, 1.3
scale_idx = {'1.05': 0, '1.1': 1, '1.2': 2, '1.3': 3}
for file in os.listdir(result_path):
    model_id = int(file[3:11])
    with open(os.path.join(result_path, file)) as f:
        for line in f.readlines():
            probs = line.split('\t')
            if probs[0] == 'scale':
                continue  # skip the first line
            _, scale, label, qt_25, qt_50, qt_75 = probs

            idx = scale_idx[scale]
            if label == '0':  # if the first label
                prob_dics[idx][model_id] = [[eval(qt_25)], [eval(qt_50)], [eval(qt_75)]]
            else:
                prob_dics[idx][model_id][0].append(eval(qt_25))
                prob_dics[idx][model_id][1].append(eval(qt_50))
                prob_dics[idx][model_id][2].append(eval(qt_75))

# store a dict for model infomation
info_path = './round2'
target_dict = {}

for folder in os.listdir(info_path):
    if folder[0:2] != 'id':
        continue
    model_id = int(folder[3:11])
    folder = os.path.join(info_path, folder)
    jpath = os.path.join(folder, 'config.json')
    with open(jpath, 'r') as f:
        j = json.load(f)
    target_dict[model_id] = j['TRIGGER_TARGET_CLASS']
# print(target_dict)

clean = [[], [], [], []]
trojan = [[], [], [], []]
scales = [1.05, 1.1, 1.2, 1.3]

for idx in range(4):
    prob_dict = prob_dics[idx]
    scale = scales[idx]
    probs_25, probs_50, probs_75 = [], [], []
    probs_25_clean, probs_50_clean, probs_75_clean = [], [], []

    for (model_id, probs) in prob_dict.items():
        prob_25, prob_50, prob_75 = probs
        target = target_dict[model_id]

        if target is not None:  # for trojan model
            probs_25 += prob_25
            probs_50 += prob_50
            probs_75 += prob_75
        else:
            probs_25_clean += prob_25
            probs_50_clean += prob_50
            probs_75_clean += prob_75
    plt.figure(idx, figsize=(15, 11))
    plt.subplot(2, 3, 1)
    plt.title('scale: {0}'.format(scales[idx]))
    plt.hist(probs_25, bins=30, density=False, facecolor='blue', edgecolor='black')
    plt.xlabel('probs_25_troj')
    plt.ylabel('times')

    plt.subplot(2, 3, 2)
    plt.hist(probs_50, bins=30, density=False, facecolor='blue', edgecolor='black')
    plt.xlabel('probs_50_troj')
    plt.ylabel('times')

    plt.subplot(2, 3, 3)
    plt.hist(probs_75, bins=30, density=False, facecolor='blue', edgecolor='black')
    plt.xlabel('probs_75_troj')
    plt.ylabel('times')

    plt.subplot(2, 3, 4)
    plt.hist(probs_25_clean, bins=30, density=False, facecolor='blue', edgecolor='black')
    plt.xlabel('probs_25_clean')
    plt.ylabel('times')

    plt.subplot(2, 3, 5)
    plt.hist(probs_50_clean, bins=30, density=False, facecolor='blue', edgecolor='black')
    plt.xlabel('probs_50_clean')
    plt.ylabel('times')

    plt.subplot(2, 3, 6)
    plt.hist(probs_75_clean, bins=30, density=False, facecolor='blue', edgecolor='black')
    plt.xlabel('probs_75_clean')
    plt.ylabel('times')
    plt.savefig('fig_{0}.png'.format(idx))







