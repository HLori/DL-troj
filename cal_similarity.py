import matplotlib.pyplot as plt
import os
import numpy as np
import json


def get_output(model_id):
    root = './result-data'
    j_path = os.path.join(root, 'id-%08d.json' % model_id)
    outputs_p = []
    outputs = []
    with open(j_path, 'r') as f:
        j_dict = json.load(f)
        for (label, info) in j_dict.items():
            output_p = np.array(info['output_p'])
            output = np.array(info['output'])
            outputs_p.append(output_p)
            outputs.append(output)
    return outputs_p, outputs


def get_similarity(a, b, types='cos'):
    similarity = 0
    if types == 'cos':
        similarity = np.dot(a, np.transpose(b))/(np.linalg.norm(a)*np.linalg.norm(b))
    return similarity


def cal_similarity(model_id):   # model 1041 is missing
    similarity_model = []
    qts = []
    outputs_p, outputs = get_output(model_id)
    for label in range(len(outputs_p)):
        output_p, output = outputs_p[label], outputs[label]
        cos = []
        for i in range(output.shape[0]):   # calculate cosine similarity
            cos_similarity = get_similarity(output_p[i], output[i])
            cos.append(cos_similarity)
        qt = np.quantile(cos, 0.25)   # find 10% cosine similarity
        similarity_model.append(cos)
        qts.append(qt)
    return similarity_model, qts


def ground_truth(model_id):
    j_path = os.path.join('./round2', 'id-%08d' % model_id)
    j_path = os.path.join(j_path, 'config.json')
    with open(j_path) as f:
        config = json.load(f)
        target = config['TRIGGER_TARGET_CLASS']
        triggered = config['TRIGGERED_CLASSES']
    return target, triggered


def set_axis_style(ax, labels):
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels)+1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels)+0.75)
    ax.set_xlabel('Type of similarity')


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def main():
    target_sim = []
    trojan_sim = []
    trojan_clean_sim = []
    triggered_sim = []
    clean_sim = []
    type = 'single'
    acc = 0
    acc2 = 0
    clean_total = 0

    for model_id in range(1000, 1100):
        if model_id == 1041:
            continue
        similarity, qts = cal_similarity(model_id)
        target, triggered = ground_truth(model_id)
        flag = False

        if type == 'all':
            if target is None:
                for sim in similarity:
                    clean_sim += sim
            else:
                target_sim += similarity[target]
                for sim in similarity:
                    trojan_sim += sim
                for idx in triggered:
                    triggered_sim += similarity[idx]
                triggered.append(target)
                for s in range(len(similarity)):
                    if s not in triggered:
                        trojan_clean_sim += similarity[s]
        elif type == 'single':
            for idx in range(len(qts)):
                qt = qts[idx]
                simi = sorted(similarity[idx])
                count = 0
                for s in simi:
                    count += 1 if s < min(qt, 0.4) else 0
                if count >= 3:
                    flag = True
                if target is None:
                    clean_sim.append(count)
                elif target == idx:
                    target_sim.append(count)
                    trojan_sim.append(count)
                elif idx in triggered:
                    triggered_sim.append(count)
                    trojan_sim.append(count)
                else:
                    trojan_sim.append(count)
                    trojan_clean_sim.append(count)
        print('model', model_id, 'done')
        if target is None:
            clean_total += 1
            acc += 1 if flag else 0
        else:
            acc2 += 1 if not flag else 0

    # plot clean, trojan
    fig, ax = plt.subplots()
    ax.set_title('less than 0.4 count')
    ax.set_ylabel('similarity')
    clean_sim = np.clip(clean_sim, 0, 10)
    triggered_sim = np.clip(triggered_sim, 0, 10)
    trojan_clean_sim = np.clip(trojan_clean_sim, 0, 10)
    trojan_sim = np.clip(trojan_sim, 0, 10)
    target_sim = np.clip(target_sim, 0, 10)
    # print(sorted((clean_sim)))
    data = [sorted(clean_sim), sorted(triggered_sim), sorted(trojan_clean_sim), sorted(trojan_sim), sorted(target_sim)]
    for idx in range(len(data)):
        print(idx)
        print(np.median(data[idx]))
        print(np.mean(data[idx]))

    ax.violinplot(data, showmeans=True, showmedians=True, showextrema=True)
    # qt1, qt2, qt3 = np.percentile(data, [25, 50, 75])
    # print(qt1, qt2, qt3)
    # whiskers = np.array([adjacent_values(a, q1, q3) for a, q1, q3 in zip(data, qt1, qt3)])
    # w_min, w_max = whiskers[:, 0], whiskers[:, 1]
    # inds = np.arange(1, len(qt2)+1)
    # ax.scatter(inds, qt2, marker='o', color='white', s=30, zorder=3)
    # ax.vlines(inds, qt1, qt3, color='k', linestyles='-', lw=1)
    # ax.vlines(inds, w_min, w_max, color='k', linestyles='-', lw=1)
    labels = ['clean', 'triggered', 'trojan_clean', 'trojan', 'target']
    set_axis_style(ax, labels)
    plt.savefig('lt40.png')
    print('last acc: ', acc, '/', clean_total, 'acc2: ', acc2)


if __name__ == '__main__':
    main()
