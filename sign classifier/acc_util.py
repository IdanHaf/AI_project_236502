import numpy as np


def overlap(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    return (x2 - x1) * (y2 - y1)


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def evaluate_bbox_accuracy(predicted_labels, true_labels, predicted_bboxes, true_bboxes):
    mapping = {}
    for i, bbox in enumerate(predicted_bboxes):
        max_overlap_idx = -1
        max_overlap = 0
        for j, true_bbox in enumerate(true_bboxes):
            if j not in mapping.values():
                if overlap(true_bbox, bbox) > max_overlap:
                    max_overlap = overlap(true_bbox, bbox)
                    max_overlap_idx = j
        mapping[i] = max_overlap_idx

    # In the mapping we attempted to give each predicted box the closest true box in greedy way
    # Now we want to estimate how many of them are correct
    unique_bboxes = len(true_bboxes) + len(predicted_bboxes) - len(mapping.keys())

    correct_labels = 0
    # We want to estimate how fitting is the predicted bounding box
    correct_bboxes = []
    for i, j in mapping.items():
        correct_labels += 1 if true_labels[j] == predicted_labels[i] else 0
        o = overlap(true_bboxes[j], predicted_bboxes[i])
        correct_bboxes.append(o / (area(predicted_bboxes[i]) + area(true_bboxes[j]) - o))

    return correct_labels / unique_bboxes, np.mean(np.array(correct_bboxes))
