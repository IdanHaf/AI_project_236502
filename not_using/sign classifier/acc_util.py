import numpy as np


def overlap(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou


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


def compute_classification_accuracy(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_threshold=0.5):
    tp = 0
    total_preds = len(pred_boxes)
    total_gt = len(gt_boxes)

    matched_gt = [False] * total_gt

    # Iterate over predictions and ground truth to find matches
    for pred_idx, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
        best_iou = 0
        best_gt_idx = -1

        # Find the ground truth with the highest IoU for the predicted box
        for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            iou = overlap(pred_box, gt_box)

            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt_idx = gt_idx

        # Check if the best matching ground truth box is a valid match
        if best_gt_idx != -1 and pred_label == gt_labels[best_gt_idx] and not matched_gt[best_gt_idx]:
            tp += 1
            matched_gt[best_gt_idx] = True  # Mark ground truth as matched

    accuracy = tp / total_preds if total_preds > 0 else 0
    return accuracy

