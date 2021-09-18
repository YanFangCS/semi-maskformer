import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def ap_cal(pred_logits, gt_onehot):
    idx_des = np.argsort(-pred_logits)
    pred_logits, gt_onehot = np.array(pred_logits[idx_des]), np.array(gt_onehot[idx_des])
    tp = []
    # cal tps
    for i in range(len(pred_logits)):
        if pred_logits[i] > 0.49999:
            if gt_onehot[i] == 1:
                tp.append(1)
            else:
                tp.append(0)

    tp = np.array(tp)
    tpc = tp.cumsum()
    fpc = (1 - tp).cumsum()
    gt_num = gt_onehot.sum()

    recall = []
    prec = []

    # precison-recall curve
    recall_curve = tpc / (gt_num + 1e-16)
    recall.append(recall_curve[-1])

    precision_curve = tpc / (tpc + fpc)
    prec.append(precision_curve[-1])

    # ap calculate
    mrec = np.concatenate(([0.0], recall_curve, [1.0]))
    mprec = np.concatenate(([0.0], precision_curve, [0.0]))

    for i in range(mprec.size - 1, 0, -1):
        mprec[i-1] = np.maximum(mprec[i-1], mprec[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mprec[i+1]) 

    return ap

classid2name = {
    0 : "background",
    1 : "aeroplane",
    2 : "bicycle",
    3 : "bird",
    4 : "boat",
    5 : "bottle",
    6 : "bus", 
    7 : "car", 
    8 : "cat", 
    9 : "chair", 
    10 : "cow", 
    11 : "dining table", 
    12 : "dog", 
    13 : "horse", 
    14 : "motorbike", 
    15 : "person", 
    16 : "potted plant", 
    17 : "sheep", 
    18 : "sofa", 
    19 : "train", 
    20 : "tv monitor"
}

if __name__ == "__main__":
    pred_file = "results.txt"
    gt_file = "results_gt.txt"
    
    pred = np.loadtxt(pred_file)
    gt = np.loadtxt(gt_file)

    results = []
    
    with open("ap_results.txt", "w") as f:
        for i in range(21):
            pred_i = pred[:,i]
            gt_i = gt[:,i]
            max_logits = np.max(pred_i)
            pred_i = pred_i / max_logits
            ap_i = ap_cal(pred_i, gt_i)
            results.append(ap_i)
            print("%-12s" % classid2name[i],"\t", "%.4f"%ap_i, file = f)
        
        map = np.mean(results)
        print("map", map)
        print("%-12s" % "map", "\t","%.4f"%map, file = f)
    