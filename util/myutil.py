import warnings, csv, os, random, torch
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import (hamming_loss, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, matthews_corrcoef,
                             roc_curve, auc, precision_recall_curve, average_precision_score,accuracy_score)
class_16_category_label_map = {
    "AAP": 0,
    "ABP": 1,
    "ACP": 2,
    "AFP": 3,
    "AHP": 4,
    "AIP": 5,
    "ATP": 6,
    "AVP": 7,
    "BP": 8,
    "BBBP": 9,
    "DPP-IV": 10,
    "NP": 11,
    "PSBP": 12,
    "QSP": 13,
    "THP": 14,
    "UP": 15
}
class_5_category_label_map = {
    "AMP": 0,
    "ACP": 1,
    "ADP": 2,
    "AHP": 3,
    "AIP": 4,
}
def save_multi_label_metric(epoch, Y_pred_tensor, Y_test_numpy, saveroot):
    # 初始化新的 Y_pred，用于存放基于最佳阈值的结果
    Y_pred_optimal = np.zeros_like(Y_pred_tensor)
    # 初始化存储每个类别的结果
    metrics_per_class = {}
    # 生成反转的映射，方便通过 label 查找类别名称
    if Y_pred_optimal.shape[1]==16:
        label_category_map = {v: k for k, v in class_16_category_label_map.items()}
    elif Y_pred_optimal.shape[1]==9:
        label_category_map = {v: k for k, v in class_9_category_label_map.items()}
    elif Y_pred_optimal.shape[1]==5:
        label_category_map = {v: k for k, v in class_5_category_label_map.items()}
    

    # 保存用于绘制 ROC 和 PR 曲线的前置数据
    roc_pr_data = []
    
    # 计算每个类别的最佳阈值
    for class_idx in range(len(label_category_map)):
        y_true = Y_test_numpy[:, class_idx]
        y_pred_prob = Y_pred_tensor[:, class_idx].numpy()

        # 计算 ROC 曲线和 PR 曲线
        fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred_prob)
        precision, recall, thresholds_pr = precision_recall_curve(y_true, y_pred_prob)
        
        # 使用 Youden's J statistic 来选择最佳阈值
        J_scores = tpr - fpr
        best_threshold_idx = np.argmax(J_scores)
        best_threshold = thresholds_roc[best_threshold_idx]

        # 或者，使用 F1-score 最大值来选择最佳阈值
        # f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        # best_threshold_idx = np.argmax(f1_scores)
        # best_threshold = thresholds_pr[best_threshold_idx]
        
        # 根据最佳阈值划分为 0 或 1
        # Y_pred_optimal[:, class_idx] = (y_pred_prob >= best_threshold).astype(int)
        # 根据0.5阈值划分为 0 或 1
        Y_pred_optimal[:, class_idx] = (y_pred_prob >= 0.5).astype(int)

        # 使用新的 Y_pred_optimal 进行后续的指标计算
        y_pred = Y_pred_optimal[:, class_idx]

        # 计算混淆矩阵 (TP, TN, FP, FN)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        # Sn (Sensitivity) = Recall = TPR
        Sn = tp / (tp + fn) if (tp + fn) > 0 else 0
        # Sp (Specificity) = TNR
        Sp = tn / (tn + fp) if (tn + fp) > 0 else 0
        # Acc (Accuracy)
        Acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        # MCC (Matthews Correlation Coefficient)
        MCC = matthews_corrcoef(y_true, y_pred) if (tp + tn + fp + fn) > 0 else 0
        # AUC (Area Under Curve)
        try:
            AUC = auc(fpr, tpr)
        except ValueError:
            AUC = float('nan')  # 如果AUC计算失败
        # AUPR (Area Under Precision-Recall curve)
        try:
            AUPR = auc(recall, precision)
        except ValueError:
            AUPR = float('nan')
        
        # 保存每个类别的最佳阈值和相关指标
        class_metrics = {
            'class_name': label_category_map[class_idx],
            'best_threshold': best_threshold,
            'sensitivity_m': Sn,
            'specificity_m': Sp,
            'accuracy_m': Acc,
            'mcc_m': MCC,
            'auc_m': AUC,
            'aupr_m': AUPR
        }

        # 计算基于全零样本的负样本定义
        all_zero_mask = np.all(Y_test_numpy == 0, axis=1)
        pos_mask = y_true == 1
        neg_mask = all_zero_mask
        if np.sum(pos_mask) > 0 and np.sum(neg_mask) > 0:
            y_true_b = np.concatenate([y_true[pos_mask], Y_test_numpy[neg_mask, class_idx]])
            y_pred_b = np.concatenate([y_pred[pos_mask], Y_pred_optimal[neg_mask, class_idx]])
            tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_true_b, y_pred_b).ravel()
            Sn_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0
            Sp_b = tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0
            Acc_b = (tp_b + tn_b) / (tp_b + tn_b + fp_b + fn_b) if (tp_b + tn_b + fp_b + fn_b) > 0 else 0
            MCC_b = matthews_corrcoef(y_true_b, y_pred_b) if (tp_b + tn_b + fp_b + fn_b) > 0 else 0
            try:
                fpr_b, tpr_b, _ = roc_curve(y_true_b, np.concatenate([Y_pred_tensor[pos_mask, class_idx].numpy(),
                                                                      Y_pred_tensor[neg_mask, class_idx].numpy()]))
                AUC_b = auc(fpr_b, tpr_b)
            except ValueError:
                AUC_b = float('nan')
            try:
                precision_b, recall_b, _ = precision_recall_curve(y_true_b, np.concatenate([Y_pred_tensor[pos_mask, class_idx].numpy(),
                                                                                           Y_pred_tensor[neg_mask, class_idx].numpy()]))
                AUPR_b = auc(recall_b, precision_b)
            except ValueError:
                AUPR_b = float('nan')

            class_metrics['sensitivity_b'] = Sn_b
            class_metrics['specificity_b'] = Sp_b
            class_metrics['accuracy_b'] = Acc_b
            class_metrics['mcc_b'] = MCC_b
            class_metrics['auc_b'] = AUC_b
            class_metrics['aupr_b'] = AUPR_b
        else:
            class_metrics.update({
                'sensitivity_b': float('nan'),
                'specificity_b': float('nan'),
                'accuracy_b': float('nan'),
                'mcc_b': float('nan'),
                'auc_b': float('nan'),
                'aupr_b': float('nan'),
            })

        # 保存每个类的指标到CSV或字典
        # binary_header = [
        #     'Epoch',
        #     'sensitivity_m',
        #     'specificity_m',
        #     'accuracy_m',
        #     'mcc_m',
        #     'auc_m',
        #     'aupr_m'
        # ]
        binary_header = [
            'Epoch',
            'sensitivity_b', 'sensitivity_m',
            'specificity_b', 'specificity_m',
            'accuracy_b', 'accuracy_m',
            'mcc_b', 'mcc_m',
            'auc_b', 'auc_m',
            'aupr_b', 'aupr_m'
        ]
        write_to_csv(saveroot + f'/{label_category_map[class_idx]}.csv', binary_header,
                     [epoch,
                      "{:.4f}".format(class_metrics['sensitivity_b']),
                      "{:.4f}".format(class_metrics['sensitivity_m']),
                      "{:.4f}".format(class_metrics['specificity_b']),
                      "{:.4f}".format(class_metrics['specificity_m']),
                      "{:.4f}".format(class_metrics['accuracy_b']),
                      "{:.4f}".format(class_metrics['accuracy_m']),
                      "{:.4f}".format(class_metrics['mcc_b']),
                      "{:.4f}".format(class_metrics['mcc_m']),
                      "{:.4f}".format(class_metrics['auc_b']),
                      "{:.4f}".format(class_metrics['auc_m']),
                      "{:.4f}".format(class_metrics['aupr_b']),
                      "{:.4f}".format(class_metrics['aupr_m'])])
    # **计算整体评估指标：使用基于最佳阈值的 Y_pred_optimal**
    overall_metrics = {
        'Epoch':           epoch,
        'Aiming':          Aiming(Y_pred_optimal, Y_test_numpy),
        'Coverage':        Coverage(Y_pred_optimal, Y_test_numpy),
        'accuracy':        Accuracy(Y_pred_optimal, Y_test_numpy),
        'absolute_true':   AbsoluteTrue(Y_pred_optimal, Y_test_numpy),
        'absolute_false':  AbsoluteFalse(Y_pred_optimal, Y_test_numpy),
        # 'hamming_loss':    hamming_loss(Y_test_numpy, Y_pred_optimal),
        # 'precision_micro': precision_score(Y_test_numpy, Y_pred_optimal, average='micro'),
        # 'precision_macro': precision_score(Y_test_numpy, Y_pred_optimal, average='macro'),
        # 'recall_micro':    recall_score(Y_test_numpy, Y_pred_optimal, average='micro'),
        # 'recall_macro':    recall_score(Y_test_numpy, Y_pred_optimal, average='macro'),
        # 'f1_micro':        f1_score(Y_test_numpy, Y_pred_optimal, average='micro'),
        # 'f1_macro':        f1_score(Y_test_numpy, Y_pred_optimal, average='macro'),
    }

    # **保存整体指标到文件**
    overall_header = ['Epoch', 'Aiming', 'Coverage', 'accuracy', 'absolute_true', 'absolute_false', 
                    #   'hamming_loss(AbsoluteFalse)',
                    #   'precision_micro', 'precision_macro',
                    #   'recall_micro', 'recall_macro',
                    #   'f1_micro', 'f1_macro'
                      ]
    write_to_csv(saveroot + '/整体指标.csv', overall_header,
                 [epoch,
                  "{:.4f}".format(overall_metrics['Aiming']),
                  "{:.4f}".format(overall_metrics['Coverage']),
                  "{:.4f}".format(overall_metrics['accuracy']),
                  "{:.4f}".format(overall_metrics['absolute_true']),
                  "{:.4f}".format(overall_metrics['absolute_false']),
                #   "{:.4f}".format(overall_metrics['hamming_loss']),
                #   "{:.4f}".format(overall_metrics['precision_micro']),
                #   "{:.4f}".format(overall_metrics['precision_macro']),
                #   "{:.4f}".format(overall_metrics['recall_micro']),
                #   "{:.4f}".format(overall_metrics['recall_macro']),
                #   "{:.4f}".format(overall_metrics['f1_micro']),
                #   "{:.4f}".format(overall_metrics['f1_macro'])
                  ])
    
    
    
def write_to_csv(file_path, header, data):
    # Check if the file already exists or not
    file_exists = False
    try:
        with open(file_path, 'r') as file:
            # Check if the file is not empty
            file_exists = bool(file.read())
    except FileNotFoundError:
        pass

    # Write data to CSV file
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)

        # Write header only if the file is newly created
        if not file_exists:
            writer.writerow(header)

        # Write the metrics for the current epoch
        writer.writerow(data)
        
        
        
def Aiming(y_hat, y):
    '''
    the “Aiming” rate (also called “Precision”) is to reflect the average ratio of the
    correctly predicted labels over the predicted labels; to measure the percentage
    of the predicted labels that hit the target of the real labels.
    '''

    n, m = y_hat.shape

    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        sorce_k += intersection / sum(y_hat[v])
    return sorce_k / n


def Coverage(y_hat, y):
    '''
    The “Coverage” rate (also called “Recall”) is to reflect the average ratio of the
    correctly predicted labels over the real labels; to measure the percentage of the
    real labels that are covered by the hits of prediction.
    '''

    n, m = y_hat.shape

    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        sorce_k += intersection / sum(y[v])

    return sorce_k / n


def Accuracy(y_hat, y):
    '''
    The “Accuracy” rate is to reflect the average ratio of correctly predicted labels
    over the total labels including correctly and incorrectly predicted labels as well
    as those real labels but are missed in the prediction
    '''

    n, m = y_hat.shape

    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        sorce_k += intersection / union
    return sorce_k / n


def AbsoluteTrue(y_hat, y):
    '''
    exactly match is True
    '''

    n, m = y_hat.shape
    sorce_k = 0
    for v in range(n):
        if list(y_hat[v]) == list(y[v]):
            sorce_k += 1
    return sorce_k/n


def AbsoluteFalse(y_hat, y):
    '''
    hamming loss
    '''

    n, m = y_hat.shape

    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v,h] == 1 or y[v,h] == 1:
                union += 1
            if y_hat[v,h] == 1 and y[v,h] == 1:
                intersection += 1
        sorce_k += (union-intersection)/m
    return sorce_k/n


def evaluate(y_hat, y):
    aiming = Aiming(y_hat, y)
    coverage = Coverage(y_hat, y)
    accuracy = Accuracy(y_hat, y)
    absolute_true = AbsoluteTrue(y_hat, y)
    absolute_false = AbsoluteFalse(y_hat, y)
    return aiming, coverage, accuracy, absolute_true, absolute_false


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)  # 对模组进行随机数设定
    np.random.seed(seed)  # 对numpy模组进行随机数设定
    torch.manual_seed(seed)  # 对torch中的CPU部分进行随机数设定
    torch.cuda.manual_seed(seed)  # 对torch中的GPU部分进行随机数设定
    torch.cuda.manual_seed_all(seed)  # 当使用多块GPU 时，均设置随机种子
    torch.backends.cudnn.deterministic = True  # 设置每次返回的卷积算法是一致的
    torch.backends.cudnn.benchmark = False  # cuDNN使用的非确定性算法自动寻找最适合当前配置的高效算法，设置为False 则每次的算法一致
    torch.backends.cudnn.enabled = True  # pytorch 使用CUDANN 加速，即使用GPU加速