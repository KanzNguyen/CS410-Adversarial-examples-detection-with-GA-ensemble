from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score

from maha_extension.base import ALL_NET_TYPES, ALL_DATASETS, ALL_ADV_TYPES, n_classes, n_layers
from maha_extension.model import get_model_transforms
from maha_extension.data import Datasets, LabelledTrainLoader, LabelledTestLoader, LabelledGAValLoader, DatasetsGA
from sklearn.metrics import precision_recall_curve, roc_auc_score, accuracy_score, auc

from utils import aupr

import numpy as np
import pickle as pkl
import argparse

def f1_scr(y_true, y_pred, pos_label=0):
    precision, recall, _ = precision_recall_curve(y_true, y_pred, pos_label=pos_label)
    f1_scores = 2*(precision * recall) / (precision + recall + 1e-8)
    return np.max(f1_scores)

def run_enad_binary_stack_all(ds_name, net_type, adv_type, binary_list, outf='/kaggle/working', eval_on_test=False):
    if len(binary_list) != 8 or not all(isinstance(x, int) and x in [0, 1] for x in binary_list):
        raise ValueError("binary_list must be a list of 8 integers, each 0 or 1")

    if all(x == 0 for x in binary_list):
        return (0, 0)
        
    feature_sources = [
        ('lid', 'LID', 0, lambda: np.load(f"/kaggle/input/best-numpy-new/best numpy new/best {ds_name}/LID_best_{ds_name}_{adv_type}.npy")),
        ('mahalanobis', 'Mahalanobis', 1, lambda: np.load(f"/kaggle/input/best-numpy-new/best numpy new/best {ds_name}/Mahalanobis_best_{ds_name}_{adv_type}.npy")),
        ('ocsvm', 'OCSVM', 2, lambda: (
            pkl.load(open(f'/kaggle/input/ocsvm-new-pkl/ocsvm pkl new/{ds_name}/{adv_type}/OCSVM_net_detector_{net_type}_{ds_name}_{adv_type}.pkl', 'rb')),
            np.load(f"/kaggle/input/ocsvm-new-pkl/ocsvm pkl new/{ds_name}/{adv_type}/OCSVM_{net_type}_{ds_name}_{adv_type}.npy")
        )),
        ('knn', 'KNN', 3, lambda: (
            pkl.load(open(f'/kaggle/input/new-ratio-full-pkl/{ds_name}/KNN/{adv_type}/KNN_net_detector_{net_type}_{ds_name}_{adv_type}.pkl', 'rb')),
            np.load(f"/kaggle/input/new-ratio-full-pkl/{ds_name}/KNN/{adv_type}/KNN_{net_type}_{ds_name}_{adv_type}.npy")
        )),
        ('randomforest', 'RF', 4, lambda: (
            pkl.load(open(f'/kaggle/input/new-ratio-full-pkl/{ds_name}/Random Forest/{adv_type}/RF_net_detector_{net_type}_{ds_name}_{adv_type}.pkl', 'rb')),
            np.load(f"/kaggle/input/new-ratio-full-pkl/{ds_name}/Random Forest/{adv_type}/RF_{net_type}_{ds_name}_{adv_type}.npy")
        )),
        ('adaboost', 'AB', 5, lambda: (
            pkl.load(open(f'/kaggle/input/new-ratio-full-pkl/{ds_name}/AdaBoost/{adv_type}/AB_net_detector_{net_type}_{ds_name}_{adv_type}.pkl', 'rb')),
            np.load(f"/kaggle/input/new-ratio-full-pkl/{ds_name}/AdaBoost/{adv_type}/AB_{net_type}_{ds_name}_{adv_type}.npy")
        )),
        ('xgboost', 'XGB', 6, lambda: (
            pkl.load(open(f'/kaggle/input/new-ratio-full-pkl/{ds_name}/XGBoost/{adv_type}/XGB_net_detector_{net_type}_{ds_name}_{adv_type}.pkl', 'rb')),
            np.load(f"/kaggle/input/new-ratio-full-pkl/{ds_name}/XGBoost/{adv_type}/XGB_{net_type}_{ds_name}_{adv_type}.npy")
        )),
        ('lightgbm', 'LGBM', 7, lambda: (
            pkl.load(open(f'/kaggle/input/new-ratio-full-pkl/{ds_name}/LightGBM/{adv_type}/LGBM_net_detector_{net_type}_{ds_name}_{adv_type}.pkl', 'rb')),
            np.load(f"/kaggle/input/new-ratio-full-pkl/{ds_name}/LightGBM/{adv_type}/LGBM_{net_type}_{ds_name}_{adv_type}.npy")
        )),
    ]

    model, in_transform = get_model_transforms(net_type, ds_name, n_classes(ds_name))
    
    # Use DatasetsGA to get proper 4-way split with GA validation
    dataset = DatasetsGA(ds_name, in_transform, net_type, adv_type, 
                        f"/kaggle/input/attacked-pth-files/{ds_name.upper()}/{adv_type}")
    
    # Training data loader (same as before)
    data_loader = LabelledTrainLoader(model, dataset)
    
    # Choose evaluation indices based on parameter
    if eval_on_test:
        eval_idxs = dataset.idxs_test
        eval_loader = LabelledTestLoader(model, dataset)
    else:
        eval_idxs = dataset.idxs_ga_val  # Use GA validation set
        eval_loader = LabelledGAValLoader(model, dataset)

    n_layers_used = n_layers(net_type)
    X_train_all = []
    X_eval_all = []

    for fname, prefix, idx, load_fn in feature_sources:
        if binary_list[idx] == 1:
            try:
                if fname in ['lid', 'mahalanobis']:
                    data = load_fn()
                    X_train_all.append(data[dataset.idxs_train][:, :n_layers_used])
                    X_eval_all.append(data[eval_idxs][:, :n_layers_used])
                else:
                    det, det_test = load_fn()
                    det_train_data = det.get_layer_scores(data_loader)
                    det_eval_data = det.get_layer_scores(eval_loader)
                    X_train_all.append(det_train_data[:, :n_layers_used])
                    X_eval_all.append(det_eval_data[:, :n_layers_used])
            except Exception as e:
                print(f"Error loading {prefix}: {e}")
                continue
                
    if not X_train_all or not X_eval_all:
        raise ValueError("No features were successfully loaded based on the binary list")

    X_train = np.c_[tuple(X_train_all)]
    X_eval = np.c_[tuple(X_eval_all)]

    lr = LogisticRegressionCV(penalty='l1', solver='liblinear', max_iter=7000, n_jobs=-1)
    lr.fit(X_train, dataset.adv_test[dataset.idxs_train])

    adv_conf_eval = lr.predict_proba(X_eval)[:, 0]
    auroc_eval = roc_auc_score(dataset.adv_test[eval_idxs], -adv_conf_eval)
    aupr_eval = aupr(dataset.adv_test[eval_idxs], adv_conf_eval, pos_label=0)

    f1_test = f1_scr(dataset.adv_test[eval_idxs], adv_conf_eval, pos_label=0)

    return (auroc_eval, aupr_eval, f1_test)