from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score

from maha_extension.base import ALL_NET_TYPES, ALL_DATASETS, ALL_ADV_TYPES, n_classes, n_layers
from maha_extension.model import get_model_transforms
from maha_extension.data import Datasets, LabelledTrainLoader, LabelledTestLoader, LabelledGAValLoader, DatasetsGA, LabelledValLoader
from sklearn.metrics import precision_recall_curve, roc_auc_score, accuracy_score, auc

from utils import aupr
from enad_ga import f1_scr

import numpy as np
import pickle as pkl
import argparse

def run_enad_binary_stack_transfer(ds_name, net_type, adv_type, adv_transfer_type, binary_list, 
                                 outf='/kaggle/working', eval_on_test=False):

    if len(binary_list) != 8 or not all(isinstance(x, int) and x in [0, 1] for x in binary_list):
        raise ValueError("binary_list must be a list of 8 integers, each 0 or 1")

    if all(x == 0 for x in binary_list):
        return (0, 0)
        
    feature_sources = [
        ('lid', 'LID', 0, 
         lambda: np.load(f"/kaggle/input/lid-maha/LID_Maha/best {ds_name}/LID_best_{ds_name}_{adv_type}.npy"),
         lambda: np.load(f"/kaggle/input/lid-maha/LID_Maha/best {ds_name}/LID_best_{ds_name}_{adv_transfer_type}.npy")
        ),
        ('mahalanobis', 'Mahalanobis', 1, 
         lambda: np.load(f"/kaggle/input/lid-maha/LID_Maha/best {ds_name}/Mahalanobis_best_{ds_name}_{adv_type}.npy"),
         lambda: np.load(f"/kaggle/input/lid-maha/LID_Maha/best {ds_name}/Mahalanobis_best_{ds_name}_{adv_transfer_type}.npy")
        ),
        ('ocsvm', 'OCSVM', 2, 
         lambda: (
            pkl.load(open(f'/kaggle/input/ocsvm-new-pkl/ocsvm pkl new/{ds_name}/{adv_type}/OCSVM_net_detector_{net_type}_{ds_name}_{adv_type}.pkl', 'rb')),
            np.load(f"/kaggle/input/ocsvm-new-pkl/ocsvm pkl new/{ds_name}/{adv_type}/OCSVM_{net_type}_{ds_name}_{adv_type}.npy")
        ),
        lambda: (
            pkl.load(open(f'/kaggle/input/ocsvm-new-pkl/ocsvm pkl new/{ds_name}/{adv_transfer_type}/OCSVM_net_detector_{net_type}_{ds_name}_{adv_transfer_type}.pkl', 'rb')),
            np.load(f"/kaggle/input/ocsvm-new-pkl/ocsvm pkl new/{ds_name}/{adv_transfer_type}/OCSVM_{net_type}_{ds_name}_{adv_transfer_type}.npy")
        )
        ),
        ('knn', 'KNN', 3, 
        lambda: (
            pkl.load(open(f'/kaggle/input/new-ratio-full-pkl/{ds_name}/KNN/{adv_type}/KNN_net_detector_{net_type}_{ds_name}_{adv_type}.pkl', 'rb')),
            np.load(f"/kaggle/input/new-ratio-full-pkl/{ds_name}/KNN/{adv_type}/KNN_{net_type}_{ds_name}_{adv_type}.npy")
        ),
        lambda: (
            pkl.load(open(f'/kaggle/input/new-ratio-full-pkl/{ds_name}/KNN/{adv_transfer_type}/KNN_net_detector_{net_type}_{ds_name}_{adv_transfer_type}.pkl', 'rb')),
            np.load(f"/kaggle/input/new-ratio-full-pkl/{ds_name}/KNN/{adv_transfer_type}/KNN_{net_type}_{ds_name}_{adv_transfer_type}.npy")
        )
        ),
        ('randomforest', 'RF', 4, 
        lambda: (
            pkl.load(open(f'/kaggle/input/new-ratio-full-pkl/{ds_name}/Random Forest/{adv_type}/RF_net_detector_{net_type}_{ds_name}_{adv_type}.pkl', 'rb')),
            np.load(f"/kaggle/input/new-ratio-full-pkl/{ds_name}/Random Forest/{adv_type}/RF_{net_type}_{ds_name}_{adv_type}.npy")
        ),
        lambda: (
            pkl.load(open(f'/kaggle/input/new-ratio-full-pkl/{ds_name}/Random Forest/{adv_transfer_type}/RF_net_detector_{net_type}_{ds_name}_{adv_transfer_type}.pkl', 'rb')),
            np.load(f"/kaggle/input/new-ratio-full-pkl/{ds_name}/Random Forest/{adv_transfer_type}/RF_{net_type}_{ds_name}_{adv_transfer_type}.npy")
        )
        ),
        ('adaboost', 'AB', 5, 
        lambda: (
            pkl.load(open(f'/kaggle/input/new-ratio-full-pkl/{ds_name}/AdaBoost/{adv_type}/AB_net_detector_{net_type}_{ds_name}_{adv_type}.pkl', 'rb')),
            np.load(f"/kaggle/input/new-ratio-full-pkl/{ds_name}/AdaBoost/{adv_type}/AB_{net_type}_{ds_name}_{adv_type}.npy")
        ),
        lambda: (
            pkl.load(open(f'/kaggle/input/new-ratio-full-pkl/{ds_name}/AdaBoost/{adv_transfer_type}/AB_net_detector_{net_type}_{ds_name}_{adv_transfer_type}.pkl', 'rb')),
            np.load(f"/kaggle/input/new-ratio-full-pkl/{ds_name}/AdaBoost/{adv_transfer_type}/AB_{net_type}_{ds_name}_{adv_transfer_type}.npy")
        )
        ),
        ('xgboost', 'XGB', 6, 
        lambda: (
            pkl.load(open(f'/kaggle/input/new-ratio-full-pkl/{ds_name}/XGBoost/{adv_type}/XGB_net_detector_{net_type}_{ds_name}_{adv_type}.pkl', 'rb')),
            np.load(f"/kaggle/input/new-ratio-full-pkl/{ds_name}/XGBoost/{adv_type}/XGB_{net_type}_{ds_name}_{adv_type}.npy")
        ),
        lambda: (
            pkl.load(open(f'/kaggle/input/new-ratio-full-pkl/{ds_name}/XGBoost/{adv_transfer_type}/XGB_net_detector_{net_type}_{ds_name}_{adv_transfer_type}.pkl', 'rb')),
            np.load(f"/kaggle/input/new-ratio-full-pkl/{ds_name}/XGBoost/{adv_transfer_type}/XGB_{net_type}_{ds_name}_{adv_transfer_type}.npy")
        )
        ),
        ('lightgbm', 'LGBM', 7, 
        lambda: (
            pkl.load(open(f'/kaggle/input/new-ratio-full-pkl/{ds_name}/LightGBM/{adv_type}/LGBM_net_detector_{net_type}_{ds_name}_{adv_type}.pkl', 'rb')),
            np.load(f"/kaggle/input/new-ratio-full-pkl/{ds_name}/LightGBM/{adv_type}/LGBM_{net_type}_{ds_name}_{adv_type}.npy")
        ),
        lambda: (
            pkl.load(open(f'/kaggle/input/new-ratio-full-pkl/{ds_name}/LightGBM/{adv_transfer_type}/LGBM_net_detector_{net_type}_{ds_name}_{adv_transfer_type}.pkl', 'rb')),
            np.load(f"/kaggle/input/new-ratio-full-pkl/{ds_name}/LightGBM/{adv_transfer_type}/LGBM_{net_type}_{ds_name}_{adv_transfer_type}.npy")
        )
        ),
    ]

    model, in_transform = get_model_transforms(net_type, ds_name, n_classes(ds_name))
    
    # Use DatasetsGA for proper GA validation split
    source_dataset = DatasetsGA(ds_name, in_transform, net_type, adv_type, 
                               f"/kaggle/input/attacked-pth-files/{ds_name.upper()}/{adv_type}")
    target_dataset = DatasetsGA(ds_name, in_transform, net_type, adv_transfer_type, 
                               f"/kaggle/input/attacked-pth-files/{ds_name.upper()}/{adv_transfer_type}")
    
    # Data loaders for source (training)
    source_train_loader = LabelledTrainLoader(model, source_dataset)
    source_val_loader = LabelledValLoader(model, source_dataset)
    
    # Data loaders for target (evaluation)
    if eval_on_test:
        target_eval_loader = LabelledTestLoader(model, target_dataset)
        target_eval_idxs = target_dataset.idxs_test
    else:
        target_eval_loader = LabelledGAValLoader(model, source_dataset)
        target_eval_idxs = source_dataset.idxs_ga_val

    n_layers_used = n_layers(net_type)
    X_train_all = []
    X_val_all = []
    X_eval_target_all = []

    for fname, prefix, idx, load_source_fn, load_target_fn in feature_sources:
        if binary_list[idx] == 1:
            try:
                if fname in ['lid', 'mahalanobis']:
                    # Source data (for training)
                    source_data = load_source_fn()
                    X_train_all.append(source_data[source_dataset.idxs_train][:, :n_layers_used])
                    X_val_all.append(source_data[source_dataset.idxs_val][:, :n_layers_used])
                    
                    # Target data (for evaluation)
                    if eval_on_test:
                        target_data = load_target_fn()
                        X_eval_target_all.append(target_data[target_eval_idxs][:, :n_layers_used])
                    else:
                        X_eval_target_all.append(source_data[target_eval_idxs][:, :n_layers_used])
                    
                else:
                    # Source detectors and data (for training)
                    source_det, _ = load_source_fn()
                    source_train_data = source_det.get_layer_scores(source_train_loader)
                    source_val_data = source_det.get_layer_scores(source_val_loader)
                    
                    X_train_all.append(source_train_data[:, :n_layers_used])
                    X_val_all.append(source_val_data[:, :n_layers_used])
                    
                    # Target detectors and data (for evaluation)
                    if eval_on_test:
                        target_det, _ = load_target_fn()
                        target_eval_data = target_det.get_layer_scores(target_eval_loader)
                        X_eval_target_all.append(target_eval_data[:, :n_layers_used])
                    else:
                        target_eval_data = source_det.get_layer_scores(target_eval_loader)
                        X_eval_target_all.append(target_eval_data[:, :n_layers_used])
                    
            except Exception as e:
                print(f"Error loading {prefix}: {e}")
                continue

    if not X_train_all or not X_val_all or not X_eval_target_all:
        raise ValueError("No features were successfully loaded based on the binary list")

    # Concatenate features
    X_train = np.c_[tuple(X_train_all)]
    X_val = np.c_[tuple(X_val_all)]
    X_eval_target = np.c_[tuple(X_eval_target_all)]
    
    X_train_combined = np.vstack((X_train, X_val))
    y_train_combined = np.concatenate((
        source_dataset.adv_test[source_dataset.idxs_train], 
        source_dataset.adv_test[source_dataset.idxs_val]
    ))

    lr = LogisticRegressionCV(penalty='l2', max_iter=5000, n_jobs=-1)
    lr.fit(X_train_combined, y_train_combined)

    if eval_on_test:
        y_eval_target = target_dataset.adv_test[target_eval_idxs]
    else:
        y_eval_target = source_dataset.adv_test[target_eval_idxs]
        
    adv_conf_eval = lr.predict_proba(X_eval_target)[:, 0]
    auroc_eval = roc_auc_score(y_eval_target, -adv_conf_eval)
    aupr_eval = aupr(y_eval_target, adv_conf_eval, pos_label=0)

    f1_test = f1_scr(y_eval_target, adv_conf_eval, pos_label=0)
    
    return (aupr_eval, auroc_eval, f1_test)