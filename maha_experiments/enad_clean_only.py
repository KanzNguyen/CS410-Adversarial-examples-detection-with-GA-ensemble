from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score

from maha_extension.base import ALL_NET_TYPES, ALL_DATASETS, ALL_ADV_TYPES, n_classes, n_layers
from maha_extension.model import get_model_transforms
from maha_extension.data import Datasets, LabelledTrainLoader, LabelledTestLoader, LabelledGAValLoader, DatasetsGA, LabelledCleanOnlyTestLoader, LabelledCleanOnlyGAValLoader
from sklearn.metrics import precision_recall_curve, roc_auc_score, accuracy_score, auc

from utils import aupr

import numpy as np
import pickle as pkl
import argparse

def run_enad_binary_stack_clean_only_GA(ds_name, net_type, adv_type, binary_list, outf='/kaggle/working', eval_on_test=False):
    if len(binary_list) != 8 or not all(isinstance(x, int) and x in [0, 1] for x in binary_list):
        raise ValueError("binary_list must be a list of 8 integers, each 0 or 1")

    if all(x == 0 for x in binary_list):
        return (0, 0, 0)
        
    feature_sources = [
        ('lid', 'LID', 0, lambda: np.load(f"/kaggle/input/lid-maha/LID_Maha/best {ds_name}/LID_best_{ds_name}_{adv_type}.npy")),
        ('mahalanobis', 'Mahalanobis', 1, lambda: np.load(f"/kaggle/input/lid-maha/LID_Maha/best {ds_name}/Mahalanobis_best_{ds_name}_{adv_type}.npy")),
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
    
    # Use Datasets to get proper splits
    dataset = DatasetsGA(ds_name, in_transform, net_type, adv_type, 
                      f"/kaggle/input/attacked-pth-files/{ds_name.upper()}/{adv_type}")
    
    # Training data loader (still use mixed data for training)
    data_loader = LabelledTrainLoader(model, dataset)
    
    # Choose clean-only evaluation indices and loaders
    if eval_on_test:
        eval_idxs = dataset.idxs_test_clean_only
        eval_loader = LabelledCleanOnlyTestLoader(model, dataset)
        eval_labels = dataset.adv_test_clean_only[eval_idxs]  # All 1s (clean)
    else:
        eval_idxs = dataset.idxs_ga_val_clean_only
        eval_loader = LabelledCleanOnlyGAValLoader(model, dataset)
        eval_labels = dataset.adv_test_clean_only[eval_idxs]  # All 1s (clean)

    n_layers_used = n_layers(net_type)
    X_train_all = []
    X_eval_all = []

    for fname, prefix, idx, load_fn in feature_sources:
        if binary_list[idx] == 1:
            try:
                if fname in ['lid', 'mahalanobis']:
                    data = load_fn()
                    X_train_all.append(data[dataset.idxs_train][:, :n_layers_used])
                    
                    # For clean-only evaluation, extract features from clean portion only
                    p_size = len(data) // 3
                    clean_data = data[p_size:2*p_size]  # Clean portion from original mixed data
                    if eval_on_test:
                        X_eval_all.append(clean_data[dataset.idxs_test_clean_only][:, :n_layers_used])
                    else:
                        X_eval_all.append(clean_data[dataset.idxs_ga_val_clean_only][:, :n_layers_used])
                        
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

    # Train on mixed data (adversarial labels: 0=adversarial, 1=clean)
    lr = LogisticRegressionCV(penalty='l1', solver='liblinear', max_iter=7000, n_jobs=-1)
    lr.fit(X_train, dataset.adv_test[dataset.idxs_train])

    # Predict on clean-only data
    clean_probs = lr.predict_proba(X_eval)[:, 1]  # Probability of being clean (class 1)
    
    # Use accuracy-based metrics for clean-only evaluation
    threshold = 0.5
    predicted_labels = (clean_probs > threshold).astype(int)
    
    # Calculate accuracy metrics
    accuracy = np.mean(predicted_labels == eval_labels)  # Overall accuracy on clean samples
    avg_confidence = np.mean(clean_probs)  # Average confidence in clean prediction
    false_positive_rate = np.mean(predicted_labels == 0)  # Rate of clean samples flagged as adversarial

    return (accuracy, avg_confidence, false_positive_rate)