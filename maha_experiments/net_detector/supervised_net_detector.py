from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from skopt import BayesSearchCV
from skopt.space import Integer
import pickle
import os
import torch
import logging
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from net_detector.supervised_net_detector import GroupedScaler
from maha_extension.base import n_layers, n_classes

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from skopt.space import Integer, Real
from maha_extension.model import get_model_transforms
from maha_extension.data import DatasetsGA, TrainValLoader, LabelledTrainLoader, LabelledTestLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm.auto import tqdm as tqdm_auto
import numpy as np
import pickle
import json

import argparse
import sys


class SupervisedDetector:
    def __init__(self, n_layers, detector_class, tqdm=True, logger=None, exp_name="",
                 outf="", pre_computed=True, pre_computed_path=None):
        
        self.detector_class = detector_class
        self.tqdm = not tqdm
        self.n_layers = n_layers
        self.logger = logger
        self.exp_name = exp_name
        self.outf = outf
        self.pre_computed = pre_computed
        self.pre_computed_path = pre_computed_path

    def fit(self, dl_train, dl_unseen_train, adv_unseen_train):
        self.logger.info(f"{self.exp_name}: training layer detectors...")
        self.detectors = self.train_layer_detectors(dl_train)
        
        train_scores = self.get_layer_scores(dl_unseen_train)
        self.logger.info(f"{self.exp_name}: training final logistic...")
        self.lr = self.train_logistic_regression(train_scores, adv_unseen_train)

    def predict(self, data_loader):
        predicted_scores = self.get_layer_scores(data_loader)
        metrics = self.get_final_metrics(predicted_scores)
        return predicted_scores, metrics
        
    def train_layer_detectors(self, data_loader):
        detectors = []
        for layer_idx in tqdm_auto(range(self.n_layers), disable=not self.tqdm, 
                                   desc="Training layer detectors..."):
            self.logger.info(f"{self.exp_name}: started layer {layer_idx}")
            X_train, X_valid, y_train, y_valid, adv_train, adv_valid = data_loader(layer_idx)
            
            # Combine train and validation data
            X = np.concatenate((X_train, X_valid))
            y = np.concatenate((y_train, y_valid))
            adv = np.concatenate((adv_train, adv_valid))
            
            # Debug: Print unique classes and counts in adv
            unique, counts = np.unique(adv, return_counts=True)
            print(f"Layer {layer_idx} adv label distribution: {dict(zip(unique, counts))}")
            
            if self.pre_computed and self.pre_computed_path:
                # Load pre-computed parameters
                with open(self.pre_computed_path, "r") as fname:
                    params = json.load(fname)
                detector = clone(self.detector_class)
                best_params = params.get(f"{self.exp_name}_{layer_idx}", {})
                detector = detector.set_params(**best_params).fit(np.c_[X, y], adv)
            else:
                # Perform hyperparameter search with stratified cross-validation
                srch = clone(self.detector_class)
                
                # Set up stratified k-fold cross-validation
                stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                srch.set_params(cv=stratified_cv)
                
                # Fit with combined features and adversarial labels
                srch.fit(np.c_[X, y], adv)
                
                # Save the search results
                with open(f"{self.outf}/bayes_{self.exp_name}_{layer_idx}.pkl", "wb") as outp:
                    pickle.dump(srch, outp, pickle.HIGHEST_PROTOCOL)
                    
                # Train final detector with best parameters
                detector = srch.estimator.set_params(**srch.best_params_).fit(np.c_[X, y], adv)
                self.logger.info(f"{self.exp_name}: ACC = {srch.best_score_}, BEST_PARAMS = {srch.best_params_}")
            
            detectors.append(detector)
        return detectors

    def get_layer_scores(self, data_loader):
        scores = []
        for layer_idx in tqdm_auto(range(self.n_layers), disable=not self.tqdm, 
                                   desc="Extracting layer scores...", leave=False):
            X, y = data_loader(layer_idx)
            # Get probability predictions
            proba = self.detectors[layer_idx].predict_proba(np.c_[X, y])
            
            # Check what classes the detector learned
            classes = self.detectors[layer_idx].classes_
            #print(f"Layer {layer_idx} detector classes: {classes}")
            
            # Find the index corresponding to adversarial samples (class 0)
            if 0 in classes:
                adv_idx = np.where(classes == 0)[0][0]
            else:
                # Fallback to first class
                adv_idx = 0
                
            #print(f"Using index {adv_idx} for adversarial probability")
            scores.append(proba[:, adv_idx])  # Probability of adversarial class (0)
        
        return np.vstack(scores).T

    def train_logistic_regression(self, X, adv):
        lr = LogisticRegressionCV(penalty="l1", solver="liblinear", max_iter=10000, n_jobs=-1)
        lr.fit(X, adv)
        return lr
        
    def get_final_metrics(self, X):
        preds = self.lr.predict(X)
        probas = self.lr.predict_proba(X)
        return np.array([preds, probas[:, 0]]).T

def run_knn(net_type, ds_name, adv_type, outf='/kaggle/input/attacked-pth-files', knn_fname='knn', bayes_n_points=1, bayes_n_iter=10, use_logger=True, batch_size=100, pre_computed=True):
    outf = outf + f"/{ds_name.upper()}/{adv_type}"
    data_outf = f"/kaggle/working/{ds_name.upper()}"
    if not os.path.isdir(data_outf):
        os.mkdir(data_outf)
    data_outf = f"/kaggle/working/{ds_name.upper()}/{adv_type}"
    if not os.path.isdir(data_outf):
        os.mkdir(data_outf)
    knn_outf = f"{data_outf}/{knn_fname}"
    if not os.path.isdir(knn_outf):
        os.mkdir(knn_outf)

    torch.cuda.manual_seed(0)
    torch.cuda.set_device(0)
    np.random.seed(0)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fhandler = logging.FileHandler(f'{knn_outf}/scan.log', 'a+', 'utf-8')
    shandler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt="%Y-%m-%d %H:%M")
    fhandler.setFormatter(formatter)
    shandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.addHandler(shandler)
    logger.disables = not use_logger

    exp_name = f"{net_type}_{ds_name}_{adv_type}"
    logger.info(f"{exp_name}: STARTED")

    model, in_transform = get_model_transforms(net_type, ds_name, n_classes(ds_name))
    ds = DatasetsGA(ds_name, in_transform, net_type, adv_type, outf)

    clf = Pipeline(steps=[
        ('scaler', GroupedScaler()),
        ('PCA', PCA(whiten=True)),
        ('clf', KNeighborsClassifier())])

    if pre_computed:
        layer_det = clf
    else:
        # REMOVED PredefinedSplit - StratifiedKFold will be set in SupervisedDetector
        search_spaces = {
            "clf__n_neighbors": Integer(1, 15),
            "clf__weights": ["uniform", "distance"]
        }
        layer_det = BayesSearchCV(clf, search_spaces, n_iter=bayes_n_iter, n_points=bayes_n_points,
            n_jobs=-1, scoring='accuracy', return_train_score=False, refit=False, verbose=0)

    knn_trainer = SupervisedDetector(n_layers(net_type), layer_det, tqdm=False, logger=logger, exp_name=exp_name, outf=knn_outf, pre_computed=pre_computed)
    knn_trainer.fit(TrainValLoader(model, ds, batch_size=batch_size), LabelledTrainLoader(model, ds, batch_size=batch_size), ds.adv_test[ds.idxs_train])

    test_scores, output = knn_trainer.predict(LabelledTestLoader(model, ds, batch_size=batch_size))
    all_output = np.hstack((test_scores, output, np.expand_dims(ds.adv_test[ds.idxs_test], axis=1)))

    with open(f'{knn_outf}/KNN_net_detector_{exp_name}.pkl', 'wb') as outp:
        pickle.dump(knn_trainer, outp, pickle.HIGHEST_PROTOCOL)

    np.save(f"{knn_outf}/KNN_{exp_name}", all_output)

    accuracy = accuracy_score(ds.adv_test[ds.idxs_test], output[:, 0])
    auroc = roc_auc_score(ds.adv_test[ds.idxs_test], -output[:, 1])

    logger.info(f"{exp_name}: ACC = {accuracy}, AUROC = {round(auroc*100, 2)}")

def run_randomforest(net_type, ds_name, adv_type, outf='/kaggle/input/attacked-pth-files', rf_fname='randomforest', bayes_n_points=1, bayes_n_iter=10, use_logger=True, batch_size=100, pre_computed=True):
    outf = outf + f"/{ds_name.upper()}/{adv_type}"
    data_outf = f"/kaggle/working/{ds_name.upper()}"
    if not os.path.isdir(data_outf):
        os.mkdir(data_outf)
    data_outf = f"/kaggle/working/{ds_name.upper()}/{adv_type}"
    if not os.path.isdir(data_outf):
        os.mkdir(data_outf)
    rf_outf = f"{data_outf}/{rf_fname}"
    if not os.path.isdir(rf_outf):
        os.mkdir(rf_outf)

    torch.cuda.manual_seed(0)
    torch.cuda.set_device(0)
    np.random.seed(0)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fhandler = logging.FileHandler(f'{rf_outf}/scan.log', 'a+', 'utf-8')
    shandler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt="%Y-%m-%d %H:%M")
    fhandler.setFormatter(formatter)
    shandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.addHandler(shandler)
    logger.disables = not use_logger

    exp_name = f"{net_type}_{ds_name}_{adv_type}"
    logger.info(f"{exp_name}: STARTED")

    model, in_transform = get_model_transforms(net_type, ds_name, n_classes(ds_name))
    ds = DatasetsGA(ds_name, in_transform, net_type, adv_type, outf)

    clf = Pipeline(steps=[
        ('scaler', GroupedScaler()),
        ('PCA', PCA(whiten=True)),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=0))
    ])

    if pre_computed:
        layer_det = clf
    else:
        search_spaces = {
            "clf__n_estimators": Integer(50, 200),
            "clf__max_depth": Integer(3, 20)
        }
        layer_det = BayesSearchCV(clf, search_spaces, n_iter=bayes_n_iter, n_points=bayes_n_points,
            n_jobs=-1, scoring='accuracy', return_train_score=False, refit=False, verbose=0)

    rf_trainer = SupervisedDetector(n_layers(net_type), layer_det, tqdm=False, logger=logger, exp_name=exp_name, outf=rf_outf, pre_computed=pre_computed)
    rf_trainer.fit(TrainValLoader(model, ds, batch_size=batch_size), LabelledTrainLoader(model, ds, batch_size=batch_size), ds.adv_test[ds.idxs_train])

    test_scores, output = rf_trainer.predict(LabelledTestLoader(model, ds, batch_size=batch_size))
    all_output = np.hstack((test_scores, output, np.expand_dims(ds.adv_test[ds.idxs_test], axis=1)))

    with open(f'{rf_outf}/RF_net_detector_{exp_name}.pkl', 'wb') as outp:
        pickle.dump(rf_trainer, outp, pickle.HIGHEST_PROTOCOL)

    np.save(f"{rf_outf}/RF_{exp_name}", all_output)

    accuracy = accuracy_score(ds.adv_test[ds.idxs_test], output[:, 0])
    auroc = roc_auc_score(ds.adv_test[ds.idxs_test], -output[:, 1])

    logger.info(f"{exp_name}: ACC = {accuracy}, AUROC = {round(auroc*100, 2)}")
    
def run_adaboost(net_type, ds_name, adv_type, outf='/kaggle/input/attacked-pth-files', ab_fname='adaboost', bayes_n_points=1, bayes_n_iter=10, use_logger=True, batch_size=100, pre_computed=True):
    outf = outf + f"/{ds_name.upper()}/{adv_type}"
    data_outf = f"/kaggle/working/{ds_name.upper()}"
    if not os.path.isdir(data_outf):
        os.mkdir(data_outf)
    data_outf = f"/kaggle/working/{ds_name.upper()}/{adv_type}"
    if not os.path.isdir(data_outf):
        os.mkdir(data_outf)
    ab_outf = f"{data_outf}/{ab_fname}"
    if not os.path.isdir(ab_outf):
        os.mkdir(ab_outf)

    torch.cuda.manual_seed(0)
    torch.cuda.set_device(0)
    np.random.seed(0)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fhandler = logging.FileHandler(f'{ab_outf}/scan.log', 'a+', 'utf-8')
    shandler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt="%Y-%m-%d %H:%M")
    fhandler.setFormatter(formatter)
    shandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.addHandler(shandler)
    logger.disabled = not use_logger

    exp_name = f"{net_type}_{ds_name}_{adv_type}"
    logger.info(f"{exp_name}: STARTED")

    model, in_transform = get_model_transforms(net_type, ds_name, n_classes(ds_name))
    ds = DatasetsGA(ds_name, in_transform, net_type, adv_type, outf)

    clf = Pipeline(steps=[
        ('scaler', GroupedScaler()),
        ('PCA', PCA(whiten=True)),
        ('clf', AdaBoostClassifier(n_estimators=100, random_state=0))
    ])

    if pre_computed:
        layer_det = clf
    else:
        search_spaces = {
            "clf__n_estimators": Integer(50, 200)
        }
        layer_det = BayesSearchCV(clf, search_spaces, n_iter=bayes_n_iter, n_points=bayes_n_points,
            n_jobs=-1, scoring='accuracy', return_train_score=False, refit=False, verbose=0)

    ab_trainer = SupervisedDetector(n_layers(net_type), layer_det, tqdm=False, logger=logger, exp_name=exp_name, outf=ab_outf, pre_computed=pre_computed)
    ab_trainer.fit(TrainValLoader(model, ds, batch_size=batch_size), LabelledTrainLoader(model, ds, batch_size=batch_size), ds.adv_test[ds.idxs_train])

    test_scores, output = ab_trainer.predict(LabelledTestLoader(model, ds, batch_size=batch_size))
    all_output = np.hstack((test_scores, output, np.expand_dims(ds.adv_test[ds.idxs_test], axis=1)))

    with open(f'{ab_outf}/AB_net_detector_{exp_name}.pkl', 'wb') as outp:
        pickle.dump(ab_trainer, outp, pickle.HIGHEST_PROTOCOL)

    np.save(f"{ab_outf}/AB_{exp_name}", all_output)

    accuracy = accuracy_score(ds.adv_test[ds.idxs_test], output[:, 0])
    auroc = roc_auc_score(ds.adv_test[ds.idxs_test], -output[:, 1])

    logger.info(f"{exp_name}: ACC = {accuracy}, AUROC = {round(auroc*100, 2)}")

def run_gbdt(net_type, ds_name, adv_type, outf='/kaggle/input/attacked-pth-files', gbdt_fname='gbdt', bayes_n_points=1, bayes_n_iter=10, use_logger=True, batch_size=100, pre_computed=True):
    outf = outf + f"/{ds_name.upper()}/{adv_type}"
    data_outf = f"/kaggle/working/{ds_name.upper()}"
    if not os.path.isdir(data_outf):
        os.mkdir(data_outf)
    data_outf = f"/kaggle/working/{ds_name.upper()}/{adv_type}"
    if not os.path.isdir(data_outf):
        os.mkdir(data_outf)
    gbdt_outf = f"{data_outf}/{gbdt_fname}"
    if not os.path.isdir(gbdt_outf):
        os.mkdir(gbdt_outf)

    torch.cuda.manual_seed(0)
    torch.cuda.set_device(0)
    np.random.seed(0)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fhandler = logging.FileHandler(f'{gbdt_outf}/scan.log', 'a+', 'utf-8')
    shandler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt="%Y-%m-%d %H:%M")
    fhandler.setFormatter(formatter)
    shandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.addHandler(shandler)
    logger.disabled = not use_logger

    exp_name = f"{net_type}_{ds_name}_{adv_type}"
    logger.info(f"{exp_name}: STARTED")

    model, in_transform = get_model_transforms(net_type, ds_name, n_classes(ds_name))
    ds = DatasetsGA(ds_name, in_transform, net_type, adv_type, outf)

    clf = Pipeline(steps=[
        ('scaler', GroupedScaler()),
        ('PCA', PCA(whiten=True)),
        ('clf', GradientBoostingClassifier(
            n_estimators=100, 
            max_depth=3, 
            min_samples_leaf=20, 
            subsample=0.8, 
            random_state=0
        ))
    ])

    if pre_computed:
        layer_det = clf
    else:
        search_spaces = {
            "clf__n_estimators": Integer(50, 200),
            "clf__max_depth": Integer(3, 10)
        }
        layer_det = BayesSearchCV(clf, search_spaces, n_iter=bayes_n_iter, n_points=bayes_n_points,
            n_jobs=-1, scoring='accuracy', return_train_score=False, refit=False, verbose=0)

    gbdt_trainer = SupervisedDetector(n_layers(net_type), layer_det, tqdm=False, logger=logger, exp_name=exp_name, outf=gbdt_outf, pre_computed=pre_computed)
    gbdt_trainer.fit(TrainValLoader(model, ds, batch_size=batch_size), LabelledTrainLoader(model, ds, batch_size=batch_size), ds.adv_test[ds.idxs_train])

    test_scores, output = gbdt_trainer.predict(LabelledTestLoader(model, ds, batch_size=batch_size))
    all_output = np.hstack((test_scores, output, np.expand_dims(ds.adv_test[ds.idxs_test], axis=1)))

    with open(f'{gbdt_outf}/GBDT_net_detector_{exp_name}.pkl', 'wb') as outp:
        pickle.dump(gbdt_trainer, outp, pickle.HIGHEST_PROTOCOL)

    np.save(f"{gbdt_outf}/GBDT_{exp_name}", all_output)

    accuracy = accuracy_score(ds.adv_test[ds.idxs_test], output[:, 0])
    auroc = roc_auc_score(ds.adv_test[ds.idxs_test], -output[:, 1])

    logger.info(f"{exp_name}: ACC = {accuracy}, AUROC = {round(auroc*100, 2)}")

def run_lightgbm(net_type, ds_name, adv_type, outf='/kaggle/input/attacked-pth-files', lgbm_fname='lightgbm', bayes_n_points=1, bayes_n_iter=10, use_logger=True, batch_size=100, pre_computed=True):
    outf = outf + f"/{ds_name.upper()}/{adv_type}"
    data_outf = f"/kaggle/working/{ds_name.upper()}"
    if not os.path.isdir(data_outf):
        os.mkdir(data_outf)
    data_outf = f"/kaggle/working/{ds_name.upper()}/{adv_type}"
    if not os.path.isdir(data_outf):
        os.mkdir(data_outf)
    lgbm_outf = f"{data_outf}/{lgbm_fname}"
    if not os.path.isdir(lgbm_outf):
        os.mkdir(lgbm_outf)

    torch.cuda.manual_seed(0)
    torch.cuda.set_device(0)
    np.random.seed(0)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fhandler = logging.FileHandler(f'{lgbm_outf}/scan.log', 'a+', 'utf-8')
    shandler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt="%Y-%m-%d %H:%M")
    fhandler.setFormatter(formatter)
    shandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.addHandler(shandler)
    logger.disabled = not use_logger

    exp_name = f"{net_type}_{ds_name}_{adv_type}"
    logger.info(f"{exp_name}: STARTED")

    model, in_transform = get_model_transforms(net_type, ds_name, n_classes(ds_name))
    ds = DatasetsGA(ds_name, in_transform, net_type, adv_type, outf)

    clf = Pipeline(steps=[
        ('scaler', GroupedScaler()),
        ('PCA', PCA(whiten=True)),
        ('clf', lgb.LGBMClassifier(n_estimators=100, max_depth=5, random_state=0))
    ])

    if pre_computed:
        layer_det = clf
    else:
        search_spaces = {
            "clf__n_estimators": Integer(50, 200),
            "clf__max_depth": Integer(3, 10)
        }
        layer_det = BayesSearchCV(clf, search_spaces, n_iter=bayes_n_iter, n_points=bayes_n_points,
            n_jobs=-1, scoring='accuracy', return_train_score=False, refit=False, verbose=0)

    lgbm_trainer = SupervisedDetector(n_layers(net_type), layer_det, tqdm=False, logger=logger, exp_name=exp_name, outf=lgbm_outf, pre_computed=pre_computed)
    lgbm_trainer.fit(TrainValLoader(model, ds, batch_size=batch_size), LabelledTrainLoader(model, ds, batch_size=batch_size), ds.adv_test[ds.idxs_train])

    test_scores, output = lgbm_trainer.predict(LabelledTestLoader(model, ds, batch_size=batch_size))
    all_output = np.hstack((test_scores, output, np.expand_dims(ds.adv_test[ds.idxs_test], axis=1)))

    with open(f'{lgbm_outf}/LGBM_net_detector_{exp_name}.pkl', 'wb') as outp:
        pickle.dump(lgbm_trainer, outp, pickle.HIGHEST_PROTOCOL)

    np.save(f"{lgbm_outf}/LGBM_{exp_name}", all_output)

    accuracy = accuracy_score(ds.adv_test[ds.idxs_test], output[:, 0])
    auroc = roc_auc_score(ds.adv_test[ds.idxs_test], -output[:, 1])

    logger.info(f"{exp_name}: ACC = {accuracy}, AUROC = {round(auroc*100, 2)}")

def run_xgboost(net_type, ds_name, adv_type, outf='/kaggle/input/attacked-pth-files', xgb_fname='xgboost', bayes_n_points=1, bayes_n_iter=10, use_logger=True, batch_size=100, pre_computed=True):
    outf = outf + f"/{ds_name.upper()}/{adv_type}"
    data_outf = f"/kaggle/working/{ds_name.upper()}"
    if not os.path.isdir(data_outf):
        os.mkdir(data_outf)
    data_outf = f"/kaggle/working/{ds_name.upper()}/{adv_type}"
    if not os.path.isdir(data_outf):
        os.mkdir(data_outf)
    xgb_outf = f"{data_outf}/{xgb_fname}"
    if not os.path.isdir(xgb_outf):
        os.mkdir(xgb_outf)

    torch.cuda.manual_seed(0)
    torch.cuda.set_device(0)
    np.random.seed(0)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fhandler = logging.FileHandler(f'{xgb_outf}/scan.log', 'a+', 'utf-8')
    shandler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt="%Y-%m-%d %H:%M")
    fhandler.setFormatter(formatter)
    shandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.addHandler(shandler)
    logger.disables = not use_logger

    exp_name = f"{net_type}_{ds_name}_{adv_type}"
    logger.info(f"{exp_name}: STARTED")

    model, in_transform = get_model_transforms(net_type, ds_name, n_classes(ds_name))
    ds = DatasetsGA(ds_name, in_transform, net_type, adv_type, outf)

    clf = Pipeline(steps=[
        ('scaler', GroupedScaler()),
        ('PCA', PCA(whiten=True)),
        ('clf', xgb.XGBClassifier(n_estimators=100, max_depth=5))
    ])

    if pre_computed:
        layer_det = clf
    else:
        search_spaces = {
            "clf__n_estimators": Integer(50, 200),
            "clf__max_depth": Integer(3, 10)
        }
        layer_det = BayesSearchCV(clf, search_spaces, n_iter=bayes_n_iter, n_points=bayes_n_points,
            n_jobs=-1, scoring='accuracy', return_train_score=False, refit=False, verbose=0)

    xgb_trainer = SupervisedDetector(n_layers(net_type), layer_det, tqdm=False, logger=logger, exp_name=exp_name, outf=xgb_outf, pre_computed=pre_computed)
    xgb_trainer.fit(TrainValLoader(model, ds, batch_size=batch_size), LabelledTrainLoader(model, ds, batch_size=batch_size), ds.adv_test[ds.idxs_train])

    test_scores, output = xgb_trainer.predict(LabelledTestLoader(model, ds, batch_size=batch_size))
    all_output = np.hstack((test_scores, output, np.expand_dims(ds.adv_test[ds.idxs_test], axis=1)))

    with open(f'{xgb_outf}/XGB_net_detector_{exp_name}.pkl', 'wb') as outp:
        pickle.dump(xgb_trainer, outp, pickle.HIGHEST_PROTOCOL)

    np.save(f"{xgb_outf}/XGB_{exp_name}", all_output)

    accuracy = accuracy_score(ds.adv_test[ds.idxs_test], output[:, 0])
    auroc = roc_auc_score(ds.adv_test[ds.idxs_test], -output[:, 1])

    logger.info(f"{exp_name}: ACC = {accuracy}, AUROC = {round(auroc*100, 2)}")

def main():
    parser = argparse.ArgumentParser(
        description='Run supervised adversarial detection experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset and model configuration
    parser.add_argument('--dataset', '--ds', type=str, 
                       choices=['cifar10', 'svhn'], 
                       default='cifar10',
                       help='Dataset to use')
    
    parser.add_argument('--net-type', '--net', type=str, 
                       default='resnet',
                       help='Network architecture type')
    
    parser.add_argument('--adv-type', '--adv', type=str,
                       choices=['DeepFool', 'FGSM', 'BIM', 'CWL2'],
                       default='DeepFool',
                       help='Adversarial attack type')
    
    # Algorithm selection
    parser.add_argument('--algorithm', '--alg', type=str,
                       choices=['knn', 'randomforest', 'adaboost', 'gbdt', 'xgboost', 'lightgbm', 'all'],
                       default='knn',
                       help='Detection algorithm to run')
    
    # Hyperparameter optimization
    parser.add_argument('--bayes-n-points', type=int, default=1,
                       help='Number of points for Bayesian optimization')
    
    parser.add_argument('--bayes-n-iter', type=int, default=40,
                       help='Number of iterations for Bayesian optimization')
    
    parser.add_argument('--pre-computed', action='store_true',
                       help='Use pre-computed hyperparameters (skip optimization)')
    
    # Training configuration
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for data loading')
    
    # Output configuration
    parser.add_argument('--outf', type=str, 
                       default='/kaggle/input/attacked-pth-files',
                       help='Output folder for attacked files')
    
    parser.add_argument('--output-name', type=str, default=None,
                       help='Custom output name (defaults to algorithm name)')
    
    # Logging
    parser.add_argument('--no-logger', action='store_true',
                       help='Disable logging')
    
    # Specific algorithm arguments
    parser.add_argument('--knn-fname', type=str, default='knn',
                       help='Output folder name for KNN')
    
    parser.add_argument('--rf-fname', type=str, default='randomforest',
                       help='Output folder name for Random Forest')
    
    parser.add_argument('--ab-fname', type=str, default='adaboost',
                       help='Output folder name for AdaBoost')
    
    parser.add_argument('--gbdt-fname', type=str, default='gbdt',
                       help='Output folder name for GBDT')
    
    parser.add_argument('--xgb-fname', type=str, default='xgboost',
                       help='Output folder name for XGBoost')
    
    parser.add_argument('--lgbm-fname', type=str, default='lightgbm',
                       help='Output folder name for LightGBM')
    
    args = parser.parse_args()
    
    # Common parameters
    common_params = {
        'net_type': args.net_type,
        'ds_name': args.dataset,
        'adv_type': args.adv_type,
        'outf': args.outf,
        'bayes_n_points': args.bayes_n_points,
        'bayes_n_iter': args.bayes_n_iter,
        'use_logger': not args.no_logger,
        'batch_size': args.batch_size,
        'pre_computed': args.pre_computed
    }
    
    # Algorithm execution
    if args.algorithm == 'knn' or args.algorithm == 'all':
        print(f"Running KNN with dataset={args.dataset}, adv_type={args.adv_type}")
        run_knn(knn_fname=args.knn_fname, **common_params)
    
    if args.algorithm == 'randomforest' or args.algorithm == 'all':
        print(f"Running Random Forest with dataset={args.dataset}, adv_type={args.adv_type}")
        run_randomforest(rf_fname=args.rf_fname, **common_params)
    
    if args.algorithm == 'adaboost' or args.algorithm == 'all':
        print(f"Running AdaBoost with dataset={args.dataset}, adv_type={args.adv_type}")
        run_adaboost(ab_fname=args.ab_fname, **common_params)
    
    if args.algorithm == 'gbdt' or args.algorithm == 'all':
        print(f"Running GBDT with dataset={args.dataset}, adv_type={args.adv_type}")
        run_gbdt(gbdt_fname=args.gbdt_fname, **common_params)
    
    if args.algorithm == 'xgboost' or args.algorithm == 'all':
        print(f"Running XGBoost with dataset={args.dataset}, adv_type={args.adv_type}")
        run_xgboost(xgb_fname=args.xgb_fname, **common_params)
    
    if args.algorithm == 'lightgbm' or args.algorithm == 'all':
        print(f"Running LightGBM with dataset={args.dataset}, adv_type={args.adv_type}")
        run_lightgbm(lgbm_fname=args.lgbm_fname, **common_params)
    
    print("Experiment completed successfully!")

if __name__ == '__main__':
    main()