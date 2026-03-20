import numpy as np
import torch

from .model import extract_activations, activations_from_loader

from mahalanobis import data_loader

class Datasets():
    def __init__(self, ds_name, in_transform, net_type, adv_type, outf):
        self.ds_name = ds_name
        self.net_type = net_type
        self.adv_type = adv_type

        self.train_loader, _ = data_loader.getTargetDataSet(ds_name, 100, in_transform, './data')

        fname = f"{net_type}_{ds_name}_{adv_type}.pth"

        test_data = torch.load(f'{outf}/clean_data_{fname}')
        new_size = 100 * (len(test_data) // 100)
        test_data = test_data[:new_size]

        noisy_data = torch.load(f'{outf}/noisy_data_{fname}')[:new_size]
        adv_data = torch.load(f'{outf}/adv_data_{fname}')[:new_size]
        targets = torch.load(f'{outf}/label_{fname}').cpu().numpy()[:new_size]

        self.X_test = torch.cat([adv_data, test_data, noisy_data]).cuda()
        self.y_test = np.tile(targets, 3)
        self.adv_test = np.array([0] * len(adv_data) + [1] * (len(test_data) + len(noisy_data)))

        # Splits
        p_size = len(test_data)
        p_split = int(p_size*0.1)
        idxs_trainval = np.concatenate([
            np.arange(p_split),
            np.arange(p_size, p_size+p_split),
            np.arange(2*p_size, 2*p_size+p_split)
        ])

        self.idxs_test = np.delete(np.arange(len(self.X_test)), idxs_trainval)

        pivot = int(len(idxs_trainval) / 6)
        self.idxs_train = np.concatenate([idxs_trainval[:pivot], idxs_trainval[2*pivot:3*pivot], idxs_trainval[4*pivot:5*pivot]])
        self.idxs_val = np.concatenate([idxs_trainval[pivot:2*pivot], idxs_trainval[3*pivot:4*pivot], idxs_trainval[5*pivot:]])
        self._setup_clean_only_test_set(p_size, p_split)

    def _setup_clean_only_test_set(self, p_size, p_split):
        """Create clean-only test set from the clean portion of the mixed data"""
        # Clean data is in the middle third: indices [p_size : 2*p_size]
        clean_start = p_size
        clean_end = 2 * p_size
        
        # Extract only the clean portion
        self.X_test_clean_only = self.X_test[clean_start:clean_end]
        self.y_test_clean_only = self.y_test[clean_start:clean_end]
        
        # For clean-only test set, all samples are clean (label = 1)
        self.adv_test_clean_only = np.ones(len(self.X_test_clean_only))
        
        # Create clean-only test indices using same proportion as mixed data
        # 10% was used for trainval in original, so use same for clean-only
        trainval_clean_span = p_split  # Same as original p_split
        
        # Clean-only indices (90% of clean data for testing, 10% reserved)
        self.idxs_test_clean_only = np.arange(trainval_clean_span, p_size)
        
        # For validation on clean-only (if needed)
        self.idxs_val_clean_only = np.arange(trainval_clean_span // 2, trainval_clean_span)
        self.idxs_train_clean_only = np.arange(trainval_clean_span // 2)
        
        # Add GA validation indices for clean-only (for compatibility with GA version)
        ga_span = trainval_clean_span // 2  # Half of trainval span
        self.idxs_ga_val_clean_only = np.arange(trainval_clean_span // 2, trainval_clean_span)

class TrainValLoader:
    def __init__(self, model, ds, batch_size=100):
        self.ds = ds
        self.model = model
        self.batch_size = batch_size
        
    def __call__(self, layer_idx):
        X_train, y_train = activations_from_loader(self.model, layer_idx, self.ds.train_loader)
        adv_train = np.repeat(1, len(X_train))

        X_valid = self.ds.X_test[self.ds.idxs_val]
        acts_valid, y_valid = [], []

        for batch in torch.split(X_valid, self.batch_size):
            a_temp, y_temp = extract_activations(batch, self.model, layer_idx, return_pred=True)
            acts_valid.append(a_temp.cpu().numpy())
            y_valid.append(y_temp.cpu().numpy())

        X_valid = np.concatenate(acts_valid)
        y_valid = np.concatenate(y_valid)
        adv_valid = self.ds.adv_test[self.ds.idxs_val]

        return X_train, X_valid, y_train, y_valid, adv_train, adv_valid

class LabelledTrainLoader:
    def __init__(self, model, ds, batch_size=100):
        self.ds = ds
        self.model = model
        self.batch_size = batch_size
        
    def __call__(self, layer_idx):
        X_test = self.ds.X_test[self.ds.idxs_train]

        acts, y_test = [], []
        for batch in torch.split(X_test, self.batch_size):
            a_temp, y_temp = extract_activations(batch, self.model, layer_idx, return_pred=True)
            acts.append(a_temp.cpu().numpy())
            y_test.append(y_temp.cpu().numpy())
        acts = np.concatenate(acts)
        y_test = np.concatenate(y_test)

        return acts, y_test

class LabelledValLoader:
    def __init__(self, model, ds, batch_size=100):
        self.ds = ds
        self.model = model
        self.batch_size = batch_size
        
    def __call__(self, layer_idx):
        X_test = self.ds.X_test[self.ds.idxs_val]

        acts, y_test = [], []
        for batch in torch.split(X_test, self.batch_size):
            a_temp, y_temp = extract_activations(batch, self.model, layer_idx, return_pred=True)
            acts.append(a_temp.cpu().numpy())
            y_test.append(y_temp.cpu().numpy())
        acts = np.concatenate(acts)
        y_test = np.concatenate(y_test)

        return acts, y_test

class LabelledTestLoader:
    def __init__(self, model, ds, batch_size=100):
        self.ds = ds
        self.model = model
        self.batch_size = batch_size
        
    def __call__(self, layer_idx):
        X_test = self.ds.X_test[self.ds.idxs_test]

        acts, y_test = [], []
        for batch in torch.split(X_test, self.batch_size):
            a_temp, y_temp = extract_activations(batch, self.model, layer_idx, return_pred=True)
            acts.append(a_temp.cpu().numpy())
            y_test.append(y_temp.cpu().numpy())
        acts = np.concatenate(acts)
        y_test = np.concatenate(y_test)

        return acts, y_test

def idxs_train_val_test(data_size):
    p_size = data_size // 3
    p_split = int(p_size*0.1)
    idxs_trainval = np.concatenate([np.arange(p_split), np.arange(p_size, p_size+p_split),
        np.arange(2*p_size, 2*p_size+p_split)])

    idxs_test = np.delete(np.arange(data_size), idxs_trainval)

    pivot = int(len(idxs_trainval) / 6)
    idxs_train = np.concatenate([idxs_trainval[:pivot], idxs_trainval[2*pivot:3*pivot], idxs_trainval[4*pivot:5*pivot]])
    idxs_val = np.concatenate([idxs_trainval[pivot:2*pivot], idxs_trainval[3*pivot:4*pivot], idxs_trainval[5*pivot:]])

    return idxs_train, idxs_val, idxs_test

def idxs_train_val_test_ga(data_size):
    """Updated to match DatasetsGA 4-way split structure"""
    p_size = data_size // 3  # Still 3 parts: adv, clean, noisy
    
    # Use same split logic as DatasetsGA
    trainval_frac = 0.1
    ga_frac = 0.1
    
    train_val_span = int(p_size * trainval_frac)
    ga_span = int(p_size * ga_frac)
    
    # Training + Validation indices (10% total)
    idxs_trainval = np.concatenate([
        np.arange(train_val_span),
        np.arange(p_size, p_size + train_val_span),
        np.arange(2*p_size, 2*p_size + train_val_span)
    ])
    
    # GA Validation indices (10% total) 
    idxs_ga_val = np.concatenate([
        np.arange(train_val_span, train_val_span + ga_span),
        np.arange(p_size + train_val_span, p_size + train_val_span + ga_span),
        np.arange(2*p_size + train_val_span, 2*p_size + train_val_span + ga_span)
    ])
    
    # Test indices (remaining 80%)
    all_used_idxs = np.concatenate([idxs_trainval, idxs_ga_val])
    idxs_test = np.delete(np.arange(p_size * 3), all_used_idxs)
    
    # Split trainval into train and validation
    pivot = int(len(idxs_trainval) / 6)
    idxs_train = np.concatenate([
        idxs_trainval[:pivot],
        idxs_trainval[2*pivot:3*pivot],
        idxs_trainval[4*pivot:5*pivot]
    ])
    idxs_val = np.concatenate([
        idxs_trainval[pivot:2*pivot],
        idxs_trainval[3*pivot:4*pivot],
        idxs_trainval[5*pivot:]
    ])
    
    # Return 4 values to match DatasetsGA
    return idxs_train, idxs_val, idxs_ga_val, idxs_test

class LabelledGAValLoader:
    def __init__(self, model, ds, batch_size=100):
        self.ds = ds
        self.model = model
        self.batch_size = batch_size
        
    def __call__(self, layer_idx):
        # Fix: Use the GA validation indices to slice X_test, not use idxs_ga_val directly
        X_test = self.ds.X_test[self.ds.idxs_ga_val]  # This gives us the actual data

        acts, y_test = [], []
        for batch in torch.split(X_test, self.batch_size):  # Now split the tensor data
            a_temp, y_temp = extract_activations(batch, self.model, layer_idx, return_pred=True)
            acts.append(a_temp.cpu().numpy())
            y_test.append(y_temp.cpu().numpy())
        acts = np.concatenate(acts)
        y_test = np.concatenate(y_test)

        return acts, y_test

class DatasetsGA(Datasets):
    def __init__(self, ds_name, in_transform, net_type, adv_type, outf):
        # Call parent constructor to set up all basic data loading
        super().__init__(ds_name, in_transform, net_type, adv_type, outf)
        
        # Override splits with 4-way GA split
        p_size = len(self.X_test) // 3
        self.idxs_train, self.idxs_val, self.idxs_ga_val, self.idxs_test = self._train_val_ga_test_split(p_size)
        
        # Override clean-only test set with GA-specific splits
        self._setup_clean_only_test_set_ga(p_size)
        
    def _train_val_ga_test_split(self, split_size, trainval_frac=0.1, ga_frac=0.1):
        # 10% for training+validation (same as original)
        train_val_span = int(split_size * trainval_frac)
        
        # 10% for GA validation
        ga_span = int(split_size * ga_frac)
        
        # Training + Validation indices (10% total)
        idxs_trainval = np.concatenate([
            np.arange(train_val_span),
            np.arange(split_size, split_size + train_val_span),
            np.arange(2*split_size, 2*split_size + train_val_span)
        ])
        
        # GA Validation indices (10% total)
        idxs_ga_val = np.concatenate([
            np.arange(train_val_span, train_val_span + ga_span),
            np.arange(split_size + train_val_span, split_size + train_val_span + ga_span),
            np.arange(2*split_size + train_val_span, 2*split_size + train_val_span + ga_span)
        ])
        
        # Test indices (remaining 80%)
        all_used_idxs = np.concatenate([idxs_trainval, idxs_ga_val])
        idxs_test = np.delete(np.arange(split_size * 3), all_used_idxs)
        
        # Split trainval into train and validation (same as original)
        pivot = int(len(idxs_trainval) / 6)
        idxs_train = np.concatenate([
            idxs_trainval[:pivot],
            idxs_trainval[2*pivot:3*pivot],
            idxs_trainval[4*pivot:5*pivot]
        ])
        idxs_val = np.concatenate([
            idxs_trainval[pivot:2*pivot],
            idxs_trainval[3*pivot:4*pivot],
            idxs_trainval[5*pivot:]
        ])
        
        return idxs_train, idxs_val, idxs_ga_val, idxs_test

    def _setup_clean_only_test_set_ga(self, p_size):
        """Create clean-only test set from the clean portion with GA-specific splits"""
        # Clean data is in the middle third: indices [p_size : 2*p_size]
        clean_start = p_size
        clean_end = 2 * p_size
        
        # Extract only the clean portion (override parent's setup)
        self.X_test_clean_only = self.X_test[clean_start:clean_end]
        self.y_test_clean_only = self.y_test[clean_start:clean_end]
        
        # For clean-only test set, all samples are clean (label = 1)
        self.adv_test_clean_only = np.ones(len(self.X_test_clean_only))
        
        # Create clean-only test indices using GA split proportions
        total_clean = len(self.X_test_clean_only)
        trainval_frac = 0.1
        ga_frac = 0.1
        
        train_val_span = int(total_clean * trainval_frac)
        ga_span = int(total_clean * ga_frac)
        
        # Clean-only indices for GA
        self.idxs_test_clean_only = np.arange(train_val_span + ga_span, total_clean)
        self.idxs_ga_val_clean_only = np.arange(train_val_span, train_val_span + ga_span)
        self.idxs_val_clean_only = np.arange(train_val_span // 2, train_val_span)
        self.idxs_train_clean_only = np.arange(train_val_span // 2)

class LabelledCleanOnlyTestLoader:
    def __init__(self, model, ds, batch_size=100):
        self.ds = ds
        self.model = model
        self.batch_size = batch_size
        
    def __call__(self, layer_idx):
        X_test = self.ds.X_test_clean_only[self.ds.idxs_test_clean_only]

        acts, y_test = [], []
        for batch in torch.split(X_test, self.batch_size):
            a_temp, y_temp = extract_activations(batch, self.model, layer_idx, return_pred=True)
            acts.append(a_temp.cpu().numpy())
            y_test.append(y_temp.cpu().numpy())
        acts = np.concatenate(acts)
        y_test = np.concatenate(y_test)

        return acts, y_test

class LabelledCleanOnlyGAValLoader:
    def __init__(self, model, ds, batch_size=100):
        self.ds = ds
        self.model = model
        self.batch_size = batch_size
        
    def __call__(self, layer_idx):
        X_test = self.ds.X_test_clean_only[self.ds.idxs_ga_val_clean_only]

        acts, y_test = [], []
        for batch in torch.split(X_test, self.batch_size):
            a_temp, y_temp = extract_activations(batch, self.model, layer_idx, return_pred=True)
            acts.append(a_temp.cpu().numpy())
            y_test.append(y_temp.cpu().numpy())
        acts = np.concatenate(acts)
        y_test = np.concatenate(y_test)

        return acts, y_test