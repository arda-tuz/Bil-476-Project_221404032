import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from datetime import date, timedelta
import random
from pathlib import Path

class SalesDataset(Dataset):
    """
    PyTorch Dataset for time series sales forecasting.
    
    WINDOWING AND BATCHING STRATEGY:
    This dataset implements a sophisticated sliding window approach for time series:
    
    1. SLIDING WINDOWS: Each sample uses a window of 'timesteps' historical days
       to predict the next 16 days. The window slides through time.
    
    2. RANDOM TIME OFFSETS: During training, we randomly select different starting
       points (n_range) to create diverse training samples from the same store-family.
       This acts as a form of data augmentation.
    
    3. EFFICIENT PIVOTED ACCESS: The data is stored in pivoted format where each
       feature is a matrix with rows=store-family and columns=dates. This allows
       fast sequential access when creating windows.
    
    4. TRAIN/VAL/TEST SPLIT: Different pred_start_dates ensure temporal separation
       between train, validation, and test sets to prevent data leakage.
    """
    
    def __init__(self, pivoted_data, static_df, metadata, 
                 timesteps=200, pred_start_date=None, 
                 n_range=1, day_skip=7, is_train=True,
                 date_filter=None):
        
        self.pivoted_data = pivoted_data
        self.static_df = static_df
        self.metadata = metadata
        
        # Window parameters
        self.timesteps = timesteps  # Historical window size
        self.pred_start_date = pred_start_date  # Start date for predictions
        self.n_range = n_range  # Number of different time offsets for augmentation
        self.day_skip = day_skip  # Days to skip between offsets
        self.is_train = is_train
        self.date_filter = date_filter  # Optional date range filtering
        
        # Get all valid store-family combinations
        self.indices = static_df.index.tolist()
        
        # Filter indices based on data availability in date range
        if date_filter is not None and 'sales' in pivoted_data:
            valid_indices = []
            for store_nbr, family in self.indices:
                try:
                    sales_data = pivoted_data['sales'].loc[(store_nbr, family)]
                    # Only include if there's actual sales data in the range
                    if sales_data[date_filter['start']:date_filter['end']].notna().any():
                        valid_indices.append((store_nbr, family))
                except:
                    pass
            self.indices = valid_indices
            print(f"Filtered to {len(self.indices)} store-family combinations with data in date range")
        
        # Feature categorization for organized processing
        self.numerical_features = [
            'sales', 'onpromotion', 'dcoilwtico', 'transactions',
            'promo_roll_sum_7', 'promo_roll_sum_30',
            'oil_pct_change', 'days_since_last_promo',
            'days_to_holiday', 'days_from_holiday'
        ]
        
        self.binary_features = [
            'is_weekend', 'is_holiday', 'promo_weekend_interaction'
        ]
        
        self.cyclical_features = [
            'month_sin', 'month_cos', 'dow_sin', 'dow_cos'
        ]
        
        self.interaction_features = [
            'store_family_interaction_encoded',
            'onpromo_promo_sum7_interaction_encoded',
            'onpromo_state_interaction_encoded',
            'promo_sum7_state_interaction_encoded'
        ]
        
    def __len__(self):
        """
        Dataset size depends on whether we're training or evaluating.
        Training uses multiple time offsets for augmentation.
        """
        if self.is_train:
            return len(self.indices) * self.n_range
        else:
            return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Get a single sample (one time window) from the dataset.
        
        For training: Randomly selects a time offset for data augmentation
        For validation/test: Uses fixed time offset for consistency
        """
        if self.is_train:
            # Training: use random time offset for augmentation
            actual_idx = idx % len(self.indices)
            time_offset = random.randint(0, self.n_range - 1)
        else:
            # Validation/Test: fixed time offset
            actual_idx = idx
            time_offset = 0
            
        store_nbr, family = self.indices[actual_idx]
        
        # Calculate actual prediction start date with offset
        pred_start = self.pred_start_date - timedelta(days=int(self.day_skip * time_offset))
        
        # Create feature window
        features = self._create_features(store_nbr, family, pred_start)
        
        return features
    
    def _create_features(self, store_nbr, family, pred_start):
        """
        Extract all features for a specific store-family combination and time window.
        
        This is where the WINDOWING STRATEGY is implemented:
        1. Historical window: [pred_start - timesteps, pred_start - 1]
        2. Prediction window: [pred_start, pred_start + 15] (16 days)
        
        The pivoted data structure allows efficient slicing of these windows.
        """
        
        # Define time windows
        hist_start = pred_start - timedelta(days=self.timesteps)
        hist_end = pred_start - timedelta(days=1)
        pred_end = pred_start + timedelta(days=15)  # 16-day prediction horizon
        
        features = {}
        
        # 1. Extract historical numerical features
        for feature in self.numerical_features:
            if feature in self.pivoted_data:
                if feature == 'sales':
                    # Sales history (input for model)
                    hist_data = self._get_date_range_data(
                        self.pivoted_data[feature], store_nbr, family, hist_start, hist_end
                    )
                    features['sales_hist'] = torch.FloatTensor(hist_data)
                    
                    # Target sales (ground truth for training)
                    if self.is_train:
                        target_data = self._get_date_range_data(
                            self.pivoted_data[feature], store_nbr, family, pred_start, pred_end
                        )
                        features['sales_target'] = torch.FloatTensor(target_data)
                    else:
                        features['sales_target'] = torch.zeros(16)
                else:
                    # Other features: get both historical and future values
                    full_data = self._get_date_range_data(
                        self.pivoted_data[feature], store_nbr, family, hist_start, pred_end
                    )
                    features[feature] = torch.FloatTensor(full_data)
        
        # 2. Extract binary features
        for feature in self.binary_features:
            if feature in self.pivoted_data:
                full_data = self._get_date_range_data(
                    self.pivoted_data[feature], store_nbr, family, hist_start, pred_end
                )
                features[feature] = torch.FloatTensor(full_data)
        
        # 3. Extract cyclical features
        for feature in self.cyclical_features:
            if feature in self.pivoted_data:
                full_data = self._get_date_range_data(
                    self.pivoted_data[feature], store_nbr, family, hist_start, pred_end
                )
                features[feature] = torch.FloatTensor(full_data)
        
        # 4. Extract interaction features (categorical, need integer type)
        for feature in self.interaction_features:
            if feature in self.pivoted_data:
                full_data = self._get_date_range_data(
                    self.pivoted_data[feature], store_nbr, family, hist_start, pred_end
                )
                features[feature] = torch.LongTensor(full_data.astype(int))
        
        # 5. Extract static categorical features
        static_features = self._get_static_features(store_nbr, family)
        features['static_cat'] = torch.LongTensor(static_features)
        
        # 6. Weight for loss calculation (uniform in this implementation)
        features['weight'] = torch.FloatTensor([1.0])
        
        # 7. Store metadata for tracking
        features['store_nbr'] = store_nbr
        features['family'] = family
        
        return features
    
    def _get_date_range_data(self, df, store_nbr, family, start_date, end_date):
        """
        Extract data for a specific date range from pivoted DataFrame.
        
        The pivoted structure allows this to be a simple column slice,
        which is much faster than filtering rows in long format.
        """
        try:
            dates = pd.date_range(start_date, end_date)
            # Filter to available dates in the DataFrame
            available_dates = [d for d in dates if d in df.columns]
            
            if len(available_dates) == 0:
                # No data available for this range
                num_days = (end_date - start_date).days + 1
                return np.zeros(num_days, dtype=np.float32)
            
            # Extract data using pivoted structure
            data = df.loc[(store_nbr, family), available_dates].values
            
            # Handle missing dates by padding with zeros
            if len(data) < len(dates):
                full_data = np.zeros(len(dates), dtype=np.float32)
                for i, d in enumerate(dates):
                    if d in available_dates:
                        idx = available_dates.index(d)
                        full_data[i] = data[idx]
                return full_data
            
            return data.astype(np.float32)
        except:
            # Return zeros if any error occurs
            num_days = (end_date - start_date).days + 1
            return np.zeros(num_days, dtype=np.float32)
    
    def _get_static_features(self, store_nbr, family):
        """
        Get static categorical features that don't change over time.
        These are stored separately from time-varying features for efficiency.
        """
        try:
            row = self.static_df.loc[(store_nbr, family)]
            features = []
            
            # Extract encoded categorical features in expected order
            for col in ['city_encoded', 'state_encoded', 'store_type_encoded', 'cluster_encoded']:
                if col in row:
                    features.append(int(row[col]))
                else:
                    features.append(0)
            
            # Add perishable flag (placeholder)
            features.append(0)
            
            # Add encoded store and family IDs
            features.append(int(row.get('store_nbr_encoded', 0)))
            features.append(int(row.get('family_encoded', 0)))
            
            return np.array(features, dtype=np.int64)
        except Exception as e:
            print(f"Warning: Error getting static features for store {store_nbr}, family {family}: {e}")
            return np.zeros(7, dtype=np.int64)


def collate_fn(batch):
    """
    Custom collation function for batching samples.
    
    This function handles the complex structure of our features:
    - Time series data needs proper shape (batch, time, features)
    - Static features remain 2D (batch, features)
    - Metadata is kept as lists
    """
    collated = {}
    
    # Process each feature type appropriately
    for key in batch[0].keys():
        if key in ['store_nbr', 'family']:
            # Keep metadata as lists
            collated[key] = [b[key] for b in batch]
        elif key == 'sales_hist':
            # Shape: (batch, timesteps, 1) for CNN channel format
            collated[key] = torch.stack([b[key] for b in batch]).unsqueeze(-1)
        elif key in ['sales_target', 'weight', 'static_cat']:
            # Keep original shape
            collated[key] = torch.stack([b[key] for b in batch])
        else:
            # Time-varying features
            tensor_list = [b[key] for b in batch]
            stacked = torch.stack(tensor_list)
            
            # Add channel dimension if needed for time series
            if len(stacked.shape) == 2 and stacked.shape[1] > 20:  # Heuristic for time series
                collated[key] = stacked.unsqueeze(-1)
            else:
                collated[key] = stacked
    
    return collated


def create_dataloaders(pivoted_data, static_df, metadata,
                      batch_size=32, timesteps=200, num_workers=4):
    """
    Create train, validation, and test dataloaders.
    
    This function is deprecated - use create_dataloaders_with_test instead
    for proper train/val/test splits.
    """
    
    # Training dataset with multiple time offsets for augmentation
    train_dataset = SalesDataset(
        pivoted_data, static_df, metadata,
        timesteps=timesteps,
        pred_start_date=date(2017, 7, 5),
        n_range=16,  # 16 different time offsets for augmentation
        day_skip=7,
        is_train=True
    )
    
    # Validation dataset with fixed date
    val_dataset = SalesDataset(
        pivoted_data, static_df, metadata,
        timesteps=timesteps,
        pred_start_date=date(2017, 7, 26),
        n_range=1,
        day_skip=1,
        is_train=True
    )
    
    # Test dataset without targets
    test_dataset = SalesDataset(
        pivoted_data, static_df, metadata,
        timesteps=timesteps,
        pred_start_date=date(2017, 8, 16),
        n_range=1,
        day_skip=1,
        is_train=False
    )
    
    # Create DataLoaders with appropriate settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,  # Shuffle for training
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True  # Faster GPU transfer
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for validation
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def create_dataloaders_with_test(pivoted_data, static_df, metadata,
                                batch_size=32, timesteps=200, num_workers=4):
    """
    Create train, val, and test dataloaders with proper temporal splits.
    
    TEMPORAL SPLIT STRATEGY:
    - Train: Uses data up to July 9, 2017 with sliding windows
    - Validation: Predicts July 10-25, 2017
    - Test: Predicts July 26 - August 10, 2017
    
    This ensures no data leakage between splits.
    """
    
    # Training dataset with extensive augmentation
    train_dataset = SalesDataset(
        pivoted_data, static_df, metadata,
        timesteps=timesteps,
        pred_start_date=date(2017, 7, 9),
        n_range=20,  # More augmentation for training
        day_skip=7,
        is_train=True,
        date_filter={'start': date(2016, 1, 1), 'end': date(2017, 7, 9)}
    )
    
    # Validation dataset - fixed future period
    val_dataset = SalesDataset(
        pivoted_data, static_df, metadata,
        timesteps=timesteps,
        pred_start_date=date(2017, 7, 25),
        n_range=1,
        day_skip=1,
        is_train=True
    )
    
    # Test dataset - further future period
    test_dataset = SalesDataset(
        pivoted_data, static_df, metadata,
        timesteps=timesteps,
        pred_start_date=date(2017, 7, 26),
        n_range=1,
        day_skip=1,
        is_train=True
    )

    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def inspect_batch(dataloader):
    """
    Utility function to inspect the structure of a batch.
    Useful for debugging and understanding data flow.
    """
    batch = next(iter(dataloader))
    
    print("Batch contents:")
    print("-" * 50)
    
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key:30s} | Shape: {str(value.shape):20s} | Type: {value.dtype}")
        elif isinstance(value, list):
            print(f"{key:30s} | Length: {len(value)} | Type: list")
        else:
            print(f"{key:30s} | Type: {type(value)}")
    
    return batch