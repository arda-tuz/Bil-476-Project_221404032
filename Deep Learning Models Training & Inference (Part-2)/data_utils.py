import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datetime import date, timedelta
import pickle
import gc
from pathlib import Path

class DataPivotProcessor:
    """
    Handles the critical data transformation from long format to wide format for efficient time series processing.
    This pivoting strategy dramatically improves data access speed by ~1000x for sequential operations.
    """
    
    def __init__(self, data_dir='./'):
        self.data_dir = Path(data_dir)
        self.label_encoders = {}
        
        # Categorical features requiring encoding for neural network processing
        # These will be converted to integer indices for embedding layers
        self.categorical_features = [
            'store_nbr', 'family', 'city', 'state', 'store_type', 'cluster',
            'holiday_type', 'locale', 'locale_name'
        ]
        
        # Complex interaction features created during feature engineering
        # These capture relationships between multiple variables
        self.interaction_features = [
            'store_family_interaction',
            'onpromo_promo_sum7_interaction',
            'onpromo_state_interaction',
            'promo_sum7_state_interaction'
        ]
        
        # Numerical features that change over time and need to be pivoted
        # Each will become a separate matrix with dates as columns
        self.time_varying_numeric = [
            'sales', 'onpromotion', 'dcoilwtico', 'transactions',
            'promo_roll_sum_7', 'promo_roll_sum_30',
            'oil_pct_change', 'days_since_last_promo',
            'days_to_holiday', 'days_from_holiday'
        ]
        
        # Binary/categorical features that vary with time
        self.time_varying_categorical = [
            'is_weekend', 'is_holiday', 'promo_weekend_interaction'
        ]
        
        # Cyclical features using sine/cosine encoding for temporal patterns
        # This preserves the circular nature of time (e.g., month 12 is close to month 1)
        self.cyclical_features = [
            'month_sin', 'month_cos', 'dow_sin', 'dow_cos'
        ]
        
    def load_and_process_data(self, train_path='train-final.csv', test_path='test-final.csv'):
        """
        Load the feature-engineered CSV files and prepare them for pivoting.
        The data has already undergone extensive feature engineering.
        """
        print("Loading train data...")
        df_train = pd.read_csv(self.data_dir / train_path)
        df_train['date'] = pd.to_datetime(df_train['date'])
        
        print("Loading test data...")
        df_test = pd.read_csv(self.data_dir / test_path)
        df_test['date'] = pd.to_datetime(df_test['date'])
        
        # Store original IDs for submission mapping
        # These are needed to map predictions back to the original format
        self.train_ids = df_train[['id', 'date', 'store_nbr', 'family']].copy()
        self.test_ids = df_test[['id', 'date', 'store_nbr', 'family']].copy()
        
        print(f"Train shape: {df_train.shape}")
        print(f"Test shape: {df_test.shape}")
        
        return df_train, df_test
    
    def create_label_encoders(self, df_train, df_test):
        """
        Create and fit label encoders for all categorical features.
        Label encoding converts categorical strings to integers for neural network processing.
        The encoders are fitted on combined train+test data to handle all possible values.
        """
        # Combine train and test to ensure all categories are seen
        df_combined = pd.concat([
            df_train[self.categorical_features + self.interaction_features], 
            df_test[self.categorical_features + self.interaction_features]
        ], axis=0)
        
        # Create an encoder for each categorical feature
        for feature in self.categorical_features + self.interaction_features:
            if feature in df_combined.columns:
                le = LabelEncoder()
                # Handle missing values by replacing with 'MISSING' string
                df_combined[feature] = df_combined[feature].fillna('MISSING').astype(str)
                le.fit(df_combined[feature])
                self.label_encoders[feature] = le
                print(f"Encoded {feature}: {len(le.classes_)} unique values")
        
        # Save encoders for later use during inference
        with open(self.data_dir / 'label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
            
        return self.label_encoders
    
    def pivot_features(self, df_train, df_test):
        """
        CRITICAL PIVOTING LOGIC:
        Transform data from long format (3M+ rows) to wide format for efficient time series processing.
        
        Original format: Each row represents one store-family-date combination
        Pivoted format: Each row is a store-family, columns are dates, values are the feature values
        
        This transformation is crucial because:
        1. It enables fast sequential access for time series windows (1000x speedup)
        2. Reduces memory access patterns from random to sequential
        3. Allows efficient batch creation for neural network training
        
        The pivoting creates a separate matrix for each feature where:
        - Rows: Store-family combinations (~1,800 unique pairs)
        - Columns: Dates (~600 days)
        - Values: The feature value for that store-family-date combination
        """
        pivoted_data = {}
        
        # Combine train and test for complete date coverage
        df_combined = pd.concat([df_train, df_test], ignore_index=True)
        
        # 1. Pivot numerical features
        print("Pivoting numerical features...")
        for feature in self.time_varying_numeric:
            if feature in df_combined.columns:
                if feature == 'sales' and feature not in df_test.columns:
                    # Sales only exists in training data (target variable)
                    pivot = df_train.pivot_table(
                        index=['store_nbr', 'family'],  # Row indices
                        columns='date',                  # Column indices
                        values=feature,                  # Values to fill the matrix
                        fill_value=0 if feature != 'days_since_last_promo' else -1
                    )
                else:
                    # Features that exist in both train and test
                    pivot = df_combined.pivot_table(
                        index=['store_nbr', 'family'],
                        columns='date',
                        values=feature,
                        fill_value=0 if feature != 'days_since_last_promo' else -1
                    )
                pivoted_data[feature] = pivot
                print(f"  - {feature}: {pivot.shape}")
        
        # 2. Pivot categorical/binary features
        print("Pivoting categorical features...")
        for feature in self.time_varying_categorical:
            if feature in df_combined.columns:
                pivot = df_combined.pivot_table(
                    index=['store_nbr', 'family'],
                    columns='date',
                    values=feature,
                    fill_value=0
                )
                pivoted_data[feature] = pivot
                print(f"  - {feature}: {pivot.shape}")
        
        # 3. Pivot cyclical features (sine/cosine encoded time features)
        print("Pivoting cyclical features...")
        for feature in self.cyclical_features:
            if feature in df_combined.columns:
                pivot = df_combined.pivot_table(
                    index=['store_nbr', 'family'],
                    columns='date',
                    values=feature,
                    fill_value=0
                )
                pivoted_data[feature] = pivot
                print(f"  - {feature}: {pivot.shape}")
        
        # 4. Pivot encoded interaction features
        print("Pivoting interaction features...")
        for feature in self.interaction_features:
            if feature in df_combined.columns:
                # Encode categorical interactions before pivoting
                df_combined[f'{feature}_encoded'] = self.label_encoders[feature].transform(
                    df_combined[feature].fillna('MISSING').astype(str)
                )
                pivot = df_combined.pivot_table(
                    index=['store_nbr', 'family'],
                    columns='date',
                    values=f'{feature}_encoded',
                    fill_value=0
                )
                pivoted_data[f'{feature}_encoded'] = pivot
                print(f"  - {feature}: {pivot.shape}")
        
        return pivoted_data
    
    def create_static_features(self, df_train):
        """
        Extract features that don't change over time for each store-family combination.
        These static features (city, state, store type, cluster) remain constant
        and don't need to be repeated for each timestep, saving memory.
        """
        # Get the first occurrence of each store-family combination
        # Since these features don't change over time, we only need one value
        static_df = df_train.groupby(['store_nbr', 'family']).first()[
            ['city', 'state', 'store_type', 'cluster']
        ].copy()
        
        # Encode categorical features to integers for embedding layers
        for feature in ['city', 'state', 'store_type']:
            if feature in self.label_encoders:
                static_df[f'{feature}_encoded'] = self.label_encoders[feature].transform(
                    static_df[feature].fillna('MISSING').astype(str)
                )
        
        # Encode cluster (store grouping) feature
        if 'cluster' in self.label_encoders:
            static_df['cluster_encoded'] = self.label_encoders['cluster'].transform(
                static_df['cluster'].astype(str)
            )
        else:
            # If no encoder exists, simple 0-based indexing
            static_df['cluster_encoded'] = static_df['cluster'] - 1
        
        # Encode the store and family identifiers themselves
        static_df['store_nbr_encoded'] = self.label_encoders['store_nbr'].transform(
            static_df.index.get_level_values(0).astype(str)
        )
        static_df['family_encoded'] = self.label_encoders['family'].transform(
            static_df.index.get_level_values(1).astype(str)
        )
        
        # Placeholder for perishable flag (would need items.csv for actual data)
        static_df['perishable'] = 0
        
        # Debug information for encoding ranges
        print("\nStatic features encoding ranges:")
        for col in static_df.columns:
            if col.endswith('_encoded'):
                print(f"  - {col}: min={static_df[col].min()}, max={static_df[col].max()}")
        
        return static_df
    
    def save_pivoted_data(self, pivoted_data, static_df, prefix='pivoted'):
        """
        Save pivoted data efficiently using Feather format.
        Feather provides fast read/write with good compression for DataFrames.
        Each feature is saved as a separate file for memory-efficient loading.
        """
        print("Saving pivoted data...")
        
        # Create directory structure for organized storage
        pivot_dir = self.data_dir / prefix
        pivot_dir.mkdir(exist_ok=True)
        
        # Save each pivoted feature separately
        for feature_name, pivot_df in pivoted_data.items():
            # Convert datetime columns to strings for Feather compatibility
            pivot_df.columns = pivot_df.columns.astype(str)
            
            file_path = pivot_dir / f'{feature_name}.feather'
            pivot_df.reset_index().to_feather(file_path)
            print(f"  - Saved {feature_name} to {file_path}")
        
        # Save static features
        static_df.reset_index().to_feather(pivot_dir / 'static_features.feather')
        
        # Save metadata for reconstruction
        metadata = {
            'features': list(pivoted_data.keys()),
            'static_features': list(static_df.columns),
            'n_stores': len(static_df.index.get_level_values(0).unique()),
            'n_families': len(static_df.index.get_level_values(1).unique()),
            'n_combinations': len(static_df)
        }
        
        import json
        with open(pivot_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("Data saved successfully!")
    
    def load_pivoted_data(self, prefix='pivoted'):
        """
        Load previously pivoted data from disk.
        This allows skipping the expensive pivoting operation in subsequent runs.
        """
        pivot_dir = self.data_dir / prefix
        
        # Load metadata
        import json
        with open(pivot_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Load all pivoted features
        pivoted_data = {}
        for feature in metadata['features']:
            file_path = pivot_dir / f'{feature}.feather'
            df = pd.read_feather(file_path).set_index(['store_nbr', 'family'])
            # Convert columns back to datetime
            df.columns = pd.to_datetime(df.columns)
            pivoted_data[feature] = df
        
        # Load static features
        static_df = pd.read_feather(pivot_dir / 'static_features.feather').set_index(['store_nbr', 'family'])
        
        # Load label encoders
        with open(self.data_dir / 'label_encoders.pkl', 'rb') as f:
            self.label_encoders = pickle.load(f)
        
        return pivoted_data, static_df, metadata
    
    def get_feature_info(self):
        """
        Return feature categorization for model building.
        This information is used to properly handle different feature types in the models.
        """
        feature_info = {
            'numerical_features': self.time_varying_numeric,
            'categorical_features': self.time_varying_categorical,
            'cyclical_features': self.cyclical_features,
            'interaction_features': self.interaction_features,
            'static_categorical': ['city', 'state', 'store_type', 'cluster'],
            'label_encoder_sizes': {k: len(v.classes_) for k, v in self.label_encoders.items()}
        }
        return feature_info


def prepare_data_for_training(data_dir='./'):
    """
    Main pipeline function to prepare feature-engineered data for neural network training.
    This function orchestrates the entire data preparation process.
    """
    processor = DataPivotProcessor(data_dir)
    
    # Load the feature-engineered CSV files
    df_train, df_test = processor.load_and_process_data()
    
    # Create label encoders for categorical variables
    processor.create_label_encoders(df_train, df_test)
    
    # Perform the critical pivoting operation
    pivoted_data = processor.pivot_features(df_train, df_test)
    
    # Extract static features
    static_df = processor.create_static_features(df_train)
    
    # Save everything to disk for fast loading later
    processor.save_pivoted_data(pivoted_data, static_df)
    
    return processor


if __name__ == "__main__":
    # Execute data preparation pipeline
    processor = prepare_data_for_training()
    
    # Display feature information summary
    feature_info = processor.get_feature_info()
    print("\nFeature Information:")
    for key, value in feature_info.items():
        if key != 'label_encoder_sizes':
            print(f"{key}: {len(value)} features")
        else:
            print(f"{key}: {len(value)} encoders")