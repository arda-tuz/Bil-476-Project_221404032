import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BaseTemporalModel(nn.Module):
    """
    Base class for all temporal models providing common embedding functionality.
    
    EMBEDDING LAYER HANDLING:
    The embedding layers convert categorical integer indices to dense vector representations.
    This is crucial for neural networks to process categorical data effectively.
    
    Each categorical feature gets its own embedding layer with dimensions chosen based on:
    - Cardinality (number of unique values): Higher cardinality = larger embedding
    - Feature importance: More important features get larger embeddings
    - Empirical best practices: dim = min(50, cardinality/2) is a common heuristic
    """
    
    def __init__(self, feature_info, embedding_dims=None):
        super().__init__()
        
        self.feature_info = feature_info
        
        # Default embedding dimensions based on feature cardinality and importance
        # These dimensions were chosen through experimentation and domain knowledge
        if embedding_dims is None:
            embedding_dims = {
                'city': 8,           # Medium cardinality location feature
                'state': 8,          # Few states in Ecuador
                'store_type': 4,     # Only a few store types
                'cluster': 4,        # Store clustering feature
                'store_nbr': 8,      # 54 stores
                'family': 8,         # Product families
                'store_family_interaction': 16,  # High cardinality interaction
                'onpromo_promo_sum7_interaction': 8,
                'onpromo_state_interaction': 8,
                'promo_sum7_state_interaction': 8
            }
        
        # Create embedding layers for each categorical feature
        # Embeddings map sparse one-hot encoded categories to dense vectors
        self.embeddings = nn.ModuleDict()
        
        # Static categorical embeddings (features that don't change over time)
        for feature in ['city', 'state', 'store_type', 'cluster', 'store_nbr', 'family']:
            if feature in feature_info['label_encoder_sizes']:
                num_classes = feature_info['label_encoder_sizes'][feature]
                embed_dim = embedding_dims.get(feature, 8)
                # padding_idx=0 reserves index 0 for missing/unknown values
                self.embeddings[f'{feature}_encoded'] = nn.Embedding(
                    num_embeddings=num_classes + 1,  # +1 for potential out-of-vocabulary
                    embedding_dim=embed_dim,
                    padding_idx=0  # Index 0 represents padding/missing values
                )
                print(f"Created embedding for {feature}_encoded: {num_classes + 1} x {embed_dim}")
        
        # Interaction feature embeddings (complex categorical features)
        for feature in feature_info['interaction_features']:
            if feature in feature_info['label_encoder_sizes']:
                num_classes = feature_info['label_encoder_sizes'][feature]
                embed_dim = embedding_dims.get(feature, 8)
                # Large embedding table for high-cardinality interactions
                self.embeddings[f'{feature}_encoded'] = nn.Embedding(
                    num_embeddings=max(num_classes + 1, 40000),  # Ensure sufficient capacity
                    embedding_dim=embed_dim,
                    padding_idx=0
                )
                print(f"Created embedding for {feature}_encoded: {max(num_classes + 1, 40000)} x {embed_dim}")
    
    def get_embedding_output_dim(self):
        """Calculate total dimension after concatenating all embeddings"""
        total_dim = 0
        for name, embedding in self.embeddings.items():
            total_dim += embedding.embedding_dim
        return total_dim
    
    def get_static_embedding_dim(self):
        """Calculate dimension for static features only (used in model architectures)"""
        static_dim = 0
        for feature in ['city', 'state', 'store_type', 'cluster', 'store_nbr', 'family']:
            feature_key = f'{feature}_encoded'
            if feature_key in self.embeddings:
                static_dim += self.embeddings[feature_key].embedding_dim
        return static_dim


class TemporalCnn(BaseTemporalModel):
    """
    1D CNN MODEL ARCHITECTURE:
    
    This model uses dilated temporal convolutions to capture patterns at multiple time scales.
    CNNs are effective for time series because they can:
    1. Process sequences in parallel (faster than RNNs)
    2. Capture local temporal patterns through convolutions
    3. Expand receptive field exponentially with dilated convolutions
    
    Architecture details:
    - 5 conv layers with increasing dilation rates [1, 2, 4, 8, 16]
    - This creates an exponentially growing receptive field (~31 timesteps)
    - Combines early (local) and late (global) features for better representations
    - Uses residual connections to improve gradient flow
    
    PREDICTION APPROACH: ONE-SHOT
    This model predicts all 16 future days in a single forward pass,
    avoiding error accumulation from autoregressive generation.
    """
    
    def __init__(self, feature_info, timesteps=200, latent_dim=32, dropout=0.25):
        super().__init__(feature_info)
        
        self.timesteps = timesteps
        self.latent_dim = latent_dim
        self.dropout_rate = dropout
        
        # Calculate number of input channels for CNN
        # Each temporal feature becomes a channel in the 1D convolution
        self.num_time_features = (
            1 +  # sales_hist
            len(['onpromotion', 'dcoilwtico', 'transactions',
                 'promo_roll_sum_7', 'promo_roll_sum_30',
                 'oil_pct_change', 'days_since_last_promo',
                 'days_to_holiday', 'days_from_holiday']) +
            len(['is_weekend', 'is_holiday', 'promo_weekend_interaction']) +
            len(['month_sin', 'month_cos', 'dow_sin', 'dow_cos'])
        )
        
        # Temporal convolutional layers with exponentially increasing dilation
        # Dilation allows the network to have a large receptive field with fewer parameters
        self.conv1 = nn.Conv1d(self.num_time_features, latent_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(latent_dim, latent_dim, kernel_size=3, dilation=2, padding=2)
        self.conv3 = nn.Conv1d(latent_dim, latent_dim, kernel_size=3, dilation=4, padding=4)
        self.conv4 = nn.Conv1d(latent_dim, latent_dim, kernel_size=3, dilation=8, padding=8)
        self.conv5 = nn.Conv1d(latent_dim, latent_dim, kernel_size=3, dilation=16, padding=16)
        
        # Combine early and late features for multi-scale representation
        self.conv_out = nn.Conv1d(latent_dim * 2, 16, kernel_size=1)
        
        # Process global temporal features
        self.global_fc = nn.Linear(timesteps * 16, 128)
        
        # Process static categorical features
        self.static_fc = nn.Linear(self.get_static_embedding_dim() + 1, 64)  # +1 for perishable flag
        
        # Process future features (known information about next 16 days)
        future_features = 16 * 10  # Approximate number of future features
        self.future_fc = nn.Linear(future_features, 64)
        
        # Final layers combine all feature types
        total_features = 128 + 64 + 64  # global + static + future
        self.fc1 = nn.Linear(total_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 16)  # Output 16 days of predictions at once
        
        # Regularization layers
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(latent_dim)
        self.batch_norm2 = nn.BatchNorm1d(256)
    
    def forward(self, batch):
        batch_size = batch['sales_hist'].size(0)
        
        # 1. Process temporal features through dilated CNN
        temporal_features = []
        
        # Collect all temporal features into channels
        temporal_features.append(batch['sales_hist'])  # Historical sales
        
        # Add other temporal features
        for feature in ['onpromotion', 'dcoilwtico', 'transactions',
                       'promo_roll_sum_7', 'promo_roll_sum_30',
                       'oil_pct_change', 'days_since_last_promo',
                       'days_to_holiday', 'days_from_holiday',
                       'is_weekend', 'is_holiday', 'promo_weekend_interaction',
                       'month_sin', 'month_cos', 'dow_sin', 'dow_cos']:
            if feature in batch:
                # Extract historical portion
                feat = batch[feature][:, :self.timesteps]
                if len(feat.shape) == 2:
                    feat = feat.unsqueeze(-1)
                temporal_features.append(feat)
        
        # Concatenate features and transpose for CNN (batch, channels, time)
        x = torch.cat(temporal_features, dim=2)
        x = x.transpose(1, 2)
        
        # Apply dilated convolutions with residual connections
        c1 = F.relu(self.conv1(x))
        c1 = self.batch_norm1(c1)
        
        c2 = F.relu(self.conv2(c1))
        c3 = F.relu(self.conv3(c2))
        c4 = F.relu(self.conv4(c3))
        c5 = F.relu(self.conv5(c4))
        
        # Combine early (local) and late (global) features
        conv_combined = torch.cat([c1, c5], dim=1)
        conv_out = F.relu(self.conv_out(conv_combined))
        conv_out = self.dropout(conv_out)
        
        # Flatten temporal features
        conv_flat = conv_out.view(batch_size, -1)
        global_features = F.relu(self.global_fc(conv_flat))
        global_features = self.dropout(global_features)
        
        # 2. Process static categorical features through embeddings
        static_embeddings = []
        static_cat = batch['static_cat']
        
        # Debug information for first batch
        if not hasattr(self, '_debug_printed'):
            print(f"\nStatic features shape: {static_cat.shape}")
            print(f"Static features sample: {static_cat[0].tolist()}")
            self._debug_printed = True
        
        # Embed each static categorical feature
        feature_names = ['city_encoded', 'state_encoded', 'store_type_encoded', 
                        'cluster_encoded', 'perishable', 'store_nbr_encoded', 'family_encoded']
        
        for idx, feature in enumerate(feature_names[:4]):  # First 4 are embeddings
            if feature in self.embeddings:
                try:
                    # Clamp values to valid embedding range
                    max_idx = self.embeddings[feature].num_embeddings - 1
                    values = torch.clamp(static_cat[:, idx], 0, max_idx)
                    embedded = self.embeddings[feature](values)
                    static_embeddings.append(embedded)
                except Exception as e:
                    print(f"Error embedding {feature} at idx {idx}: {e}")
                    print(f"Values: min={static_cat[:, idx].min()}, max={static_cat[:, idx].max()}")
                    print(f"Embedding size: {self.embeddings[feature].num_embeddings}")
                    raise
        
        # Add perishable flag (binary feature)
        perishable = static_cat[:, 4].unsqueeze(1).float()
        static_embeddings.append(perishable)
        
        # Add store and family embeddings
        for idx, feature in enumerate(['store_nbr_encoded', 'family_encoded'], start=5):
            if feature in self.embeddings:
                try:
                    max_idx = self.embeddings[feature].num_embeddings - 1
                    values = torch.clamp(static_cat[:, idx], 0, max_idx)
                    embedded = self.embeddings[feature](values)
                    static_embeddings.append(embedded)
                except Exception as e:
                    print(f"Error embedding {feature} at idx {idx}: {e}")
                    print(f"Values: min={static_cat[:, idx].min()}, max={static_cat[:, idx].max()}")
                    print(f"Embedding size: {self.embeddings[feature].num_embeddings}")
                    raise
        
        static_features = torch.cat(static_embeddings, dim=1)
        static_features = F.relu(self.static_fc(static_features))
        
        # 3. Process future features (known information about prediction period)
        future_features_list = []
        
        for feature in ['onpromotion', 'dcoilwtico', 'promo_roll_sum_7',
                       'is_weekend', 'is_holiday', 'month_sin', 'month_cos',
                       'dow_sin', 'dow_cos', 'days_to_holiday']:
            if feature in batch:
                # Extract future portion (last 16 values)
                future_feat = batch[feature][:, -16:]
                if len(future_feat.shape) == 3:
                    future_feat = future_feat.squeeze(-1)
                future_features_list.append(future_feat)
        
        if future_features_list:
            future_concat = torch.cat(future_features_list, dim=1)
            future_features = F.relu(self.future_fc(future_concat))
        else:
            future_features = torch.zeros(batch_size, 64).to(batch['sales_hist'].device)
        
        # 4. Combine all feature types
        combined = torch.cat([global_features, static_features, future_features], dim=1)
        
        # Final layers with non-linearity and regularization
        x = F.relu(self.fc1(combined))
        x = self.batch_norm2(x)
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # ONE-SHOT OUTPUT: All 16 predictions at once
        # ReLU ensures non-negative sales predictions
        output = F.relu(self.fc3(x))
        
        return output


class AutoregressiveEncoderDecoderGruBased(BaseTemporalModel):
    """
    AUTOREGRESSIVE ENCODER-DECODER GRU ARCHITECTURE:
    
    This is a sequence-to-sequence model with attention mechanism that performs
    AUTOREGRESSIVE prediction - generating one day at a time, using each
    prediction as input for the next step.
    
    Architecture consists of:
    
    1. ENCODER: Bidirectional GRU that processes historical data
       - Bidirectional processing captures both forward and backward temporal dependencies
       - 2 layers deep for hierarchical feature learning
       - Note: Actually uses LSTM internally (historical naming inconsistency)
       
    2. ATTENTION: Multi-head attention mechanism
       - Allows the decoder to focus on relevant parts of history
       - 8 attention heads for diverse attention patterns
       
    3. DECODER: GRU/LSTM that generates predictions autoregressively
       - Generates ONE prediction at a time
       - Each prediction feeds into the next timestep
       - This allows the model to adjust based on its own predictions
       - Risk: Can accumulate errors over the 16-day horizon
    
    PREDICTION APPROACH: AUTOREGRESSIVE
    This model generates predictions sequentially, where each day's prediction
    becomes part of the input for predicting the next day. This is the key
    characteristic that makes it autoregressive.
    """
    
    def __init__(self, feature_info, timesteps=200, latent_dim=100, dropout=0.25):
        super().__init__(feature_info)
        
        self.timesteps = timesteps
        self.latent_dim = latent_dim
        self.dropout_rate = dropout
        
        # Count temporal features for input dimension
        self.num_temporal_features = 17  # Based on actual feature count
        
        # Encoder: Bidirectional LSTM for processing historical sequences
        # Note: Despite the class name mentioning GRU, we use LSTM for better performance
        self.encoder_lstm = nn.LSTM(
            input_size=self.num_temporal_features,
            hidden_size=latent_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0,
            bidirectional=True  # Process sequence in both directions
        )
        
        # Attention mechanism for encoder-decoder connection
        # Allows decoder to selectively focus on encoder outputs
        self.attention = nn.MultiheadAttention(
            embed_dim=latent_dim * 2,  # *2 for bidirectional
            num_heads=8,
            dropout=dropout
        )
        
        # Process attention context
        self.context_fc = nn.Linear(latent_dim * 2, latent_dim)
        
        # Decoder: LSTM for autoregressive generation
        self.decoder_lstm = nn.LSTM(
            input_size=self.num_temporal_features,
            hidden_size=latent_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )
        
        # Output projection layers - NOTE: outputs single day prediction
        self.output_fc1 = nn.Linear(latent_dim, 64)
        self.output_fc2 = nn.Linear(64, 1)  # CRITICAL: Outputs 1 day, not 16
        
        self.dropout = nn.Dropout(dropout)
        
    def prepare_temporal_features(self, batch, start_idx, end_idx):
        """
        Prepare temporal features for a specific time range.
        This is used to extract features for both encoding and decoding phases.
        """
        features = []
        
        # Collect all temporal features
        for feature in ['sales_hist', 'onpromotion', 'dcoilwtico', 'transactions',
                       'promo_roll_sum_7', 'promo_roll_sum_30',
                       'oil_pct_change', 'days_since_last_promo',
                       'days_to_holiday', 'days_from_holiday',
                       'is_weekend', 'is_holiday', 'promo_weekend_interaction',
                       'month_sin', 'month_cos', 'dow_sin', 'dow_cos']:
            if feature in batch:
                if feature == 'sales_hist':
                    feat = batch[feature][:, start_idx:end_idx]
                else:
                    feat = batch[feature][:, start_idx:end_idx]
                
                if len(feat.shape) == 2:
                    feat = feat.unsqueeze(-1)
                features.append(feat)
        
        return torch.cat(features, dim=2)
    
    def forward(self, batch):
        batch_size = batch['sales_hist'].size(0)
        device = batch['sales_hist'].device
        
        # Prepare encoder input (historical features)
        encoder_features = self.prepare_temporal_features(batch, 0, self.timesteps)
        
        # Verify feature dimension
        if encoder_features.size(2) != self.num_temporal_features:
            raise ValueError(f"Expected {self.num_temporal_features} features, got {encoder_features.size(2)}")
        
        # ENCODING PHASE: Process historical sequence
        encoder_output, (hidden, cell) = self.encoder_lstm(encoder_features)
        
        # Apply attention to encoder outputs
        # This helps the decoder focus on relevant historical patterns
        encoder_output_t = encoder_output.transpose(0, 1)  # (seq, batch, features)
        attended_output, _ = self.attention(
            encoder_output_t[-1:],  # Query: last encoder state
            encoder_output_t,        # Keys: all encoder states
            encoder_output_t         # Values: all encoder states
        )
        attended_output = attended_output.transpose(0, 1)
        
        # Process attention context
        context = self.context_fc(attended_output.squeeze(1))
        
        # Initialize decoder hidden state from encoder
        # Combine forward and backward hidden states
        h_fwd = hidden[-2]  # Last layer, forward
        h_bwd = hidden[-1]  # Last layer, backward
        decoder_hidden = (h_fwd + h_bwd).unsqueeze(0).repeat(2, 1, 1)
        
        c_fwd = cell[-2]
        c_bwd = cell[-1]
        decoder_cell = (c_fwd + c_bwd).unsqueeze(0).repeat(2, 1, 1)
        
        # AUTOREGRESSIVE DECODING PHASE:
        # Generate predictions one day at a time, using previous prediction as input
        outputs = []
        last_sales = batch['sales_hist'][:, -1]  # Last known sales value
        
        for t in range(16):  # Generate 16 days autoregressively
            # Prepare decoder input for current timestep
            future_idx = self.timesteps + t
            
            # Get known future features for this day
            future_features = []
            for feature in ['onpromotion', 'dcoilwtico', 'promo_roll_sum_7',
                          'is_weekend', 'is_holiday', 'month_sin', 'month_cos',
                          'dow_sin', 'dow_cos', 'days_to_holiday']:
                if feature in batch and batch[feature].size(1) > future_idx:
                    feat = batch[feature][:, future_idx:future_idx+1]
                    if len(feat.shape) == 2:
                        feat = feat.unsqueeze(-1)
                    future_features.append(feat)
            
            # CRITICAL AUTOREGRESSIVE STEP:
            # Combine PREVIOUS PREDICTION with future features
            # This is what makes the model autoregressive
            decoder_input = torch.cat([last_sales.unsqueeze(1)] + future_features, dim=2)
            
            # Pad to match expected input dimension
            if decoder_input.size(2) < self.num_temporal_features:
                padding = torch.zeros(
                    batch_size, 1, 
                    self.num_temporal_features - decoder_input.size(2)
                ).to(device)
                decoder_input = torch.cat([decoder_input, padding], dim=2)
            
            # Decode one step
            decoder_output, (decoder_hidden, decoder_cell) = self.decoder_lstm(
                decoder_input, (decoder_hidden, decoder_cell)
            )
            
            # Generate prediction for THIS SINGLE DAY
            output = F.relu(self.output_fc1(decoder_output[:, -1]))
            output = self.dropout(output)
            prediction = F.relu(self.output_fc2(output))  # Single day prediction
            
            outputs.append(prediction)
            last_sales = prediction  # USE THIS PREDICTION FOR NEXT STEP (autoregressive)
        
        # Stack all 16 individual predictions
        final_output = torch.cat(outputs, dim=1)
        
        return final_output


class OneShotLstmBased(BaseTemporalModel):
    """
    ONE-SHOT LSTM-BASED ARCHITECTURE:
    
    This model performs ONE-SHOT prediction of all 16 days simultaneously,
    WITHOUT any autoregressive generation. Despite having an encoder-decoder
    structure similar to the autoregressive model, the key difference is in
    the output mechanism.
    
    Architecture:
    - Encoder: Same bidirectional LSTM as AutoregressiveEncoderDecoderGruBased
    - Attention: Same multi-head attention mechanism
    - Output: DIRECT projection to 16 outputs (not autoregressive)
    
    KEY DIFFERENCE FROM AUTOREGRESSIVE MODEL:
    - No loop for generating predictions
    - Single output layer produces all 16 predictions at once
    - No dependency between predictions (parallel generation)
    - No error accumulation but less flexibility in modeling dependencies
    
    PREDICTION APPROACH: ONE-SHOT
    All 16 days are predicted in a single forward pass without any sequential
    dependency between the predictions.
    """
    
    def __init__(self, feature_info, timesteps=200, latent_dim=100, dropout=0.25):
        super().__init__(feature_info)
        
        self.timesteps = timesteps
        self.latent_dim = latent_dim
        self.dropout_rate = dropout
        
        # Calculate encoder input dimension
        self.num_temporal_features = 17
        
        # Encoder: Bidirectional LSTM (same as autoregressive model)
        self.encoder_lstm = nn.LSTM(
            input_size=self.num_temporal_features,
            hidden_size=latent_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0,
            bidirectional=True
        )
        
        # Attention mechanism (same as autoregressive model)
        self.attention = nn.MultiheadAttention(
            embed_dim=latent_dim * 2,
            num_heads=8,
            dropout=dropout
        )
        
        # Process encoded features
        self.encoder_fc = nn.Linear(latent_dim * 2, latent_dim)
        
        # Static features processing
        self.static_fc = nn.Linear(self.get_static_embedding_dim() + 1, 64)
        
        # Future features processing (all 16 days at once)
        future_features_dim = 16 * 10  # 16 days * ~10 features per day
        self.future_fc = nn.Linear(future_features_dim, 128)
        
        # Combine all features
        combined_dim = latent_dim + 64 + 128  # encoded + static + future
        
        # Deep processing layers
        self.fc1 = nn.Linear(combined_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        
        # CRITICAL DIFFERENCE: Direct projection to all 16 predictions
        # This is what makes it ONE-SHOT instead of autoregressive
        self.output_head = nn.Linear(128, 16)  # Outputs ALL 16 days at once
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.batch_norm2 = nn.BatchNorm1d(256)
    
    def prepare_temporal_features(self, batch, start_idx, end_idx):
        """Prepare temporal features for encoding"""
        features = []
        
        for feature in ['sales_hist', 'onpromotion', 'dcoilwtico', 'transactions',
                       'promo_roll_sum_7', 'promo_roll_sum_30',
                       'oil_pct_change', 'days_since_last_promo',
                       'days_to_holiday', 'days_from_holiday',
                       'is_weekend', 'is_holiday', 'promo_weekend_interaction',
                       'month_sin', 'month_cos', 'dow_sin', 'dow_cos']:
            if feature in batch:
                if feature == 'sales_hist':
                    feat = batch[feature][:, start_idx:end_idx]
                else:
                    feat = batch[feature][:, start_idx:end_idx]
                
                if len(feat.shape) == 2:
                    feat = feat.unsqueeze(-1)
                features.append(feat)
        
        return torch.cat(features, dim=2)
    
    def process_static_features(self, batch):
        """Process static categorical features through embeddings"""
        static_embeddings = []
        static_cat = batch['static_cat']
        
        # Embed static categorical features
        feature_names = ['city_encoded', 'state_encoded', 'store_type_encoded', 
                        'cluster_encoded', 'perishable', 'store_nbr_encoded', 'family_encoded']
        
        for idx, feature in enumerate(feature_names[:4]):
            if feature in self.embeddings:
                max_idx = self.embeddings[feature].num_embeddings - 1
                values = torch.clamp(static_cat[:, idx], 0, max_idx)
                embedded = self.embeddings[feature](values)
                static_embeddings.append(embedded)
        
        # Add perishable flag
        perishable = static_cat[:, 4].unsqueeze(1).float()
        static_embeddings.append(perishable)
        
        # Add store and family embeddings
        for idx, feature in enumerate(['store_nbr_encoded', 'family_encoded'], start=5):
            if feature in self.embeddings:
                max_idx = self.embeddings[feature].num_embeddings - 1
                values = torch.clamp(static_cat[:, idx], 0, max_idx)
                embedded = self.embeddings[feature](values)
                static_embeddings.append(embedded)
        
        return torch.cat(static_embeddings, dim=1)
    
    def forward(self, batch):
        batch_size = batch['sales_hist'].size(0)
        device = batch['sales_hist'].device
        
        # 1. Encode historical sequences
        encoder_features = self.prepare_temporal_features(batch, 0, self.timesteps)
        
        if encoder_features.size(2) != self.num_temporal_features:
            raise ValueError(f"Expected {self.num_temporal_features} features, got {encoder_features.size(2)}")
        
        # Encode with bidirectional LSTM
        encoder_output, (hidden, cell) = self.encoder_lstm(encoder_features)
        
        # Apply attention to get context vector
        encoder_output_t = encoder_output.transpose(0, 1)
        attended_output, _ = self.attention(
            encoder_output_t[-1:], encoder_output_t, encoder_output_t
        )
        attended_output = attended_output.transpose(0, 1)
        
        # Process encoded features
        encoded_features = self.encoder_fc(attended_output.squeeze(1))
        encoded_features = self.dropout(encoded_features)
        
        # 2. Process static features
        static_features = self.process_static_features(batch)
        static_features = F.relu(self.static_fc(static_features))
        static_features = self.dropout(static_features)
        
        # 3. Process ALL future features at once (NOT autoregressively)
        # This is different from the autoregressive model which processes day by day
        future_features_list = []
        
        for feature in ['onpromotion', 'dcoilwtico', 'promo_roll_sum_7',
                       'is_weekend', 'is_holiday', 'month_sin', 'month_cos',
                       'dow_sin', 'dow_cos', 'days_to_holiday']:
            if feature in batch:
                # Extract all 16 future days at once
                future_feat = batch[feature][:, -16:]
                if len(future_feat.shape) == 3:
                    future_feat = future_feat.squeeze(-1)
                future_features_list.append(future_feat)
        
        if future_features_list:
            future_concat = torch.cat(future_features_list, dim=1)
            future_features = F.relu(self.future_fc(future_concat))
            future_features = self.dropout(future_features)
        else:
            future_features = torch.zeros(batch_size, 128).to(device)
        
        # 4. Combine all feature types
        combined = torch.cat([encoded_features, static_features, future_features], dim=1)
        
        # 5. Process through deep layers
        x = F.relu(self.fc1(combined))
        x = self.batch_norm1(x)
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.batch_norm2(x)
        x = self.dropout(x)
        
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        
        # 6. ONE-SHOT OUTPUT: Predict all 16 days simultaneously
        # NO autoregressive loop - all predictions made in parallel
        output = F.relu(self.output_head(x))
        
        return output


def create_model(model_type, feature_info, timesteps=200, **kwargs):
    """
    Factory function to create models based on type.
    Centralizes model instantiation and parameter passing.
    """
    
    if model_type == 'cnn':
        return TemporalCnn(feature_info, timesteps, **kwargs)
    elif model_type == 'seq2seq':
        # This creates the AUTOREGRESSIVE model
        return AutoregressiveEncoderDecoderGruBased(feature_info, timesteps, **kwargs)
    elif model_type == 'multi-step-seq2seq':
        # This creates the ONE-SHOT LSTM model
        return OneShotLstmBased(feature_info, timesteps, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")