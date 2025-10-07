# 07_Training_LSTM.py
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="LSTM Training", page_icon="🔄", layout="wide")
st.title("🔄 LSTM Sequence Model Training")

class DocumentSequenceDataset(Dataset):
    """Dataset that provides document sequences for LSTM"""
    def __init__(self, df, feature_cols, max_seq_len=512):
        self.sequences = []
        self.labels = []
        
        # Group by document and page for sequences
        for (doc_id, page), group in df.groupby(['doc_id', 'page_number']):
            # Sort by reading order
            group = group.sort_values('reading_order')
            
            # Extract features and labels
            features = group[feature_cols].fillna(0).values
            labels = group['y_boundary'].values
            
            # Split long sequences
            for i in range(0, len(features), max_seq_len):
                end = min(i + max_seq_len, len(features))
                self.sequences.append(torch.FloatTensor(features[i:end]))
                self.labels.append(torch.FloatTensor(labels[i:end]))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]
    
    @staticmethod
    def collate_fn(batch):
        """Custom collate function for variable length sequences"""
        sequences, labels = zip(*batch)
        
        # Pad sequences
        sequences_padded = pad_sequence(sequences, batch_first=True)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)
        
        # Create mask for padded positions
        lengths = torch.tensor([len(seq) for seq in sequences])
        
        return sequences_padded, labels_padded, lengths

class BoundaryLSTM(nn.Module):
    """Bidirectional LSTM for boundary detection"""
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional
        
    def forward(self, x, lengths=None):
        # Pack sequences for efficient LSTM processing
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        lstm_out, _ = self.lstm(x)
        
        # Unpack if needed
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True
            )
        
        lstm_out = self.dropout(lstm_out)
        output = torch.sigmoid(self.classifier(lstm_out))
        
        return output.squeeze(-1)

class LSTMBoundaryModel:
    """LSTM wrapper compatible with existing pipeline"""
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, 
                 learning_rate=0.001, batch_size=16, device='cpu'):
        self.model = BoundaryLSTM(input_dim, hidden_dim, num_layers).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss(reduction='none')  # Handle padding
        self.batch_size = batch_size
        self.device = device
        self.feature_names_ = None
        self.threshold_ = 0.5
        self.training_history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
        
    def train(self, train_df, test_df=None, epochs=30, patience=7):
        """Train LSTM with sequences"""
        # Get feature columns
        metadata_cols = ['doc_id', 'page', 'span_id', 'y_boundary', 'page_number', 'reading_order']
        feature_cols = [c for c in train_df.columns if c not in metadata_cols]
        self.feature_names_ = feature_cols
        
        # Create datasets
        train_dataset = DocumentSequenceDataset(train_df, feature_cols)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=DocumentSequenceDataset.collate_fn
        )
        
        val_loader = None
        if test_df is not None:
            val_dataset = DocumentSequenceDataset(test_df, feature_cols)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                collate_fn=DocumentSequenceDataset.collate_fn
            )
        
        # Training loop
        progress_bar = st.progress(0)
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_samples = 0
            
            for batch_x, batch_y, lengths in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x, lengths)
                
                # Mask out padding
                mask = batch_y != -1
                loss = self.criterion(outputs[mask], batch_y[mask])
                loss = loss.mean()
                
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                train_loss += loss.item() * mask.sum().item()
                train_samples += mask.sum().item()
            
            train_loss /= train_samples
            self.training_history['train_loss'].append(train_loss)
            
            # Validation phase
            if val_loader:
                self.model.eval()
                val_loss = 0
                val_samples = 0
                all_preds = []
                all_labels = []
                
                with torch.no_grad():
                    for batch_x, batch_y, lengths in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        outputs = self.model(batch_x, lengths)
                        mask = batch_y != -1
                        
                        loss = self.criterion(outputs[mask], batch_y[mask])
                        val_loss += loss.sum().item()
                        val_samples += mask.sum().item()
                        
                        all_preds.extend(outputs[mask].cpu().numpy())
                        all_labels.extend(batch_y[mask].cpu().numpy())
                
                val_loss /= val_samples
                self.training_history['val_loss'].append(val_loss)
                
                # Calculate AUC
                from sklearn.metrics import roc_auc_score
                val_auc = roc_auc_score(all_labels, all_preds)
                self.training_history['val_auc'].append(val_auc)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.best_model_state = self.model.state_dict()
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    st.info(f"Early stopping at epoch {epoch+1}")
                    break
                
                progress_bar.progress(
                    (epoch + 1) / epochs,
                    f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Val AUC: {val_auc:.3f}"
                )
            else:
                progress_bar.progress(
                    (epoch + 1) / epochs,
                    f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}"
                )
        
        # Restore best model
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
        
        progress_bar.empty()
        
        return {'train_metrics': {'final_loss': train_loss}}
    
    def predict_proba(self, X):
        """Predict on DataFrame (converts to sequences internally)"""
        # Need to handle conversion from DataFrame to sequences
        # For simplicity, treating each row independently (not ideal for LSTM)
        if isinstance(X, pd.DataFrame):
            X_values = X[self.feature_names_].fillna(0).values
        else:
            X_values = X
        
        self.model.eval()
        with torch.no_grad():
            # Process in batches
            probs = []
            for i in range(0, len(X_values), self.batch_size):
                batch = torch.FloatTensor(X_values[i:i+self.batch_size]).to(self.device)
                # Add sequence dimension
                batch = batch.unsqueeze(1)  # (batch, seq_len=1, features)
                outputs = self.model(batch).squeeze()
                probs.extend(outputs.cpu().numpy())
        
        probs = np.array(probs)
        return np.column_stack([1 - probs, probs])

# UI Components
st.markdown("""
### 🔄 LSTM Advantages
- Captures sequential dependencies between spans
- Better for documents with consistent reading flow
- Can learn paragraph and section patterns
- Ideal when span order matters significantly
""")

col1, col2, col3 = st.columns(3)

with col1:
    hidden_dim = st.selectbox(
        "Hidden Dimension",
        [64, 128, 256],
        index=1
    )
    
    num_layers = st.selectbox(
        "LSTM Layers",
        [1, 2, 3],
        index=1
    )

with col2:
    learning_rate = st.select_slider(
        "Learning Rate",
        options=[0.0001, 0.0005, 0.001, 0.005],
        value=0.001
    )
    
    batch_size = st.selectbox(
        "Batch Size",
        [8, 16, 32],
        index=1,
        help="Smaller batch sizes work better for sequences"
    )

with col3:
    epochs = st.slider("Max Epochs", 10, 100, 30)
    max_seq_len = st.slider(
        "Max Sequence Length",
        64, 1024, 512,
        help="Split longer documents into chunks"
    )

# Training comparison
st.subheader("📊 Model Comparison")

comparison_df = pd.DataFrame({
    'Model': ['Random Forest', 'Neural Network', 'LSTM'],
    'Training Speed': ['Fast', 'Medium', 'Slow'],
    'Data Required': ['Low (3-5 docs)', 'Medium (10+ docs)', 'High (20+ docs)'],
    'Interpretability': ['High', 'Low', 'Very Low'],
    'Sequential Patterns': ['No', 'No', 'Yes'],
    'Best For': ['Quick baseline', 'Complex features', 'Document flow']
})

st.dataframe(comparison_df, use_container_width=True)

# Recommendations
st.info("""
**📌 Recommendations:**
- **< 5 documents**: Stick with Random Forest
- **5-15 documents**: Try Neural Network
- **15+ documents**: LSTM becomes viable
- **Complex layouts**: Neural Network often best
- **Consistent structure**: LSTM can excel
""")