import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib

sys.path.append(str(Path(__file__).parent.parent.parent))

from app.state import AppState

st.set_page_config(page_title="NN Training", page_icon="🧠", layout="wide")

AppState.init()

st.title("🧠 3 Neural Network Training")

st.markdown("""
Train deep learning models for boundary detection. Neural networks can capture complex patterns
and non-linear relationships in your data that traditional models might miss.
""")


# Neural Network Architecture
class BoundaryNN(nn.Module):
    """Feedforward neural network for boundary detection"""
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return torch.sigmoid(self.network(x))


class NeuralBoundaryModel:
    """Neural network wrapper compatible with existing pipeline"""
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], 
                 learning_rate=0.001, batch_size=32, dropout_rate=0.3, device='cpu'):
        self.model = BoundaryNN(input_dim, hidden_dims, dropout_rate).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        self.batch_size = batch_size
        self.device = device
        self.scaler = StandardScaler()
        self.feature_names_ = None
        self.threshold_ = 0.5
        self.training_history = {
            'train_loss': [], 
            'val_loss': [], 
            'val_auc': [],
            'val_acc': []
        }
        
    def prepare_data(self, df):
        """Extract and prepare features from dataframe"""
        metadata_cols = ['doc_id', 'page', 'span_id', 'y_boundary', 'page_number']
        feature_cols = [c for c in df.columns if c not in metadata_cols]
        
        X = df[feature_cols].fillna(0).values
        y = df['y_boundary'].values
        
        return X, y, feature_cols
        
    def train(self, train_df, test_df=None, epochs=50, patience=10, verbose=True):
        """Train neural network with early stopping"""
        # Extract features
        X_train, y_train, feature_cols = self.prepare_data(train_df)
        self.feature_names_ = feature_cols
        
        # Normalize features
        X_train = self.scaler.fit_transform(X_train)
        
        # Prepare validation data
        X_val, y_val = None, None
        if test_df is not None:
            X_val, y_val, _ = self.prepare_data(test_df)
            X_val = self.scaler.transform(X_val)
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        # Training loop
        if verbose:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x).squeeze()
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.training_history['train_loss'].append(train_loss)
            
            # Validation phase
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_tensor = torch.FloatTensor(X_val).to(self.device)
                    val_outputs = self.model(val_tensor).squeeze()
                    val_loss = self.criterion(
                        val_outputs, 
                        torch.FloatTensor(y_val).to(self.device)
                    )
                    
                    # Calculate metrics
                    val_preds_proba = val_outputs.cpu().numpy()
                    val_preds = (val_preds_proba >= self.threshold_).astype(int)
                    
                    val_auc = roc_auc_score(y_val, val_preds_proba)
                    val_acc = accuracy_score(y_val, val_preds)
                    
                self.training_history['val_loss'].append(val_loss.item())
                self.training_history['val_auc'].append(val_auc)
                self.training_history['val_acc'].append(val_acc)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    if verbose:
                        st.info(f"⏹️ Early stopping at epoch {epoch+1}/{epochs}")
                    break
                
                # Update progress
                if verbose:
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(
                        progress, 
                        f"Epoch {epoch+1}/{epochs}"
                    )
                    status_text.text(
                        f"Train Loss: {train_loss:.4f} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Val AUC: {val_auc:.3f} | "
                        f"Val Acc: {val_acc:.3f}"
                    )
            else:
                if verbose:
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress, f"Epoch {epoch+1}/{epochs}")
                    status_text.text(f"Train Loss: {train_loss:.4f}")
        
        # Restore best model
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
        
        if verbose:
            progress_bar.empty()
            status_text.empty()
        
        # Calculate final train metrics
        self.model.eval()
        with torch.no_grad():
            train_outputs = self.model(torch.FloatTensor(X_train).to(self.device))
            train_probs = train_outputs.squeeze().cpu().numpy()
            train_preds = (train_probs >= self.threshold_).astype(int)
            
            train_auc = roc_auc_score(y_train, train_probs)
            train_acc = accuracy_score(y_train, train_preds)
        
        train_metrics = {
            'auc_roc': train_auc,
            'accuracy': train_acc,
            'final_loss': self.training_history['train_loss'][-1]
        }
        
        return {'train_metrics': train_metrics}
    
    def predict_proba(self, X):
        """Predict probabilities"""
        import pandas as pd
        
        if isinstance(X, list):
            # Handle list of feature dictionaries
            X = pd.DataFrame(X)
        
        if isinstance(X, pd.DataFrame):
            # Ensure all expected features are present
            for col in self.feature_names_:
                if col not in X.columns:
                    X[col] = 0
            # Select features in the correct order
            X = X[self.feature_names_].fillna(0).values
        
        X = self.scaler.transform(X)
        
        self.model.eval()
        with torch.no_grad():
            probs = self.model(torch.FloatTensor(X).to(self.device))
            probs = probs.squeeze().cpu().numpy()
        
        # Handle single sample case
        if probs.ndim == 0:
            probs = np.array([probs])
        
        # Return in sklearn format (n_samples, 2)
        return np.column_stack([1 - probs, probs])
    
    def predict(self, X):
        """Predict binary labels"""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold_).astype(int)
    
    def save(self, path):
        """Save model, scaler, and metadata"""
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)
        
        # Save PyTorch model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_dim': self.model.network[0].in_features,
                'hidden_dims': [
                    layer.out_features for layer in self.model.network 
                    if isinstance(layer, nn.Linear)
                ][:-1],
                'dropout_rate': [
                    layer.p for layer in self.model.network 
                    if isinstance(layer, nn.Dropout)
                ][0] if any(isinstance(layer, nn.Dropout) for layer in self.model.network) else 0.3
            },
            'threshold': self.threshold_,
            'feature_names': self.feature_names_,
            'training_history': self.training_history
        }, path.with_suffix('.pth'))
        
        # Save scaler
        joblib.dump(self.scaler, path.with_suffix('.scaler'))
    
    @classmethod
    def load(cls, path):
        """Load saved model"""
        path = Path(path)
        
        # Check device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load PyTorch model
        checkpoint = torch.load(path.with_suffix('.pth'), map_location=device)
        
        # Recreate model
        model = cls(
            input_dim=checkpoint['model_config']['input_dim'],
            hidden_dims=checkpoint['model_config']['hidden_dims'],
            dropout_rate=checkpoint['model_config'].get('dropout_rate', 0.3),
            device=device
        )
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.threshold_ = checkpoint['threshold']
        model.feature_names_ = checkpoint['feature_names']
        model.training_history = checkpoint.get('training_history', {})
        
        # Load scaler
        model.scaler = joblib.load(path.with_suffix('.scaler'))
        
        return model
    
    def get_model_type(self):
        """Return model type for display purposes"""
        return "Neural Network"


def evaluate_model_performance(model, test_df):
    """Evaluate model on test set"""
    y_true = test_df['y_boundary'].values
    y_proba = model.predict_proba(test_df)[:, 1]
    y_pred = model.predict(test_df)
    
    metrics = {
        'auc_roc': roc_auc_score(y_true, y_proba),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    return metrics


def get_available_datasets():
    """Get list of available training datasets"""
    exports_dir = Path("exports")
    if not exports_dir.exists():
        return []
    
    datasets = []
    for file_path in exports_dir.glob("*.parquet"):
        # Skip train_ and test_ prefixed files
        if file_path.name.startswith('train_') or file_path.name.startswith('test_'):
            continue
        
        datasets.append({
            'name': file_path.stem,
            'path': file_path,
            'size': file_path.stat().st_size / (1024 * 1024),
            'modified': datetime.fromtimestamp(file_path.stat().st_mtime)
        })
    
    return datasets


def perform_document_split(df, test_size=0.25, random_state=42):
    """Split data at document level for better generalization testing"""
    if 'doc_id' not in df.columns:
        st.warning("No 'doc_id' column found. Using random split instead.")
        return train_test_split(
            df, test_size=test_size, 
            random_state=random_state, 
            stratify=df['y_boundary']
        )
    
    unique_docs = df['doc_id'].unique()
    n_docs = len(unique_docs)
    
    st.info(f"Found {n_docs} unique documents in dataset")
    
    if n_docs < 2:
        st.warning("Only 1 document found. Using random split.")
        return train_test_split(
            df, test_size=test_size, 
            random_state=random_state, 
            stratify=df['y_boundary']
        )
    
    # Calculate number of test documents
    n_test_docs = max(1, int(n_docs * test_size))
    n_train_docs = n_docs - n_test_docs
    
    # Randomly select test documents
    np.random.seed(random_state)
    test_docs = np.random.choice(unique_docs, size=n_test_docs, replace=False)
    train_docs = [doc for doc in unique_docs if doc not in test_docs]
    
    # Split data
    train_df = df[df['doc_id'].isin(train_docs)].copy()
    test_df = df[df['doc_id'].isin(test_docs)].copy()
    
    # Display split information
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Documents", n_train_docs)
        st.caption(f"Docs: {', '.join(map(str, train_docs[:3]))}{'...' if len(train_docs) > 3 else ''}")
    with col2:
        st.metric("Test Documents", n_test_docs)
        st.caption(f"Docs: {', '.join(map(str, test_docs[:3]))}{'...' if len(test_docs) > 3 else ''}")
    with col3:
        train_pct = len(train_df) / len(df) * 100
        st.metric("Train/Test Split", f"{train_pct:.0f}% / {100-train_pct:.0f}%")
    
    return train_df, test_df


# Initialize session state
if 'training_results' not in st.session_state:
    st.session_state.training_results = None


# Main UI
st.header("📊 Select Training Data")

datasets = get_available_datasets()

if not datasets:
    st.warning("⚠️ No training datasets found. Please export data from the Export page first.")
    st.stop()

# Dataset selection
dataset_options = {d['name']: d for d in datasets}
selected_dataset_name = st.selectbox(
    "Select training dataset",
    options=list(dataset_options.keys()),
    format_func=lambda x: f"{x} ({dataset_options[x]['size']:.1f} MB)"
)

selected_dataset = dataset_options[selected_dataset_name]

# Display dataset info
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Dataset", selected_dataset_name)
with col2:
    st.metric("Size", f"{selected_dataset['size']:.1f} MB")
with col3:
    st.metric("Modified", selected_dataset['modified'].strftime('%Y-%m-%d %H:%M'))

# Analyze dataset button
if st.button("📊 Analyze Dataset"):
    df = pd.read_parquet(selected_dataset['path'])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Examples", f"{len(df):,}")
    with col2:
        st.metric("Positive Boundaries", f"{df['y_boundary'].sum():,}")
    with col3:
        st.metric("Negative Examples", f"{(len(df) - df['y_boundary'].sum()):,}")
    with col4:
        boundary_rate = df['y_boundary'].mean() * 100
        st.metric("Boundary Rate", f"{boundary_rate:.1f}%")
    
    # Document distribution
    if 'doc_id' in df.columns:
        doc_counts = df['doc_id'].value_counts()
        st.write(f"**📚 Documents in dataset:** {len(doc_counts)}")
        
        fig = px.bar(
            x=doc_counts.index.astype(str),
            y=doc_counts.values,
            title="Examples per Document",
            labels={'x': 'Document', 'y': 'Examples'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


# Neural Network Configuration
st.header("🎛️ Neural Network Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    architecture = st.selectbox(
        "Architecture",
        ["Small (128, 64)", "Medium (256, 128, 64)", "Large (512, 256, 128, 64)"],
        index=1,
        help="Network size - larger networks can learn more complex patterns"
    )
    
    hidden_dims_map = {
        "Small (128, 64)": [128, 64],
        "Medium (256, 128, 64)": [256, 128, 64],
        "Large (512, 256, 128, 64)": [512, 256, 128, 64]
    }
    hidden_dims = hidden_dims_map[architecture]

with col2:
    learning_rate = st.select_slider(
        "Learning Rate",
        options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
        value=0.001,
        help="Learning rate - lower is more stable, higher trains faster"
    )
    
    dropout = st.slider(
        "Dropout Rate", 
        0.1, 0.5, 0.3, 0.05,
        help="Dropout for regularization - prevents overfitting"
    )

with col3:
    batch_size = st.selectbox(
        "Batch Size",
        [16, 32, 64, 128],
        index=1,
        help="Samples per training step"
    )
    
    epochs = st.slider(
        "Max Epochs", 
        10, 200, 50, 10,
        help="Maximum training iterations"
    )

# Device info
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    st.success(f"🚀 GPU acceleration available")
else:
    st.info(f"💻 Training on CPU")


# Training Configuration
st.header("⚙️ Training Setup")

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    model_name = st.text_input(
        "Model Name",
        value=f"nn_boundary_{datetime.now().strftime('%Y%m%d_%H%M')}",
        help="Name for the trained model"
    )

with col2:
    use_all_data = st.checkbox(
        "Train on all data",
        value=False,
        help="Skip train/test split - use for final models"
    )

with col3:
    if not use_all_data:
        test_size = st.slider(
            "Test Fraction",
            min_value=0.1,
            max_value=0.4,
            value=0.25,
            step=0.05
        )
    else:
        test_size = 0.25
        st.info("ℹ️ Using 100% for training")


# Training Section
st.header("🚀 Train Model")

col1, col2 = st.columns([1, 1])

with col1:
    train_button = st.button(
        "🧠 Start Neural Network Training", 
        type="primary", 
        use_container_width=True
    )

with col2:
    if use_all_data:
        st.warning("⚠️ Training on all data - no test metrics will be available")
    else:
        st.info(f"📊 Will use document-level split for validation")


if train_button:
    if not model_name.strip():
        st.error("❌ Please enter a model name")
    else:
        with st.spinner("🧠 Training neural network..."):
            try:
                # Load data
                df = pd.read_parquet(selected_dataset['path'])
                
                # Verify data
                if 'y_boundary' not in df.columns:
                    raise ValueError("Training data missing 'y_boundary' column")
                
                # Split or use all data
                if use_all_data:
                    st.info(f"🎯 Training on **all {len(df):,} examples**")
                    train_df = df.copy()
                    test_df = None
                else:
                    train_df, test_df = perform_document_split(df, test_size=test_size)
                
                # Get feature dimension
                metadata_cols = ['doc_id', 'page', 'span_id', 'y_boundary', 'page_number']
                feature_cols = [c for c in train_df.columns if c not in metadata_cols]
                input_dim = len(feature_cols)
                
                # Display configuration
                with st.expander("🔧 Model Configuration", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Dataset:**")
                        st.write(f"- Training samples: {len(train_df):,}")
                        if test_df is not None:
                            st.write(f"- Test samples: {len(test_df):,}")
                        st.write(f"- Features: {input_dim}")
                        st.write(f"- Positive rate: {train_df['y_boundary'].mean():.2%}")
                    with col2:
                        st.write("**Network:**")
                        st.write(f"- Architecture: {hidden_dims}")
                        st.write(f"- Learning rate: {learning_rate}")
                        st.write(f"- Batch size: {batch_size}")
                        st.write(f"- Dropout: {dropout}")
                        st.write(f"- Device: {device}")
                
                # Initialize model
                model = NeuralBoundaryModel(
                    input_dim=input_dim,
                    hidden_dims=hidden_dims,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    dropout_rate=dropout,
                    device=device
                )
                
                # Train
                training_metrics = model.train(
                    train_df, 
                    test_df, 
                    epochs=epochs, 
                    patience=10,
                    verbose=True
                )
                
                # Evaluate
                test_metrics = None
                if test_df is not None:
                    test_metrics = evaluate_model_performance(model, test_df)
                
                # Save model
                model_dir = Path("models")
                model_dir.mkdir(exist_ok=True)
                model_path = model_dir / f"{model_name.strip()}.pkl"
                model.save(model_path)
                
                # Save metadata
                metadata_path = model_dir / f"{model_name.strip()}.json"
                metadata = {
                    'model_name': model_name.strip(),
                    'model_type': 'neural_network',
                    'architecture': hidden_dims,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'dropout': dropout,
                    'epochs_trained': len(model.training_history['train_loss']),
                    'device': device,
                    'train_metrics': training_metrics['train_metrics'],
                    'test_metrics': test_metrics if test_metrics else {},
                    'n_train_samples': len(train_df),
                    'n_test_samples': len(test_df) if test_df is not None else 0,
                    'n_train_docs': len(train_df['doc_id'].unique()) if 'doc_id' in train_df.columns else 'N/A',
                    'n_test_docs': len(test_df['doc_id'].unique()) if test_df is not None and 'doc_id' in test_df.columns else 'N/A',
                    'feature_count': input_dim,
                    'trained_on_all_data': use_all_data,
                    'created_at': datetime.now().isoformat(),
                    'training_history': model.training_history
                }
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Store results
                st.session_state.training_results = {
                    **metadata,
                    'model_path': str(model_path)
                }
                
                st.success("✅ Neural network training completed!")
                
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                st.exception(e)


# Display Results
if st.session_state.training_results:
    st.header("📈 Training Results")
    
    results = st.session_state.training_results
    trained_on_all = results.get('trained_on_all_data', False)
    
    # Main metrics
    if trained_on_all:
        col1, col2, col3 = st.columns(3)
        with col1:
            train_auc = results.get('train_metrics', {}).get('auc_roc', 0)
            st.metric("Training AUC", f"{train_auc:.3f}")
        with col2:
            st.metric("Epochs Trained", results.get('epochs_trained', 'N/A'))
        with col3:
            st.metric("Total Samples", f"{results.get('n_train_samples', 0):,}")
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            train_auc = results.get('train_metrics', {}).get('auc_roc', 0)
            st.metric("Training AUC", f"{train_auc:.3f}")
        with col2:
            test_auc = results.get('test_metrics', {}).get('auc_roc', 0)
            delta = test_auc - train_auc if train_auc > 0 else 0
            st.metric("Test AUC", f"{test_auc:.3f}", delta=f"{delta:+.3f}")
        with col3:
            st.metric("Epochs Trained", results.get('epochs_trained', 'N/A'))
        with col4:
            st.metric("Test Docs", results.get('n_test_docs', 'N/A'))
    
    # Training curves
    if results.get('training_history'):
        st.subheader("📊 Training Curves")
        
        history = results['training_history']
        
        fig = go.Figure()
        
        # Train loss
        fig.add_trace(go.Scatter(
            y=history['train_loss'],
            mode='lines',
            name='Train Loss',
            line=dict(color='blue')
        ))
        
        # Validation loss
        if history.get('val_loss'):
            fig.add_trace(go.Scatter(
                y=history['val_loss'],
                mode='lines',
                name='Val Loss',
                line=dict(color='orange')
            ))
        
        # Validation AUC
        if history.get('val_auc'):
            fig.add_trace(go.Scatter(
                y=history['val_auc'],
                mode='lines',
                name='Val AUC',
                yaxis='y2',
                line=dict(color='green')
            ))
        
        fig.update_layout(
            title="Training Progress",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            yaxis2=dict(
                title="AUC",
                overlaying='y',
                side='right',
                range=[0, 1]
            ),
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance interpretation
    if not trained_on_all:
        test_auc = results.get('test_metrics', {}).get('auc_roc', 0)
        train_auc = results.get('train_metrics', {}).get('auc_roc', 0)
        
        if test_auc >= 0.85:
            st.success("🎯 **Excellent!** Neural network is performing very well.")
        elif test_auc >= 0.75:
            st.info("✅ **Good!** Model is usable, may improve with more data.")
        elif test_auc >= 0.65:
            st.warning("⚠️ **Fair.** Consider more training data or different architecture.")
        else:
            st.error("❌ **Poor.** Need more diverse training examples.")
        
        # Overfitting check
        if train_auc - test_auc > 0.15:
            st.warning("⚠️ **Overfitting detected!** Large gap between train and test. Try:\n- Increase dropout\n- Use smaller architecture\n- Add more training data")


# Existing Models
st.header("📚 Saved Neural Network Models")

models_dir = Path("models")
if models_dir.exists():
    nn_model_files = [f for f in models_dir.glob("*.pth")]
    
    if nn_model_files:
        nn_model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        for model_file in nn_model_files:
            with st.expander(f"🧠 {model_file.stem}"):
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    metadata_file = model_file.with_suffix('.json')
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        st.write(f"**Architecture:** {metadata.get('architecture', 'N/A')}")
                        st.write(f"**Training AUC:** {metadata.get('train_metrics', {}).get('auc_roc', 'N/A')}")
                        
                        if not metadata.get('trained_on_all_data', False):
                            st.write(f"**Test AUC:** {metadata.get('test_metrics', {}).get('auc_roc', 'N/A')}")
                        
                        st.write(f"**Epochs:** {metadata.get('epochs_trained', 'N/A')}")
                    else:
                        st.write("No metadata available")
                
                with col2:
                    file_size = model_file.stat().st_size / (1024 * 1024)
                    mod_time = datetime.fromtimestamp(model_file.stat().st_mtime)
                    st.write(f"**Size:** {file_size:.1f} MB")
                    st.write(f"**Modified:** {mod_time.strftime('%Y-%m-%d')}")
                
                with col3:
                    if st.button("🗑️ Delete", key=f"delete_{model_file.stem}"):
                        model_file.unlink()
                        if metadata_file.exists():
                            metadata_file.unlink()
                        # Delete scaler too
                        scaler_file = model_file.with_suffix('.scaler')
                        if scaler_file.exists():
                            scaler_file.unlink()
                        st.rerun()
    else:
        st.info("No neural network models found yet")
else:
    st.info("Models directory will be created when you train your first model")


# Information Section
st.header("💡 Neural Network Tips")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **When to use Neural Networks:**
    - Large datasets (1000+ examples)
    - Complex patterns in data
    - Non-linear relationships
    - After Random Forest baseline
    
    **Architecture Selection:**
    - **Small:** Fast, less prone to overfitting
    - **Medium:** Balanced choice (recommended)
    - **Large:** More capacity, needs more data
    """)

with col2:
    st.markdown("""
    **Training Tips:**
    - Start with medium architecture
    - Monitor train/test gap for overfitting
    - Use early stopping (automatic)
    - Lower learning rate if unstable
    - Increase dropout if overfitting
    
    **GPU Acceleration:**
    - Much faster training with CUDA
    - Automatic device selection
    """)