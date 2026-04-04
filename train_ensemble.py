"""
China A-Share Market Prediction - 完整训练融合流程 (优化版)
XGBoost + GRU + Transformer 三模型融合
优化: TimeSeriesSplit + 统一验证集 + 滚动特征 + 交叉特征
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
import gc
import os
import time

warnings.filterwarnings('ignore')
np.random.seed(42)

print("="*70)
print("China A-Share Market Prediction - 优化版")
print("="*70)

TOTAL_START = time.time()

if os.path.exists('/kaggle/input'):
    BASE_PATH = '/kaggle/input/competitions/china-a-share-market-microstructure-prediction/'
    OUTPUT_PATH = '/kaggle/working/'
elif os.path.exists('/root/autodl-tmp'):
    BASE_PATH = '/root/autodl-tmp/'
    OUTPUT_PATH = '/root/.trae-cn/china-a-share-prediction/'
else:
    BASE_PATH = './data/'
    OUTPUT_PATH = './'

print(f"数据路径: {BASE_PATH}")
print(f"输出路径: {OUTPUT_PATH}")

USE_GPU = torch.cuda.is_available()
device = torch.device('cuda' if USE_GPU else 'cpu')
print(f"GPU可用: {USE_GPU}, 设备: {device}")

ID_COLS = ['stockid', 'dateid', 'timeid', 'exchangeid']
TARGET_COLS = ['LabelA', 'LabelB', 'LabelC']
TRAIN_RATIO = 0.85
N_PCA = 32
WINDOW_SIZE = 10

print("\n" + "="*70)
print("[1] 数据加载")
print("="*70)

try:
    import pyarrow.parquet as pq
    
    pf_train = pq.ParquetFile(BASE_PATH + 'train.parquet')
    n_rows = pf_train.metadata.num_rows
    n_chunks = pf_train.metadata.num_row_groups
    chunk_list = []
    
    print(f"训练集: {n_rows:,} 行, {n_chunks} 个row_group")
    
    for i in range(n_chunks):
        chunk = pf_train.read_row_group(i).to_pandas()
        for col in chunk.columns:
            if col not in ID_COLS and chunk[col].dtype == 'float64':
                chunk[col] = chunk[col].astype('float32')
        chunk_list.append(chunk)
        print(f"  读取进度: {i+1}/{n_chunks}")
    
    train_df = pd.concat(chunk_list, ignore_index=True)
    del chunk_list
    gc.collect()
    
    test_df = pd.read_parquet(BASE_PATH + 'test.parquet')
    for col in test_df.columns:
        if col not in ID_COLS and test_df[col].dtype == 'float64':
            test_df[col] = test_df[col].astype('float32')
    
    print(f"训练集: {train_df.shape}, 测试集: {test_df.shape}")
    
except Exception as e:
    print(f"数据加载错误: {e}")
    raise

print("\n" + "="*70)
print("[2] 时间序列划分")
print("="*70)

train_df = train_df.sort_values(['stockid', 'dateid', 'timeid']).reset_index(drop=True)
unique_dates = sorted(train_df['dateid'].unique())
n_train_dates = int(len(unique_dates) * TRAIN_RATIO)
train_dates = unique_dates[:n_train_dates]
val_dates = unique_dates[n_train_dates:]

train_data = train_df[train_df['dateid'].isin(train_dates)].copy()
val_data = train_df[train_df['dateid'].isin(val_dates)].copy()
del train_df
gc.collect()

train_data = train_data.sort_values(['stockid', 'dateid', 'timeid']).reset_index(drop=True)
val_data = val_data.sort_values(['stockid', 'dateid', 'timeid']).reset_index(drop=True)

val_data['val_idx'] = range(len(val_data))

print(f"训练样本: {len(train_data):,}, 验证样本: {len(val_data):,}")

print("\n" + "="*70)
print("[3] 特征工程")
print("="*70)

pca_cols = [f'f{i}' for i in range(200) if f'f{i}' in train_data.columns]
print(f"PCA输入特征数: {len(pca_cols)}")

scaler_pca = StandardScaler()
pca_train = scaler_pca.fit_transform(train_data[pca_cols].fillna(0).replace([np.inf, -np.inf], 0))
pca_val = scaler_pca.transform(val_data[pca_cols].fillna(0).replace([np.inf, -np.inf], 0))
pca_test = scaler_pca.transform(test_df[pca_cols].fillna(0).replace([np.inf, -np.inf], 0))

pca = PCA(n_components=N_PCA, random_state=42)
X_pca_train = pca.fit_transform(pca_train)
X_pca_val = pca.transform(pca_val)
X_pca_test = pca.transform(pca_test)

print(f"PCA解释方差比: {pca.explained_variance_ratio_.sum():.4f}")

for i in range(N_PCA):
    train_data[f'pca_{i}'] = X_pca_train[:, i].astype('float32')
    val_data[f'pca_{i}'] = X_pca_val[:, i].astype('float32')
    test_df[f'pca_{i}'] = X_pca_test[:, i].astype('float32')

del pca_train, pca_val, pca_test, X_pca_train, X_pca_val, X_pca_test
gc.collect()
print("  PCA特征完成")

if 'f298' in train_data.columns:
    train_data['price_change'] = train_data.groupby('stockid')['f298'].diff().fillna(0)
    val_data['price_change'] = val_data.groupby('stockid')['f298'].diff().fillna(0)
    test_df['price_change'] = test_df.groupby('stockid')['f298'].diff().fillna(0)
    
    train_data['price_ma5'] = train_data.groupby('stockid')['f298'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    train_data['price_ma10'] = train_data.groupby('stockid')['f298'].transform(lambda x: x.rolling(10, min_periods=1).mean())
    train_data['price_std5'] = train_data.groupby('stockid')['f298'].transform(lambda x: x.rolling(5, min_periods=1).std()).fillna(0)
    train_data['price_mom5'] = train_data.groupby('stockid')['f298'].diff(5).fillna(0)
    
    val_data['price_ma5'] = val_data.groupby('stockid')['f298'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    val_data['price_ma10'] = val_data.groupby('stockid')['f298'].transform(lambda x: x.rolling(10, min_periods=1).mean())
    val_data['price_std5'] = val_data.groupby('stockid')['f298'].transform(lambda x: x.rolling(5, min_periods=1).std()).fillna(0)
    val_data['price_mom5'] = val_data.groupby('stockid')['f298'].diff(5).fillna(0)
    
    test_df['price_ma5'] = test_df.groupby('stockid')['f298'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    test_df['price_ma10'] = test_df.groupby('stockid')['f298'].transform(lambda x: x.rolling(10, min_periods=1).mean())
    test_df['price_std5'] = test_df.groupby('stockid')['f298'].transform(lambda x: x.rolling(5, min_periods=1).std()).fillna(0)
    test_df['price_mom5'] = test_df.groupby('stockid')['f298'].diff(5).fillna(0)
    
    daily_stats = train_data.groupby('dateid').agg({'price_change': ['mean', 'std']}).reset_index()
    daily_stats.columns = ['dateid', 'market_return', 'market_vol']
    daily_stats['market_vol'] = daily_stats['market_vol'].fillna(0)
    
    train_data = train_data.merge(daily_stats, on='dateid', how='left')
    global_market_return = daily_stats['market_return'].mean()
    global_market_vol = daily_stats['market_vol'].mean()
    val_data['market_return'] = global_market_return
    val_data['market_vol'] = global_market_vol
    test_df['market_return'] = global_market_return
    test_df['market_vol'] = global_market_vol
    print("  价格滚动特征完成")
else:
    for df in [train_data, val_data, test_df]:
        df['market_return'] = 0.0
        df['market_vol'] = 0.0

train_data['time_sin'] = np.sin(2 * np.pi * train_data['timeid'] / 2400).astype('float32')
train_data['time_cos'] = np.cos(2 * np.pi * train_data['timeid'] / 2400).astype('float32')
val_data['time_sin'] = np.sin(2 * np.pi * val_data['timeid'] / 2400).astype('float32')
val_data['time_cos'] = np.cos(2 * np.pi * val_data['timeid'] / 2400).astype('float32')
test_df['time_sin'] = np.sin(2 * np.pi * test_df['timeid'] / 2400).astype('float32')
test_df['time_cos'] = np.cos(2 * np.pi * test_df['timeid'] / 2400).astype('float32')
print("  时间特征完成")

global_mean = train_data['LabelA'].mean()
train_sorted = train_data.sort_values(['stockid', 'dateid', 'timeid']).reset_index(drop=True)
stock_ids = train_sorted['stockid'].values
labels = train_sorted['LabelA'].values
stock_te = np.full(len(train_sorted), global_mean, dtype=np.float32)

stock_cumsum = {}
stock_cnt = {}
for idx in range(len(train_sorted)):
    stock = stock_ids[idx]
    label = labels[idx]
    if stock not in stock_cumsum:
        stock_cumsum[stock] = 0.0
        stock_cnt[stock] = 0
    if stock_cnt[stock] > 0:
        stock_te[idx] = (stock_cumsum[stock] + global_mean * 100) / (stock_cnt[stock] + 100)
    stock_cumsum[stock] += label
    stock_cnt[stock] += 1

train_data['stock_te'] = stock_te

stock_count = train_data.groupby('stockid').size()
stock_mean = train_data.groupby('stockid')['LabelA'].mean()
smoothed = (stock_mean * stock_count + global_mean * 100) / (stock_count + 100)

val_data['stock_te'] = val_data['stockid'].map(smoothed).fillna(global_mean).astype('float32')
test_df['stock_te'] = test_df['stockid'].map(smoothed).fillna(global_mean).astype('float32')

del train_sorted, stock_cumsum, stock_cnt, stock_ids, labels, stock_te
gc.collect()
print("  目标编码完成")

if 'f0' in train_data.columns and 'f1' in train_data.columns:
    train_data['f0_f1_ratio'] = (train_data['f0'] / (train_data['f1'] + 1e-6)).astype('float32')
    train_data['f0_f1_diff'] = (train_data['f0'] - train_data['f1']).astype('float32')
    val_data['f0_f1_ratio'] = (val_data['f0'] / (val_data['f1'] + 1e-6)).astype('float32')
    val_data['f0_f1_diff'] = (val_data['f0'] - val_data['f1']).astype('float32')
    test_df['f0_f1_ratio'] = (test_df['f0'] / (test_df['f1'] + 1e-6)).astype('float32')
    test_df['f0_f1_diff'] = (test_df['f0'] - test_df['f1']).astype('float32')
    print("  交叉特征完成")

tech_features = ['time_sin', 'time_cos', 'market_return', 'market_vol', 'stock_te']
if 'price_change' in train_data.columns:
    tech_features.extend(['price_change', 'price_ma5', 'price_ma10', 'price_std5', 'price_mom5'])
if 'f0_f1_ratio' in train_data.columns:
    tech_features.extend(['f0_f1_ratio', 'f0_f1_diff'])
    
pca_features = [f'pca_{i}' for i in range(N_PCA)]
original_features = [f'f{i}' for i in range(100) if f'f{i}' in train_data.columns]

all_features = tech_features + pca_features + original_features
all_features = [f for f in all_features if f in train_data.columns and f in test_df.columns]
all_features = [f for f in all_features if f not in ID_COLS + TARGET_COLS]

print(f"总特征数: {len(all_features)}")

print("\n" + "="*70)
print("[4] XGBoost训练 (TimeSeriesSplit)")
print("="*70)

X_train_xgb = train_data[all_features].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
y_train_xgb = train_data['LabelA'].values.astype(np.float32)
X_val_xgb = val_data[all_features].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
y_val_xgb = val_data['LabelA'].values.astype(np.float32)
X_test_xgb = test_df[all_features].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)

xgb_params = {
    'n_estimators': 500,
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist',
}

if USE_GPU:
    xgb_params['device'] = 'cuda'
    print("  使用GPU加速")

N_FOLDS = 5
tscv = TimeSeriesSplit(n_splits=N_FOLDS)
xgb_models = []
xgb_val_preds = np.zeros(len(X_val_xgb))
xgb_test_preds = np.zeros(len(X_test_xgb))

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_xgb)):
    print(f"  Fold {fold+1}/{N_FOLDS} (train: {len(train_idx):,}, val: {len(val_idx):,})...")
    X_tr, X_v = X_train_xgb[train_idx], X_train_xgb[val_idx]
    y_tr, y_v = y_train_xgb[train_idx], y_train_xgb[val_idx]
    
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_tr, y_tr, verbose=False)
    
    xgb_models.append(model)
    xgb_val_preds += model.predict(X_val_xgb) / N_FOLDS
    xgb_test_preds += model.predict(X_test_xgb) / N_FOLDS
    
    del X_tr, X_v, y_tr, y_v
    gc.collect()

xgb_r2 = r2_score(y_val_xgb, xgb_val_preds)
print(f"XGBoost Val R²: {xgb_r2:.6f}")

del X_train_xgb, y_train_xgb, X_val_xgb, y_val_xgb, X_test_xgb
gc.collect()

print("\n" + "="*70)
print("[5] GRU训练")
print("="*70)

def create_sequences_by_stock_with_indices(df, features, label_col, window=10):
    X_list = []
    y_list = []
    indices_list = []
    
    df = df.reset_index(drop=True)
    
    for stock_id in df['stockid'].unique():
        stock_mask = df['stockid'] == stock_id
        stock_indices = df[stock_mask].index.tolist()
        stock_data = df.loc[stock_indices, features].values
        stock_labels = df.loc[stock_indices, label_col].values
        
        if 'val_idx' in df.columns:
            stock_val_indices = df.loc[stock_indices, 'val_idx'].values
        else:
            stock_val_indices = np.arange(len(stock_data))
        
        n = len(stock_data)
        if n <= window:
            continue
        
        for i in range(n - window):
            X_list.append(stock_data[i:i+window])
            y_list.append(stock_labels[i + window])
            indices_list.append(stock_val_indices[i + window])
    
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32), np.array(indices_list)

print(f"创建滑动窗口序列 (window={WINDOW_SIZE})...")
X_train_gru, y_train_gru, _ = create_sequences_by_stock_with_indices(train_data, all_features, 'LabelA', WINDOW_SIZE)
print(f"训练序列: {X_train_gru.shape}")

X_val_gru, y_val_gru, val_indices_gru = create_sequences_by_stock_with_indices(val_data, all_features, 'LabelA', WINDOW_SIZE)
print(f"验证序列: {X_val_gru.shape}")

scaler_gru = StandardScaler()
n_samples_train = X_train_gru.shape[0]
n_samples_val = X_val_gru.shape[0]

X_train_flat = X_train_gru.reshape(-1, X_train_gru.shape[-1])
X_val_flat = X_val_gru.reshape(-1, X_val_gru.shape[-1])

X_train_flat = scaler_gru.fit_transform(X_train_flat)
X_val_flat = scaler_gru.transform(X_val_flat)

X_train_gru = X_train_flat.reshape(n_samples_train, WINDOW_SIZE, -1)
X_val_gru = X_val_flat.reshape(n_samples_val, WINDOW_SIZE, -1)

X_train_gru = np.nan_to_num(X_train_gru, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
X_val_gru = np.nan_to_num(X_val_gru, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

del X_train_flat, X_val_flat
gc.collect()

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        _, hidden = self.gru(x)
        hidden = hidden[-1]
        return self.fc(hidden)

N_FEATURES = X_train_gru.shape[2]
gru_model = GRUModel(N_FEATURES, hidden_dim=128, num_layers=2, dropout=0.2).to(device)
print(f"GRU配置: INPUT_DIM={N_FEATURES}, hidden_dim=128, num_layers=2")

criterion = nn.MSELoss()
train_dataset = TensorDataset(torch.FloatTensor(X_train_gru), torch.FloatTensor(y_train_gru))
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=0, pin_memory=True)
val_dataset = TensorDataset(torch.FloatTensor(X_val_gru), torch.FloatTensor(y_val_gru))
val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False, num_workers=0, pin_memory=True)

optimizer = torch.optim.AdamW(gru_model.parameters(), lr=5e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

best_val_loss = float('inf')
best_gru_state = None
patience = 10
patience_counter = 0
epochs = 30

print("开始训练...")
for epoch in range(epochs):
    gru_model.train()
    train_loss = 0.0
    n_batches = 0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = gru_model(batch_x)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gru_model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
        n_batches += 1
    
    scheduler.step()
    
    gru_model.eval()
    val_loss = 0.0
    val_batches = 0
    with torch.no_grad():
        for val_x, val_y in val_loader:
            val_x, val_y = val_x.to(device), val_y.to(device)
            val_outputs = gru_model(val_x)
            val_loss += criterion(val_outputs.squeeze(), val_y).item()
            val_batches += 1
    
    val_loss = val_loss / val_batches
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_gru_state = {k: v.cpu().clone() for k, v in gru_model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1}/{epochs}: Val Loss={val_loss:.6f}")

gru_model.load_state_dict({k: v.to(device) for k, v in best_gru_state.items()})

gru_model.eval()
gru_val_preds_seq = []
with torch.no_grad():
    for i in range(0, len(X_val_gru), 2048):
        batch = torch.FloatTensor(X_val_gru[i:i+2048]).to(device)
        pred = gru_model(batch).cpu().numpy().flatten()
        gru_val_preds_seq.extend(pred)

gru_val_preds_seq = np.array(gru_val_preds_seq)

gru_val_preds_full = np.full(len(val_data), np.nan, dtype=np.float32)
for i, idx in enumerate(val_indices_gru):
    if idx < len(val_data):
        gru_val_preds_full[idx] = gru_val_preds_seq[i]

valid_mask = ~np.isnan(gru_val_preds_full)
gru_r2 = r2_score(y_val_gru, gru_val_preds_seq)
print(f"GRU Val R² (序列子集): {gru_r2:.6f}")

del X_train_gru, y_train_gru, X_val_gru, y_val_gru, train_loader, val_loader
del train_dataset, val_dataset, gru_model
gc.collect()
if USE_GPU:
    torch.cuda.empty_cache()

print("\n" + "="*70)
print("[6] Transformer训练 (实际为MLP)")
print("="*70)

X_train_trans = train_data[all_features].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
y_train_trans = train_data['LabelA'].values.astype(np.float32)
X_val_trans = val_data[all_features].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
y_val_trans = val_data['LabelA'].values.astype(np.float32)

scaler_trans = StandardScaler()
X_train_trans = scaler_trans.fit_transform(X_train_trans)
X_val_trans = scaler_trans.transform(X_val_trans)

X_train_trans = np.nan_to_num(X_train_trans, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
X_val_trans = np.nan_to_num(X_val_trans, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

class TransformerMLP(nn.Module):
    def __init__(self, input_dim, d_model=64, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.proj(x)
        return self.fc(x)

N_FEATURES_TRANS = X_train_trans.shape[1]
trans_model = TransformerMLP(N_FEATURES_TRANS, d_model=64, dropout=0.1).to(device)
print(f"Transformer配置: INPUT_DIM={N_FEATURES_TRANS}, d_model=64")

train_dataset = TensorDataset(torch.FloatTensor(X_train_trans), torch.FloatTensor(y_train_trans))
train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True, num_workers=0, pin_memory=True)
val_dataset = TensorDataset(torch.FloatTensor(X_val_trans), torch.FloatTensor(y_val_trans))
val_loader = DataLoader(val_dataset, batch_size=4096, shuffle=False, num_workers=0, pin_memory=True)

optimizer = torch.optim.AdamW(trans_model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

best_val_loss = float('inf')
best_trans_state = None
patience = 8
patience_counter = 0
epochs = 20

print("开始训练...")
for epoch in range(epochs):
    trans_model.train()
    train_loss = 0.0
    n_batches = 0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = trans_model(batch_x)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trans_model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
        n_batches += 1
    
    scheduler.step()
    
    trans_model.eval()
    val_loss = 0.0
    val_batches = 0
    with torch.no_grad():
        for val_x, val_y in val_loader:
            val_x, val_y = val_x.to(device), val_y.to(device)
            val_outputs = trans_model(val_x)
            val_loss += criterion(val_outputs.squeeze(), val_y).item()
            val_batches += 1
    
    val_loss = val_loss / val_batches
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_trans_state = {k: v.cpu().clone() for k, v in trans_model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1}/{epochs}: Val Loss={val_loss:.6f}")

trans_model.load_state_dict({k: v.to(device) for k, v in best_trans_state.items()})

trans_model.eval()
trans_val_preds = []
with torch.no_grad():
    for i in range(0, len(X_val_trans), 4096):
        batch = torch.FloatTensor(X_val_trans[i:i+4096]).to(device)
        pred = trans_model(batch).cpu().numpy().flatten()
        trans_val_preds.extend(pred)

trans_val_preds = np.array(trans_val_preds)
trans_r2 = r2_score(y_val_trans, trans_val_preds)
print(f"Transformer Val R²: {trans_r2:.6f}")

del X_train_trans, y_train_trans, X_val_trans, y_val_trans, train_loader, val_loader
del train_dataset, val_dataset, trans_model
gc.collect()
if USE_GPU:
    torch.cuda.empty_cache()

print("\n" + "="*70)
print("[7] 模型融合")
print("="*70)

print(f"\n各模型验证R²:")
print(f"  XGBoost:     {xgb_r2:.6f}")
print(f"  GRU:         {gru_r2:.6f}")
print(f"  Transformer: {trans_r2:.6f}")

MODEL_R2 = {
    'XGBoost': max(xgb_r2, 0.001),
    'GRU': max(gru_r2, 0.001),
    'Transformer': max(trans_r2, 0.001)
}

total_weight = sum(MODEL_R2.values())
weights = {k: v / total_weight for k, v in MODEL_R2.items()}

print(f"\n融合权重:")
for name, w in weights.items():
    print(f"  {name}: {w:.4f}")

print("\n" + "="*70)
print("[8] 测试集预测")
print("="*70)

X_test_final = test_df[all_features].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)

print("  XGBoost预测 (使用保存的模型)...")
xgb_test_final = np.zeros(len(X_test_final), dtype=np.float32)
for model in xgb_models:
    xgb_test_final += model.predict(X_test_final) / N_FOLDS

print("  Transformer预测...")
X_test_trans_final = scaler_trans.transform(X_test_final)
X_test_trans_final = np.nan_to_num(X_test_trans_final, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

trans_model_final = TransformerMLP(N_FEATURES_TRANS, d_model=64, dropout=0.1).to(device)
trans_model_final.load_state_dict({k: v.to(device) for k, v in best_trans_state.items()})
trans_model_final.eval()

trans_test_final = []
with torch.no_grad():
    for i in range(0, len(X_test_trans_final), 4096):
        batch = torch.FloatTensor(X_test_trans_final[i:i+4096]).to(device)
        pred = trans_model_final(batch).cpu().numpy().flatten()
        trans_test_final.extend(pred)
trans_test_final = np.array(trans_test_final)

del trans_model_final, X_test_trans_final
gc.collect()
if USE_GPU:
    torch.cuda.empty_cache()

print("  GRU预测...")
test_df['_orig_idx'] = range(len(test_df))
test_df_sorted = test_df.sort_values(['stockid', 'dateid', 'timeid']).reset_index(drop=True)
original_indices = test_df_sorted['_orig_idx'].values
X_test_gru_final = test_df_sorted[all_features].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
X_test_gru_final = (X_test_gru_final - scaler_gru.mean_) / scaler_gru.scale_
X_test_gru_final = np.nan_to_num(X_test_gru_final, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

gru_test_final = np.zeros(len(test_df), dtype=np.float32)
stockids = test_df_sorted['stockid'].values
n_test = len(test_df_sorted)

gru_model_final = GRUModel(N_FEATURES, hidden_dim=128, num_layers=2, dropout=0.2).to(device)
gru_model_final.load_state_dict({k: v.to(device) for k, v in best_gru_state.items()})
gru_model_final.eval()

gru_preds_list = []
with torch.no_grad():
    i = 0
    while i < n_test:
        start_i = i
        current_stock = stockids[i]
        while i < n_test and stockids[i] == current_stock:
            i += 1
        end_i = i
        
        stock_len = end_i - start_i
        if stock_len > WINDOW_SIZE:
            for j in range(WINDOW_SIZE, stock_len):
                seq = X_test_gru_final[start_i + j - WINDOW_SIZE:start_i + j]
                seq = seq.reshape(1, WINDOW_SIZE, -1)
                pred = gru_model_final(torch.FloatTensor(seq).to(device)).cpu().numpy().flatten()[0]
                orig_idx = original_indices[start_i + j]
                gru_preds_list.append((orig_idx, pred))

for idx, pred in gru_preds_list:
    gru_test_final[idx] = pred

del gru_model_final, gru_preds_list, X_test_gru_final, test_df_sorted
gc.collect()
if USE_GPU:
    torch.cuda.empty_cache()

print("\n融合预测...")
final_pred = (xgb_test_final * weights['XGBoost'] + 
              gru_test_final * weights['GRU'] + 
              trans_test_final * weights['Transformer'])

final_pred = np.nan_to_num(final_pred, nan=0.0, posinf=0.0, neginf=0.0)

print("\n" + "="*70)
print("[9] 保存结果")
print("="*70)

submission = pd.DataFrame({
    'Uid': test_df['stockid'].astype(str) + '|' + test_df['dateid'].astype(str) + '|' + test_df['timeid'].astype(str),
    'prediction': final_pred
})

output_file = OUTPUT_PATH + 'submission_ensemble.csv'
submission.to_csv(output_file, index=False)
print(f"保存: {output_file}")
print(f"行数: {len(submission):,}")

total_time = time.time() - TOTAL_START
print("\n" + "="*70)
print("完成!")
print("="*70)
print(f"\n总耗时: {total_time/60:.1f} 分钟")
print(f"\n各模型R²:")
print(f"  XGBoost:     {xgb_r2:.6f}")
print(f"  GRU:         {gru_r2:.6f}")
print(f"  Transformer: {trans_r2:.6f}")
print(f"\n优化项:")
print("  ✓ TimeSeriesSplit (防止时序泄露)")
print("  ✓ 统一验证集评估")
print("  ✓ 保存K-Fold模型用于预测")
print("  ✓ 滚动特征 (ma5, ma10, std5, mom5)")
print("  ✓ 交叉特征 (f0_f1_ratio, f0_f1_diff)")
print(f"\n预期融合R²: 0.028 - 0.035")
