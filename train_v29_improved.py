"""
China A-Share Market Microstructure Prediction
V29 改进版 - 基于V28成功经验

改进点：
1. 删除R²为负的模型（HistGradientBoosting、ElasticNet）
2. 添加PyTorch Transformer（替代TensorFlow）
3. XGBoost + LightGBM 双主力 + 加权集成
4. 保留V28成功的特征工程
"""

print("="*70)
print("V29 改进版 - XGBoost+LightGBM双主力 + PyTorch Transformer")
print("="*70)

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA, FastICA
from sklearn.ensemble import ExtraTreesRegressor
import warnings
import gc
import time
import os

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================
# 配置
# ============================================================

def find_data_path():
    possible_paths = [
        '/kaggle/input/competitions/china-a-share-market-microstructure-prediction/',
        '/kaggle/input/',
        '/root/autodl-tmp/data/',
        '/root/autodl-tmp/',
        '/home/user/data/',
        '/home/',
        '/data/',
        '/dataset/',
        '/mnt/data/',
        '/opt/ml/input/data/',
        '/content/',
        './data/',
        './',
    ]
    for path in possible_paths:
        train_path = os.path.join(path, 'train.parquet')
        if os.path.exists(train_path):
            return path
    return './'

def find_output_path():
    possible_paths = [
        '/kaggle/working/',
        '/root/autodl-tmp/',
        '/home/user/output/',
        '/output/',
        '/mnt/output/',
        '/opt/ml/output/',
        '/content/',
        './output/',
        './',
    ]
    for path in possible_paths:
        if os.path.exists(path) and os.access(path, os.W_OK):
            return path
    return './'

BASE_PATH = find_data_path()
OUTPUT_PATH = find_output_path()

print(f"数据路径: {BASE_PATH}")
print(f"输出路径: {OUTPUT_PATH}")

ID_COLS = ['stockid', 'dateid', 'timeid', 'exchangeid']
TARGET_COLS = ['LabelA', 'LabelB', 'LabelC']
FEATURE_COLS = [f'f{i}' for i in range(384)]

BATCH_SIZE = 500000
USE_GPU = True
TRAIN_RATIO = 0.85
N_BAGS = 20
N_FOLDS = 5

TOTAL_START_TIME = time.time()

# ============================================================
# 1. 数据加载
# ============================================================
print("\n[1] 数据加载...")
step_start = time.time()

train_chunks = []
total_rows = 0
for chunk in pd.read_parquet(BASE_PATH + 'train.parquet', chunksize=BATCH_SIZE):
    for col in chunk.columns:
        if col not in ID_COLS and chunk[col].dtype == 'float64':
            chunk[col] = chunk[col].astype('float32')
    train_chunks.append(chunk)
    total_rows += len(chunk)
    print(f"  已加载: {total_rows:,}")

train_df = pd.concat(train_chunks, ignore_index=True)
del train_chunks
gc.collect()

test_df = pd.read_parquet(BASE_PATH + 'test.parquet')
for col in test_df.columns:
    if col not in ID_COLS and test_df[col].dtype == 'float64':
        test_df[col] = test_df[col].astype('float32')

print(f"训练: {train_df.shape}, 测试: {test_df.shape}")
print(f"加载时间: {time.time() - step_start:.1f}秒")

# ============================================================
# 2. 时间序列划分 (先划分!)
# ============================================================
print("\n[2] 时间序列划分 (先划分，防止泄漏)...")

train_df = train_df.sort_values(['stockid', 'dateid', 'timeid']).reset_index(drop=True)
unique_dates = sorted(train_df['dateid'].unique())
n_dates = len(unique_dates)

n_train_dates = int(n_dates * TRAIN_RATIO)
train_dates = unique_dates[:n_train_dates]
val_dates = unique_dates[n_train_dates:]

print(f"训练日期: {len(train_dates)}, 验证日期: {len(val_dates)}")

train_data = train_df[train_df['dateid'].isin(train_dates)].copy()
val_data = train_df[train_df['dateid'].isin(val_dates)].copy()

print(f"训练样本: {len(train_data):,}, 验证样本: {len(val_data):,}")

# ============================================================
# 3. PCA特征 (V28成功配置: 50成分)
# ============================================================
print("\n[3] PCA特征 (50成分)...")

pca_cols = [f'f{i}' for i in range(100)]
pca_cols = [c for c in pca_cols if c in train_data.columns]

pca = PCA(n_components=50, random_state=42)
X_pca_train = pca.fit_transform(train_data[pca_cols].fillna(0).values)
X_pca_val = pca.transform(val_data[pca_cols].fillna(0).values)
X_pca_test = pca.transform(test_df[pca_cols].fillna(0).values)

for i in range(X_pca_train.shape[1]):
    train_data[f'pca_{i}'] = X_pca_train[:, i]
    val_data[f'pca_{i}'] = X_pca_val[:, i]
    test_df[f'pca_{i}'] = X_pca_test[:, i]

pca_features = [f'pca_{i}' for i in range(X_pca_train.shape[1])]
print(f"  PCA特征数: {len(pca_features)}, 解释方差: {pca.explained_variance_ratio_.sum():.4f}")

# ============================================================
# 4. ICA特征 (V28成功配置: 30成分)
# ============================================================
print("\n[4] ICA特征 (30成分)...")

ica_cols = [f'f{i}' for i in range(80)]
ica_cols = [c for c in ica_cols if c in train_data.columns]

ica = FastICA(n_components=30, random_state=42, max_iter=200)
X_ica_train = ica.fit_transform(train_data[ica_cols].fillna(0).values)
X_ica_val = ica.transform(val_data[ica_cols].fillna(0).values)
X_ica_test = ica.transform(test_df[ica_cols].fillna(0).values)

for i in range(X_ica_train.shape[1]):
    train_data[f'ica_{i}'] = X_ica_train[:, i]
    val_data[f'ica_{i}'] = X_ica_val[:, i]
    test_df[f'ica_{i}'] = X_ica_test[:, i]

ica_features = [f'ica_{i}' for i in range(X_ica_train.shape[1])]
print(f"  ICA特征数: {len(ica_features)}")

# ============================================================
# 5. 市场状态分类
# ============================================================
print("\n[5] 市场状态分类...")

regime_features = []
if 'f298' in train_data.columns:
    train_data['price_change'] = train_data.groupby('stockid')['f298'].diff()
    val_data['price_change'] = val_data.groupby('stockid')['f298'].diff()
    test_df['price_change'] = test_df.groupby('stockid')['f298'].diff()

    daily_stats = train_data.groupby('dateid').agg({
        'price_change': ['mean', 'std', 'sum']
    }).reset_index()
    daily_stats.columns = ['dateid', 'market_return', 'market_vol', 'market_trend']

    vol_q = daily_stats['market_vol'].quantile([0.33, 0.66]).values

    train_data = train_data.merge(daily_stats, on='dateid', how='left')

    global_market_return = daily_stats['market_return'].mean()
    global_market_vol = daily_stats['market_vol'].mean()
    global_market_trend = daily_stats['market_trend'].mean()

    val_data['market_return'] = global_market_return
    val_data['market_vol'] = global_market_vol
    val_data['market_trend'] = global_market_trend

    test_df['market_return'] = global_market_return
    test_df['market_vol'] = global_market_vol
    test_df['market_trend'] = global_market_trend

    train_data['vol_regime'] = pd.cut(train_data['market_vol'], bins=[-np.inf, vol_q[0], vol_q[1], np.inf],
                                       labels=[0, 1, 2])
    val_data['vol_regime'] = 1
    test_df['vol_regime'] = 1

    train_data['vol_regime'] = train_data['vol_regime'].astype(float).fillna(1)
    val_data['vol_regime'] = val_data['vol_regime'].astype(float)
    test_df['vol_regime'] = test_df['vol_regime'].astype(float)

    regime_features = ['market_return', 'market_vol', 'market_trend', 'vol_regime']
    print(f"  市场状态特征数: {len(regime_features)}")

# ============================================================
# 6. 傅里叶特征 (V28成功配置: 5谐波)
# ============================================================
print("\n[6] 傅里叶特征...")

n_harmonics = 5
train_data['timeid_norm'] = train_data['timeid'].values / 239.0
val_data['timeid_norm'] = val_data['timeid'].values / 239.0
test_df['timeid_norm'] = test_df['timeid'].values / 239.0

for k in range(1, n_harmonics + 1):
    train_data[f'fourier_sin{k}'] = np.sin(2 * np.pi * k * train_data['timeid_norm'])
    train_data[f'fourier_cos{k}'] = np.cos(2 * np.pi * k * train_data['timeid_norm'])
    val_data[f'fourier_sin{k}'] = np.sin(2 * np.pi * k * val_data['timeid_norm'])
    val_data[f'fourier_cos{k}'] = np.cos(2 * np.pi * k * val_data['timeid_norm'])
    test_df[f'fourier_sin{k}'] = np.sin(2 * np.pi * k * test_df['timeid_norm'])
    test_df[f'fourier_cos{k}'] = np.cos(2 * np.pi * k * test_df['timeid_norm'])

fourier_features = [f'fourier_sin{k}' for k in range(1, n_harmonics + 1)] + \
                   [f'fourier_cos{k}' for k in range(1, n_harmonics + 1)]

# ============================================================
# 7. 跨股票特征
# ============================================================
print("\n[7] 跨股票特征...")

CROSS_FEATURES = [f'f{i}' for i in range(50)]
CROSS_FEATURES = [f for f in CROSS_FEATURES if f in train_data.columns]

cross_features = []
for col in CROSS_FEATURES[:30]:
    cross_mean_train = train_data.groupby(['dateid', 'timeid'])[col].mean().reset_index()
    cross_mean_train.columns = ['dateid', 'timeid', f'{col}_cross_mean']

    train_data = train_data.merge(cross_mean_train, on=['dateid', 'timeid'], how='left')

    cross_std = train_data[f'{col}_cross_mean'].std()

    train_data[f'{col}_zscore'] = (train_data[col] - train_data[f'{col}_cross_mean']) / (cross_std + 1e-8)

    global_cross_mean = train_data[f'{col}_cross_mean'].mean()
    val_data[f'{col}_cross_mean'] = global_cross_mean
    val_data[f'{col}_zscore'] = (val_data[col] - global_cross_mean) / (cross_std + 1e-8)
    test_df[f'{col}_cross_mean'] = global_cross_mean
    test_df[f'{col}_zscore'] = (test_df[col] - global_cross_mean) / (cross_std + 1e-8)

    cross_features.extend([f'{col}_cross_mean', f'{col}_zscore'])

if 'price_change' in train_data.columns:
    market_stats = train_data.groupby(['dateid', 'timeid']).agg({
        'price_change': ['mean', 'std']
    }).reset_index()
    market_stats.columns = ['dateid', 'timeid', 'market_avg_change', 'market_volatility']

    train_data = train_data.merge(market_stats, on=['dateid', 'timeid'], how='left')

    global_avg_change = train_data['market_avg_change'].mean()
    global_volatility = train_data['market_volatility'].mean()

    val_data['market_avg_change'] = global_avg_change
    val_data['market_volatility'] = global_volatility
    test_df['market_avg_change'] = global_avg_change
    test_df['market_volatility'] = global_volatility

    cross_features.extend(['market_avg_change', 'market_volatility'])

print(f"  跨股票特征数: {len(cross_features)}")
gc.collect()

# ============================================================
# 8. 时间序列特征
# ============================================================
print("\n[8] 时间序列特征...")

key_cols = ['f0', 'f1', 'f2', 'f3']
key_cols = [c for c in key_cols if c in train_data.columns]

new_features = []

if key_cols:
    train_data = train_data.sort_values(['stockid', 'dateid', 'timeid']).reset_index(drop=True)
    val_data = val_data.sort_values(['stockid', 'dateid', 'timeid']).reset_index(drop=True)
    test_df = test_df.sort_values(['stockid', 'dateid', 'timeid']).reset_index(drop=True)

    for col in key_cols:
        for window in [5, 10]:
            train_data[f'{col}_ma{window}'] = train_data.groupby('stockid')[col].transform(
                lambda x: x.rolling(window, min_periods=1).mean())
            val_data[f'{col}_ma{window}'] = val_data.groupby('stockid')[col].transform(
                lambda x: x.rolling(window, min_periods=1).mean())
            test_df[f'{col}_ma{window}'] = test_df.groupby('stockid')[col].transform(
                lambda x: x.rolling(window, min_periods=1).mean())
            new_features.append(f'{col}_ma{window}')

            train_data[f'{col}_std{window}'] = train_data.groupby('stockid')[col].transform(
                lambda x: x.rolling(window, min_periods=1).std())
            val_data[f'{col}_std{window}'] = val_data.groupby('stockid')[col].transform(
                lambda x: x.rolling(window, min_periods=1).std())
            test_df[f'{col}_std{window}'] = test_df.groupby('stockid')[col].transform(
                lambda x: x.rolling(window, min_periods=1).std())
            new_features.append(f'{col}_std{window}')

            train_data[f'{col}_ewm{window}'] = train_data.groupby('stockid')[col].transform(
                lambda x: x.ewm(span=window, min_periods=1).mean())
            val_data[f'{col}_ewm{window}'] = val_data.groupby('stockid')[col].transform(
                lambda x: x.ewm(span=window, min_periods=1).mean())
            test_df[f'{col}_ewm{window}'] = test_df.groupby('stockid')[col].transform(
                lambda x: x.ewm(span=window, min_periods=1).mean())
            new_features.append(f'{col}_ewm{window}')

        for lag in [1, 3]:
            train_data[f'{col}_lag{lag}'] = train_data.groupby('stockid')[col].shift(lag)
            val_data[f'{col}_lag{lag}'] = val_data.groupby('stockid')[col].shift(lag)
            test_df[f'{col}_lag{lag}'] = test_df.groupby('stockid')[col].shift(lag)
            new_features.append(f'{col}_lag{lag}')

        train_data[f'{col}_diff1'] = train_data.groupby('stockid')[col].diff(1)
        val_data[f'{col}_diff1'] = val_data.groupby('stockid')[col].diff(1)
        test_df[f'{col}_diff1'] = test_df.groupby('stockid')[col].diff(1)
        new_features.append(f'{col}_diff1')

    if 'f2' in key_cols and 'f3' in key_cols:
        train_data['obi'] = (train_data['f2'] - train_data['f3']) / (train_data['f2'] + train_data['f3'] + 1e-8)
        val_data['obi'] = (val_data['f2'] - val_data['f3']) / (val_data['f2'] + val_data['f3'] + 1e-8)
        test_df['obi'] = (test_df['f2'] - test_df['f3']) / (test_df['f2'] + test_df['f3'] + 1e-8)
        new_features.append('obi')

print(f"  时间序列特征数: {len(new_features)}")
gc.collect()

# ============================================================
# 9. 特征交互
# ============================================================
print("\n[9] 特征交互...")

TOP_8 = ['f298', 'f105', 'f128', 'f28', 'f46', 'f326', 'f124', 'f314']
TOP_8 = [f for f in TOP_8 if f in train_data.columns]

interaction_features = []

for i, f1 in enumerate(TOP_8):
    for f2 in TOP_8[i+1:]:
        train_data[f'{f1}_x_{f2}'] = train_data[f1] * train_data[f2]
        val_data[f'{f1}_x_{f2}'] = val_data[f1] * val_data[f2]
        test_df[f'{f1}_x_{f2}'] = test_df[f1] * test_df[f2]
        interaction_features.append(f'{f1}_x_{f2}')

        train_data[f'{f1}_div_{f2}'] = train_data[f1] / (train_data[f2] + 1e-8)
        val_data[f'{f1}_div_{f2}'] = val_data[f1] / (val_data[f2] + 1e-8)
        test_df[f'{f1}_div_{f2}'] = test_df[f1] / (test_df[f2] + 1e-8)
        interaction_features.append(f'{f1}_div_{f2}')

print(f"  特征交互数: {len(interaction_features)}")

# ============================================================
# 10. 目标编码 (K-Fold防泄漏)
# ============================================================
print("\n[10] 目标编码...")

global_mean = train_data['LabelA'].mean()

train_data['stock_target_enc_a'] = global_mean
train_data['stock_target_enc_b'] = global_mean
train_data['stock_target_enc_c'] = global_mean

kf_enc = KFold(n_splits=5, shuffle=True, random_state=42)
train_idx_list = list(range(len(train_data)))
for _, val_idx in kf_enc.split(train_idx_list):
    enc_train = train_data.iloc[~train_data.index.isin(val_idx)]
    enc_val = train_data.iloc[val_idx]

    stock_means_a = enc_train.groupby('stockid')['LabelA'].mean()
    stock_means_b = enc_train.groupby('stockid')['LabelB'].mean()
    stock_means_c = enc_train.groupby('stockid')['LabelC'].mean()

    for idx in enc_val.index:
        stockid = train_data.loc[idx, 'stockid']
        train_data.loc[idx, 'stock_target_enc_a'] = stock_means_a.get(stockid, global_mean)
        train_data.loc[idx, 'stock_target_enc_b'] = stock_means_b.get(stockid, global_mean)
        train_data.loc[idx, 'stock_target_enc_c'] = stock_means_c.get(stockid, global_mean)

stock_means_a_full = train_data.groupby('stockid')['LabelA'].mean()
stock_means_b_full = train_data.groupby('stockid')['LabelB'].mean()
stock_means_c_full = train_data.groupby('stockid')['LabelC'].mean()

val_data['stock_target_enc_a'] = val_data['stockid'].map(stock_means_a_full).fillna(global_mean)
val_data['stock_target_enc_b'] = val_data['stockid'].map(stock_means_b_full).fillna(global_mean)
val_data['stock_target_enc_c'] = val_data['stockid'].map(stock_means_c_full).fillna(global_mean)
test_df['stock_target_enc_a'] = test_df['stockid'].map(stock_means_a_full).fillna(global_mean)
test_df['stock_target_enc_b'] = test_df['stockid'].map(stock_means_b_full).fillna(global_mean)
test_df['stock_target_enc_c'] = test_df['stockid'].map(stock_means_c_full).fillna(global_mean)

new_features.extend(['stock_target_enc_a', 'stock_target_enc_b', 'stock_target_enc_c'])

if 'exchangeid' in train_data.columns:
    exchange_means = train_data.groupby('exchangeid')['LabelA'].mean()
    train_data['exchange_target_enc'] = train_data['exchangeid'].map(exchange_means).fillna(global_mean)
    val_data['exchange_target_enc'] = val_data['exchangeid'].map(exchange_means).fillna(global_mean)
    test_df['exchange_target_enc'] = test_df['exchangeid'].map(exchange_means).fillna(global_mean)
    new_features.append('exchange_target_enc')

time_means = train_data.groupby('timeid')['LabelA'].mean()
train_data['time_target_enc'] = train_data['timeid'].map(time_means).fillna(global_mean)
val_data['time_target_enc'] = val_data['timeid'].map(time_means).fillna(global_mean)
test_df['time_target_enc'] = test_df['timeid'].map(time_means).fillna(global_mean)
new_features.append('time_target_enc')

# ============================================================
# 11. 样本权重
# ============================================================
print("\n[11] 样本权重计算...")

if 'price_change' in train_data.columns:
    train_data['market_vol_local'] = train_data.groupby(['dateid', 'timeid'])['price_change'].transform('std')

    vol_quantile = train_data['market_vol_local'].quantile(0.75)
    train_data['vol_weight'] = np.where(train_data['market_vol_local'] > vol_quantile, 1.5, 1.0)

    val_data['vol_weight'] = 1.0
    test_df['vol_weight'] = 1.0
else:
    train_data['vol_weight'] = np.ones(len(train_data))
    val_data['vol_weight'] = np.ones(len(val_data))
    test_df['vol_weight'] = np.ones(len(test_df))

train_data['time_weight'] = np.where(train_data['timeid'] < 229, 1.0, 0.1)
val_data['time_weight'] = np.where(val_data['timeid'] < 229, 1.0, 0.1)
test_df['time_weight'] = np.where(test_df['timeid'] < 229, 1.0, 0.1)

train_data['sample_weight'] = train_data['vol_weight'] * train_data['time_weight']
val_data['sample_weight'] = np.ones(len(val_data))
test_df['sample_weight'] = np.ones(len(test_df))

sample_weights = train_data['sample_weight'].values

print("  样本权重计算完成")

# ============================================================
# 12. 准备特征
# ============================================================
print("\n[12] 准备特征...")

all_features = (FEATURE_COLS + cross_features + new_features + fourier_features +
                interaction_features + pca_features + ica_features + regime_features)
all_features = [f for f in all_features if f in train_data.columns and f in val_data.columns and f in test_df.columns]
all_features = list(set(all_features))

print(f"总特征数: {len(all_features)}")

train_data[all_features] = train_data[all_features].fillna(0)
val_data[all_features] = val_data[all_features].fillna(0)
test_df[all_features] = test_df[all_features].fillna(0)

X_train_full = train_data[all_features].values.astype('float32')
y_train_a = train_data['LabelA'].values.astype('float32')
y_train_b = train_data['LabelB'].values.astype('float32')
y_train_c = train_data['LabelC'].values.astype('float32')
X_val = val_data[all_features].values.astype('float32')
y_val = val_data['LabelA'].values.astype('float32')

print(f"训练数据: {X_train_full.shape}, 验证数据: {X_val.shape}")

# ============================================================
# 13. 超参数
# ============================================================
print("\n[13] 超参数设置...")

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'n_estimators': 1500,
    'learning_rate': 0.02,
    'max_depth': 12,
    'num_leaves': 256,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.5,
    'reg_lambda': 0.5,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1,
}

xgb_params = {
    'objective': 'reg:squarederror',
    'n_estimators': 1500,
    'learning_rate': 0.02,
    'max_depth': 12,
    'min_child_weight': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.5,
    'reg_lambda': 0.5,
    'random_state': 42,
    'tree_method': 'gpu_hist' if USE_GPU else 'hist',
}

print("  使用预设超参数")

# ============================================================
# 14. LightGBM (主力模型1)
# ============================================================
print("\n[14] LightGBM (主力模型1)...")

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
lgb_cv_models = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full)):
    X_tr, X_vl = X_train_full[train_idx], X_train_full[val_idx]
    y_tr = y_train_a[train_idx]
    sw_tr = sample_weights[train_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X_tr, y_tr, sample_weight=sw_tr)
    lgb_cv_models.append(model)
    print(f"  Fold {fold+1}/{N_FOLDS} 完成")

lgb_bag_preds = []
lgb_bag_models = []
for i in range(N_BAGS):
    lgb_params_bag = {**lgb_params, 'random_state': 42 + i}
    model = lgb.LGBMRegressor(**lgb_params_bag)
    model.fit(X_train_full, y_train_a, sample_weight=sample_weights)
    lgb_bag_preds.append(model.predict(X_val))
    lgb_bag_models.append(model)
    if (i + 1) % 5 == 0:
        print(f"  Bag {i+1}/{N_BAGS} 完成")

lgb_val_pred = np.mean(lgb_bag_preds, axis=0)
lgb_r2 = r2_score(y_val, lgb_val_pred)
print(f"LightGBM R²: {lgb_r2:.6f}")

# ============================================================
# 15. XGBoost (主力模型2)
# ============================================================
print("\n[15] XGBoost (主力模型2)...")

xgb_bag_preds = []
xgb_bag_models = []
for i in range(15):
    xgb_params_bag = {**xgb_params, 'random_state': 42 + i}
    model = xgb.XGBRegressor(**xgb_params_bag)
    model.fit(X_train_full, y_train_a, sample_weight=sample_weights, verbose=0)
    xgb_bag_preds.append(model.predict(X_val))
    xgb_bag_models.append(model)
    if (i + 1) % 5 == 0:
        print(f"  Bag {i+1}/15 完成")

xgb_val_pred = np.mean(xgb_bag_preds, axis=0)
xgb_r2 = r2_score(y_val, xgb_val_pred)
print(f"XGBoost R²: {xgb_r2:.6f}")

# ============================================================
# 16. ExtraTrees
# ============================================================
print("\n[16] ExtraTrees...")

et_model = ExtraTreesRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
et_model.fit(X_train_full, y_train_a, sample_weight=sample_weights)
et_val_pred = et_model.predict(X_val)
et_r2 = r2_score(y_val, et_val_pred)
print(f"ExtraTrees R²: {et_r2:.6f}")

# ============================================================
# 17. Ridge (稳定基线)
# ============================================================
print("\n[17] Ridge...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_val_scaled = scaler.transform(X_val)

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train_a, sample_weight=sample_weights)
ridge_val_pred = ridge_model.predict(X_val_scaled)
ridge_r2 = r2_score(y_val, ridge_val_pred)
print(f"Ridge R²: {ridge_r2:.6f}")

# ============================================================
# 18. MLP
# ============================================================
print("\n[18] MLP...")

mlp_model = MLPRegressor(
    hidden_layer_sizes=(512, 256, 128),
    activation='relu',
    solver='adam',
    alpha=0.001,
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42,
    verbose=False
)
mlp_model.fit(X_train_scaled, y_train_a)
mlp_val_pred = mlp_model.predict(X_val_scaled)
mlp_r2 = r2_score(y_val, mlp_val_pred)
print(f"MLP R²: {mlp_r2:.6f}")

# ============================================================
# 19. PyTorch Transformer
# ============================================================
print("\n[19] PyTorch Transformer...")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  PyTorch设备: {DEVICE}")

    class SimpleTransformer(nn.Module):
        def __init__(self, n_features, embed_dim=128, num_heads=4, ff_dim=256, dropout=0.1, num_layers=2):
            super().__init__()
            self.feat_proj = nn.Sequential(
                nn.Linear(n_features, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                device=DEVICE
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            self.head = nn.Sequential(
                nn.Linear(embed_dim, 64),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1)
            )

        def forward(self, x):
            x = self.feat_proj(x)
            x = self.transformer(x)
            x = x.mean(dim=1)
            return self.head(x).squeeze(-1)

    n_features = X_train_full.shape[1]
    X_train_seq = X_train_scaled.reshape(X_train_scaled.shape[0], 1, n_features).astype('float32')
    X_val_seq = X_val_scaled.reshape(X_val_scaled.shape[0], 1, n_features).astype('float32')

    trans_val_preds = []
    for seed in [42, 2023]:
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = SimpleTransformer(n_features=n_features).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_seq),
            torch.FloatTensor(y_train_a)
        )
        train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)

        best_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(50):
            model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                optimizer.zero_grad()
                pred = model(batch_x)
                loss = nn.functional.mse_loss(pred, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            scheduler.step()

            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_tensor = torch.FloatTensor(X_val_seq).to(DEVICE)
                    val_pred = model(val_tensor).cpu().numpy()
                    val_r2 = r2_score(y_val, val_pred)
                    print(f"    Epoch {epoch+1}: train_loss={train_loss/len(train_loader):.6f}, val_r2={val_r2:.6f}")

                if train_loss/len(train_loader) < best_loss - 1e-5:
                    best_loss = train_loss/len(train_loader)
                    patience_counter = 0
                    best_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"    Early stopping at epoch {epoch+1}")
                        break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            val_tensor = torch.FloatTensor(X_val_seq).to(DEVICE)
            val_pred = model(val_tensor).cpu().numpy()
            trans_val_preds.append(val_pred)

    trans_val_pred = np.mean(trans_val_preds, axis=0)
    trans_r2 = r2_score(y_val, trans_val_pred)
    print(f"Transformer R²: {trans_r2:.6f}")
    USE_TRANSFORMER = True

except Exception as e:
    print(f"Transformer训练失败: {e}")
    trans_val_pred = np.zeros(len(y_val))
    trans_r2 = -999
    USE_TRANSFORMER = False

# ============================================================
# 20. 加权集成 (删除R²为负的模型)
# ============================================================
print("\n[20] 加权集成...")

model_preds = {
    'lgb': lgb_val_pred,
    'xgb': xgb_val_pred,
    'et': et_val_pred,
    'ridge': ridge_val_pred,
    'mlp': mlp_val_pred,
}

if USE_TRANSFORMER and trans_r2 > 0:
    model_preds['transformer'] = trans_val_pred

print("\n各模型验证集R²:")
valid_models = {}
for name, pred in model_preds.items():
    r2 = r2_score(y_val, pred)
    print(f"  {name}: {r2:.6f}")
    if r2 > 0:
        valid_models[name] = (r2, pred)

total_inv_r2 = sum(1/r2 for r2, _ in valid_models.values())
weights = {name: (1/r2)/total_inv_r2 for name, (r2, _) in valid_models.items()}

print("\n集成权重:")
for name, w in sorted(weights.items(), key=lambda x: -x[1]):
    print(f"  {name}: {w:.4f}")

weighted_pred = sum(pred * weights[name] for name, (_, pred) in valid_models.items())
weighted_r2 = r2_score(y_val, weighted_pred)
print(f"\n加权集成 R²: {weighted_r2:.6f}")

# ============================================================
# 21. 测试集预测
# ============================================================
print("\n[21] 测试集预测...")

test_features = test_df[all_features].fillna(0).values.astype('float32')
X_test_scaled = scaler.transform(test_features)

lgb_test_pred = np.mean([m.predict(test_features) for m in lgb_bag_models], axis=0)
xgb_test_pred = np.mean([m.predict(test_features) for m in xgb_bag_models], axis=0)
et_test_pred = et_model.predict(test_features)
ridge_test_pred = ridge_model.predict(X_test_scaled)
mlp_test_pred = mlp_model.predict(X_test_scaled)

test_preds = {
    'lgb': lgb_test_pred,
    'xgb': xgb_test_pred,
    'et': et_test_pred,
    'ridge': ridge_test_pred,
    'mlp': mlp_test_pred,
}

if USE_TRANSFORMER and trans_r2 > 0:
    X_test_seq = X_test_scaled.reshape(X_test_scaled.shape[0], 1, n_features).astype('float32')
    trans_test_preds = []
    for seed in [42, 2023]:
        torch.manual_seed(seed)
        model = SimpleTransformer(n_features=n_features).to(DEVICE)
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            test_tensor = torch.FloatTensor(X_test_seq).to(DEVICE)
            test_pred = model(test_tensor).cpu().numpy()
            trans_test_preds.append(test_pred)
    trans_test_pred = np.mean(trans_test_preds, axis=0)
    test_preds['transformer'] = trans_test_pred

final_pred = sum(test_preds[name] * weights.get(name, 0) for name in test_preds if name in weights)

print(f"测试集预测完成: {len(final_pred):,}")

# ============================================================
# 22. 生成提交文件
# ============================================================
print("\n[22] 生成提交文件...")

test_ids = pd.read_parquet(BASE_PATH + 'test.parquet', columns=['stockid', 'dateid', 'timeid'])

submission = pd.DataFrame({
    'Uid': test_ids['stockid'].astype(str) + '|' +
           test_ids['dateid'].astype(str) + '|' +
           test_ids['timeid'].astype(str),
    'prediction': final_pred
})

submission.to_csv(OUTPUT_PATH + 'submission_v29.csv', index=False)
print(f"提交文件保存: {OUTPUT_PATH}submission_v29.csv")

# ============================================================
# 总结
# ============================================================
print("\n" + "="*70)
print("V29 改进版完成！")
print("="*70)

total_time = time.time() - TOTAL_START_TIME
print(f"总耗时: {total_time/60:.1f} 分钟")

print(f"\n=== 模型性能 ===")
all_r2 = {'LightGBM': lgb_r2, 'XGBoost': xgb_r2, 'ExtraTrees': et_r2,
          'Ridge': ridge_r2, 'MLP': mlp_r2, 'Weighted_Ensemble': weighted_r2}
if USE_TRANSFORMER and trans_r2 > 0:
    all_r2['Transformer'] = trans_r2

for name, r2 in sorted(all_r2.items(), key=lambda x: -x[1]):
    print(f"  {name}: R² = {r2:.6f}")

print(f"\n最佳模型: {max(all_r2, key=all_r2.get)}, R² = {max(all_r2.values()):.6f}")
print(f"\n验证集R²: {weighted_r2:.4f}")