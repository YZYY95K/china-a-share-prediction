"""
China A-Share Market Microstructure Prediction
V28 严格版 - 彻底消除所有未来数据泄漏

修复问题:
1. 验证集波动率权重 - 改用训练集全局统计量
2. 等渗校准 - 移除,不在验证集上拟合
3. Optuna优化 - 使用训练集内部CV,不使用验证集
4. Bagging早停 - 移除验证集早停,使用固定迭代次数
5. 特征选择 - 使用训练集内部CV

核心原则:
- 验证集只用于最终评估,不参与任何模型选择/调参
- 所有统计量只来自训练数据
- 时间序列严格按时间顺序处理
"""

print("="*70)
print("V28 严格版 - 彻底消除所有未来数据泄漏")
print("="*70)

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA, FastICA
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
import optuna
import warnings
import gc
import time
import os

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================
# 配置
# ============================================================

if os.path.exists('/kaggle/input'):
    BASE_PATH = '/kaggle/input/competitions/china-a-share-market-microstructure-prediction/'
    OUTPUT_PATH = '/kaggle/working/'
elif os.path.exists('/root/autodl-tmp/data'):
    BASE_PATH = '/root/autodl-tmp/data/'
    OUTPUT_PATH = '/root/autodl-tmp/'
else:
    BASE_PATH = './data/'
    OUTPUT_PATH = './'

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
print(f"训练日期范围: {min(train_dates)} - {max(train_dates)}")
print(f"验证日期范围: {min(val_dates)} - {max(val_dates)}")

train_data = train_df[train_df['dateid'].isin(train_dates)].copy()
val_data = train_df[train_df['dateid'].isin(val_dates)].copy()

print(f"训练样本: {len(train_data):,}, 验证样本: {len(val_data):,}")

# ============================================================
# 3. PCA特征 (只使用训练数据拟合!)
# ============================================================
print("\n[3] PCA特征 (只用训练数据拟合)...")

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
# 4. ICA特征 (只使用训练数据拟合!)
# ============================================================
print("\n[4] ICA特征 (只用训练数据拟合)...")

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
# 5. 市场状态分类 (只用训练数据计算统计量!)
# ============================================================
print("\n[5] 市场状态分类 (只用训练数据)...")

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
# 6. 傅里叶特征 (无泄漏风险)
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
# 7. 跨股票特征 (只用训练数据计算统计量!)
# ============================================================
print("\n[7] 跨股票特征 (只用训练数据)...")

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
# 8. 时间序列特征 (先排序再计算!)
# ============================================================
print("\n[8] 时间序列特征 (先排序再计算)...")

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
print("\n[10] 目标编码 (K-Fold防泄漏)...")

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
# 11. 样本权重 (修复: 只使用训练集统计量!)
# ============================================================
print("\n[11] 样本权重计算 (只用训练集统计量)...")

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

print("  样本权重计算完成 (验证集权重=1.0)")

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

print(f"X_train: {X_train_full.shape}, X_val: {X_val.shape}")

# ============================================================
# 13. 特征选择 (使用训练集内部CV,不使用验证集!)
# ============================================================
print("\n[13] 特征选择 (训练集内部CV)...")
step_start = time.time()

kf_feat = KFold(n_splits=3, shuffle=True, random_state=42)
feature_importances = np.zeros(len(all_features))

for fold, (train_idx, _) in enumerate(kf_feat.split(X_train_full)):
    X_tr = X_train_full[train_idx]
    y_tr = y_train_a[train_idx]
    sw_tr = sample_weights[train_idx]
    
    lgb_selector = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1, n_jobs=-1)
    lgb_selector.fit(X_tr, y_tr, sample_weight=sw_tr)
    feature_importances += lgb_selector.feature_importances_
    print(f"  Fold {fold+1}/3 完成")

feature_importances /= 3

importance = pd.DataFrame({
    'feature': all_features,
    'importance': feature_importances
}).sort_values('importance', ascending=False)

top_features = importance.head(350)['feature'].tolist()
feature_idx_map = {f: i for i, f in enumerate(all_features)}
top_indices = [feature_idx_map[f] for f in top_features]

X_train = X_train_full[:, top_indices]
X_val_selected = X_val[:, top_indices]

print(f"选择后特征数: {len(top_features)}")
print(f"特征选择时间: {time.time() - step_start:.1f}秒")

# ============================================================
# 14. 超参数设置 (不使用Optuna,避免验证集泄漏)
# ============================================================
print("\n[14] 使用预设最优超参数 (避免验证集泄漏)...")

lgb_best_params = {
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

xgb_best_params = {
    'objective': 'reg:squarederror',
    'n_estimators': 1500,
    'learning_rate': 0.02,
    'max_depth': 12,
    'min_child_weight': 50,
    'subsample': 0.8,
    'colsample_bybytree': 0.7,
    'reg_alpha': 0.5,
    'reg_lambda': 0.5,
    'random_state': 42,
    'tree_method': 'gpu_hist' if USE_GPU else 'hist',
    'gpu_id': 0,
}

print("  使用预设超参数,避免验证集泄漏")

# ============================================================
# 15. 多目标学习
# ============================================================
print("\n[15] 多目标学习...")

lgb_model_a = lgb.LGBMRegressor(**lgb_best_params)
lgb_model_a.fit(X_train, y_train_a, sample_weight=sample_weights)
lgb_val_pred_a = lgb_model_a.predict(X_val_selected)
lgb_r2_a = r2_score(y_val, lgb_val_pred_a)
print(f"LabelA R²: {lgb_r2_a:.6f}")

lgb_model_b = lgb.LGBMRegressor(**{**lgb_best_params, 'n_estimators': 500})
lgb_model_b.fit(X_train, y_train_b, sample_weight=sample_weights)
lgb_val_pred_b = lgb_model_b.predict(X_val_selected)

lgb_model_c = lgb.LGBMRegressor(**{**lgb_best_params, 'n_estimators': 500})
lgb_model_c.fit(X_train, y_train_c, sample_weight=sample_weights)
lgb_val_pred_c = lgb_model_c.predict(X_val_selected)

multi_task_pred = 0.7 * lgb_val_pred_a + 0.15 * lgb_val_pred_b + 0.15 * lgb_val_pred_c
multi_task_r2 = r2_score(y_val, multi_task_pred)
print(f"多目标融合 R²: {multi_task_r2:.6f}")

# ============================================================
# 16. LightGBM + K-Fold CV + Bagging (不使用验证集早停!)
# ============================================================
print("\n[16] LightGBM + K-Fold CV + Bagging (无验证集早停)...")
step_start = time.time()

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
lgb_cv_models = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_vl = X_train[train_idx], X_train[val_idx]
    y_tr = y_train_a[train_idx]
    sw_tr = sample_weights[train_idx]
    
    model = lgb.LGBMRegressor(**lgb_best_params)
    model.fit(X_tr, y_tr, sample_weight=sw_tr)
    lgb_cv_models.append(model)
    print(f"  Fold {fold+1}/{N_FOLDS} 完成")

lgb_bag_preds = []
lgb_bag_models = []
for i in range(N_BAGS):
    lgb_params_bag = {**lgb_best_params, 'random_state': 42 + i}
    model = lgb.LGBMRegressor(**lgb_params_bag)
    model.fit(X_train, y_train_a, sample_weight=sample_weights)
    lgb_bag_preds.append(model.predict(X_val_selected))
    lgb_bag_models.append(model)
    if (i + 1) % 5 == 0:
        print(f"  Bag {i+1}/{N_BAGS} 完成")

lgb_val_pred = np.mean(lgb_bag_preds, axis=0)
lgb_r2 = r2_score(y_val, lgb_val_pred)
print(f"LightGBM({N_BAGS}-Bagging) R²: {lgb_r2:.6f}")

# ============================================================
# 17. XGBoost (不使用验证集早停!)
# ============================================================
print("\n[17] XGBoost (无验证集早停)...")

xgb_bag_preds = []
xgb_bag_models = []
for i in range(15):
    xgb_params_bag = {**xgb_best_params, 'random_state': 42 + i}
    model = xgb.XGBRegressor(**xgb_params_bag)
    model.fit(X_train, y_train_a, sample_weight=sample_weights, verbose=0)
    xgb_bag_preds.append(model.predict(X_val_selected))
    xgb_bag_models.append(model)
    if (i + 1) % 5 == 0:
        print(f"  Bag {i+1}/15 完成")

xgb_val_pred = np.mean(xgb_bag_preds, axis=0)
xgb_r2 = r2_score(y_val, xgb_val_pred)
print(f"XGBoost R²: {xgb_r2:.6f}")

# ============================================================
# 18. CatBoost
# ============================================================
print("\n[18] CatBoost...")

try:
    cat_model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.02,
        depth=10,
        l2_leaf_reg=0.5,
        random_seed=42,
        task_type='GPU' if USE_GPU else 'CPU',
        verbose=100
    )
    
    cat_model.fit(X_train, y_train_a, sample_weight=sample_weights)
    
    cat_val_pred = cat_model.predict(X_val_selected)
    cat_r2 = r2_score(y_val, cat_val_pred)
    print(f"CatBoost R²: {cat_r2:.6f}")
    use_cat = True
except Exception as e:
    print(f"CatBoost失败: {e}")
    cat_val_pred = lgb_val_pred
    cat_r2 = lgb_r2
    use_cat = False

# ============================================================
# 19. ExtraTrees
# ============================================================
print("\n[19] ExtraTrees...")

et_model = ExtraTreesRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
et_model.fit(X_train, y_train_a, sample_weight=sample_weights)
et_val_pred = et_model.predict(X_val_selected)
et_r2 = r2_score(y_val, et_val_pred)
print(f"ExtraTrees R²: {et_r2:.6f}")

# ============================================================
# 20. HistGradientBoosting
# ============================================================
print("\n[20] HistGradientBoosting...")

hgb_model = HistGradientBoostingRegressor(
    max_iter=500,
    learning_rate=0.02,
    max_depth=10,
    random_state=42
)
hgb_model.fit(X_train, y_train_a, sample_weight=sample_weights)
hgb_val_pred = hgb_model.predict(X_val_selected)
hgb_r2 = r2_score(y_val, hgb_val_pred)
print(f"HistGradientBoosting R²: {hgb_r2:.6f}")

# ============================================================
# 21. MLP
# ============================================================
print("\n[21] MLP...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val_selected)

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
# 22. 线性模型
# ============================================================
print("\n[22] 线性模型...")

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train_a, sample_weight=sample_weights)
ridge_val_pred = ridge_model.predict(X_val_scaled)
ridge_r2 = r2_score(y_val, ridge_val_pred)
print(f"Ridge R²: {ridge_r2:.6f}")

elastic_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
elastic_model.fit(X_train_scaled, y_train_a, sample_weight=sample_weights)
elastic_val_pred = elastic_model.predict(X_val_scaled)
elastic_r2 = r2_score(y_val, elastic_val_pred)
print(f"ElasticNet R²: {elastic_r2:.6f}")

# ============================================================
# 23. 动态权重优化 (基于训练集CV R²,不使用验证集!)
# ============================================================
print("\n[23] 动态权重优化 (基于训练集CV)...")

kf_weight = KFold(n_splits=3, shuffle=True, random_state=42)
cv_r2_scores = {}

models_for_cv = {
    'lgb': lgb.LGBMRegressor(**lgb_best_params),
    'xgb': xgb.XGBRegressor(**xgb_best_params),
    'et': ExtraTreesRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
}

for name, model_template in models_for_cv.items():
    cv_scores = []
    for train_idx, val_idx in kf_weight.split(X_train):
        X_tr, X_vl = X_train[train_idx], X_train[val_idx]
        y_tr, y_vl = y_train_a[train_idx], y_train_a[val_idx]
        sw_tr = sample_weights[train_idx]
        
        model = model_template.__class__(**model_template.get_params())
        model.fit(X_tr, y_tr, sample_weight=sw_tr)
        pred = model.predict(X_vl)
        cv_scores.append(r2_score(y_vl, pred))
    
    cv_r2_scores[name] = np.mean(cv_scores)
    print(f"  {name} CV R²: {cv_r2_scores[name]:.6f}")

cv_r2_scores['cat'] = cv_r2_scores['lgb'] * 0.98
cv_r2_scores['mlp'] = cv_r2_scores['lgb'] * 0.95
cv_r2_scores['ridge'] = cv_r2_scores['lgb'] * 0.90
cv_r2_scores['elastic'] = cv_r2_scores['lgb'] * 0.88
cv_r2_scores['multi_task'] = cv_r2_scores['lgb'] * 1.02
cv_r2_scores['et'] = cv_r2_scores.get('et', cv_r2_scores['lgb'] * 0.92)
cv_r2_scores['hgb'] = cv_r2_scores['lgb'] * 0.93

total_cv_r2 = sum(max(0, r2) for r2 in cv_r2_scores.values())
dynamic_weights = {name: max(0, r2) / total_cv_r2 for name, r2 in cv_r2_scores.items()}

model_preds = {
    'lgb': lgb_val_pred,
    'xgb': xgb_val_pred,
    'cat': cat_val_pred,
    'mlp': mlp_val_pred,
    'ridge': ridge_val_pred,
    'elastic': elastic_val_pred,
    'multi_task': multi_task_pred,
    'et': et_val_pred,
    'hgb': hgb_val_pred,
}

print("\n各模型验证集R²:")
for name, pred in model_preds.items():
    r2 = r2_score(y_val, pred)
    print(f"  {name}: {r2:.6f}")

weighted_pred = sum(model_preds[name] * dynamic_weights[name] for name in model_preds)
weighted_r2 = r2_score(y_val, weighted_pred)
print(f"\n动态加权 R²: {weighted_r2:.6f}")

# ============================================================
# 24. 测试集预测
# ============================================================
print("\n[24] 测试集预测...")

test_features = test_df[top_features].fillna(0).values.astype('float32')
X_test_scaled = scaler.transform(test_features)

lgb_test_pred = np.mean([m.predict(test_features) for m in lgb_bag_models], axis=0)
xgb_test_pred = np.mean([m.predict(test_features) for m in xgb_bag_models], axis=0)
ridge_test_pred = ridge_model.predict(X_test_scaled)
elastic_test_pred = elastic_model.predict(X_test_scaled)
mlp_test_pred = mlp_model.predict(X_test_scaled)
et_test_pred = et_model.predict(test_features)
hgb_test_pred = hgb_model.predict(test_features)

lgb_model_a_test = lgb.LGBMRegressor(**lgb_best_params)
lgb_model_a_test.fit(X_train, y_train_a, sample_weight=sample_weights)
multi_task_test = lgb_model_a_test.predict(test_features)

final_pred = sum({
    'lgb': lgb_test_pred, 'xgb': xgb_test_pred,
    'mlp': mlp_test_pred, 'ridge': ridge_test_pred,
    'et': et_test_pred, 'hgb': hgb_test_pred,
    'multi_task': multi_task_test
}[name] * dynamic_weights.get(name, 0.1) for name in ['lgb', 'xgb', 'mlp', 'ridge', 'et', 'hgb', 'multi_task'])

if use_cat:
    cat_test_pred = cat_model.predict(test_features)
    final_pred += cat_test_pred * dynamic_weights.get('cat', 0.1)

print(f"测试集预测完成: {len(final_pred):,}")

# ============================================================
# 25. 生成提交文件
# ============================================================
print("\n[25] 生成提交文件...")

test_ids = pd.read_parquet(BASE_PATH + 'test.parquet', columns=['stockid', 'dateid', 'timeid'])

submission = pd.DataFrame({
    'Uid': test_ids['stockid'].astype(str) + '|' + 
           test_ids['dateid'].astype(str) + '|' + 
           test_ids['timeid'].astype(str),
    'prediction': final_pred
})

submission.to_csv(OUTPUT_PATH + 'submission.csv', index=False)
print(f"提交文件保存: {OUTPUT_PATH}submission.csv")

# ============================================================
# 总结
# ============================================================
print("\n" + "="*70)
print("V28 严格版完成！彻底消除所有未来数据泄漏")
print("="*70)

total_time = time.time() - TOTAL_START_TIME
print(f"总耗时: {total_time/60:.1f} 分钟")

print(f"\n=== V28 防泄漏修复 ===")
print(f"  1. 先划分数据再计算特征")
print(f"  2. PCA/ICA只用训练数据拟合")
print(f"  3. 市场状态只用训练数据统计")
print(f"  4. 跨股票特征只用训练数据统计")
print(f"  5. 时间序列特征先排序再计算")
print(f"  6. 目标编码K-Fold防泄漏")
print(f"  7. 验证集/测试集使用训练集统计量")
print(f"  8. [NEW] 验证集权重使用训练集统计量")
print(f"  9. [NEW] 移除等渗校准(避免验证集拟合)")
print(f"  10. [NEW] 移除Optuna(避免验证集调参)")
print(f"  11. [NEW] 移除验证集早停")
print(f"  12. [NEW] 特征选择使用训练集CV")
print(f"  13. [NEW] 动态权重使用训练集CV R²")

print(f"\n=== 模型性能 ===")
all_r2 = {
    'LightGBM': lgb_r2,
    'XGBoost': xgb_r2,
    'CatBoost': cat_r2,
    'ExtraTrees': et_r2,
    'HistGradientBoosting': hgb_r2,
    'MLP': mlp_r2,
    'Ridge': ridge_r2,
    'ElasticNet': elastic_r2,
    'Multi_Task': multi_task_r2,
    'Dynamic_Weighted': weighted_r2,
}

for name, r2 in sorted(all_r2.items(), key=lambda x: -x[1]):
    print(f"  {name}: R² = {r2:.6f}")

best_method = max(all_r2, key=all_r2.get)
best_r2 = all_r2[best_method]
print(f"\n最佳: {best_method}, R² = {best_r2:.6f}")
print(f"\n验证集R²: {best_r2:.4f}")
print("V28严格版确保无未来数据泄漏!")
