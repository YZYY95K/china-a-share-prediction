"""
China A-Share Market Microstructure Prediction
🏆 V23 极限优化版 - PCA/ICA + 市场状态 + 更多搜索

包含优化:
1. ⭐⭐⭐⭐⭐ PCA/ICA特征工程
2. ⭐⭐⭐⭐⭐ 市场状态分类
3. ⭐⭐⭐⭐⭐ Optuna 150次
4. ⭐⭐⭐⭐⭐ 15次Bagging
"""

print("="*70)
print("🏆 V23 极限优化版 - PCA/ICA + 市场状态 + 更多搜索")
print("="*70)

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA, FastICA
import optuna
import warnings
import gc
import time

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except:
    HAS_TORCH = False
    print("PyTorch未安装")

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')
np.random.seed(42)
if HAS_TORCH:
    torch.manual_seed(42)

# ============================================================
# 配置
# ============================================================
BASE_PATH = '/kaggle/input/competitions/china-a-share-market-microstructure-prediction/'
OUTPUT_PATH = '/kaggle/working/'

print(f"数据路径: {BASE_PATH}")
print(f"输出路径: {OUTPUT_PATH}")

ID_COLS = ['stockid', 'dateid', 'timeid', 'exchangeid']
TARGET_COLS = ['LabelA', 'LabelB', 'LabelC']
FEATURE_COLS = [f'f{i}' for i in range(384)]

# V23核心优化配置
N_CHUNKS = None
BATCH_SIZE = 500000
USE_GPU = True
TRAIN_RATIO = 0.85
OPTUNA_TRIALS = 150  # 150次!
N_BAGS = 15  # 15次!
N_FOLDS = 5
PSEUDO_ROUNDS = 3
N_PCA = 50  # PCA组件数
N_ICA = 30  # ICA组件数

# ============================================================
# 1. 数据加载
# ============================================================
print("\n[1] 数据加载...")
start_time = time.time()

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
print(f"加载时间: {time.time() - start_time:.1f}秒")

# ============================================================
# 2. PCA特征 (新增!)
# ============================================================
print("\n[2] PCA特征工程...")
start_time = time.time()

pca_cols = [f'f{i}' for i in range(100)]
pca_cols = [c for c in pca_cols if c in train_df.columns]

pca = PCA(n_components=min(N_PCA, len(pca_cols)), random_state=42)
X_pca_train = pca.fit_transform(train_df[pca_cols].fillna(0).values)
X_pca_test = pca.transform(test_df[pca_cols].fillna(0).values)

for i in range(X_pca_train.shape[1]):
    train_df[f'pca_{i}'] = X_pca_train[:, i]
    test_df[f'pca_{i}'] = X_pca_test[:, i]

pca_features = [f'pca_{i}' for i in range(X_pca_train.shape[1])]
print(f"  PCA特征数: {len(pca_features)}")
print(f"  PCA解释方差: {pca.explained_variance_ratio_.sum():.4f}")
print(f"  时间: {time.time() - start_time:.1f}秒")

# ============================================================
# 3. ICA特征 (新增!)
# ============================================================
print("\n[3] ICA特征工程...")
start_time = time.time()

ica_cols = [f'f{i}' for i in range(80)]
ica_cols = [c for c in ica_cols if c in train_df.columns]

ica = FastICA(n_components=min(N_ICA, len(ica_cols)), random_state=42, max_iter=200)
X_ica_train = ica.fit_transform(train_df[ica_cols].fillna(0).values)
X_ica_test = ica.transform(test_df[ica_cols].fillna(0).values)

for i in range(X_ica_train.shape[1]):
    train_df[f'ica_{i}'] = X_ica_train[:, i]
    test_df[f'ica_{i}'] = X_ica_test[:, i]

ica_features = [f'ica_{i}' for i in range(X_ica_train.shape[1])]
print(f"  ICA特征数: {len(ica_features)}")
print(f"  时间: {time.time() - start_time:.1f}秒")

# ============================================================
# 4. 市场状态分类 (新增!)
# ============================================================
print("\n[4] 市场状态分类...")
start_time = time.time()

if 'f298' in train_df.columns:
    train_df['price_change'] = train_df.groupby('stockid')['f298'].diff()
    test_df['price_change'] = test_df.groupby('stockid')['f298'].diff()
    
    daily_stats = train_df.groupby('dateid').agg({
        'price_change': ['mean', 'std', 'sum']
    }).reset_index()
    daily_stats.columns = ['dateid', 'market_return', 'market_vol', 'market_trend']
    
    train_df = train_df.merge(daily_stats, on='dateid', how='left')
    test_df = test_df.merge(daily_stats, on='dateid', how='left')
    
    vol_q = daily_stats['market_vol'].quantile([0.33, 0.66]).values
    train_df['vol_regime'] = pd.cut(train_df['market_vol'], bins=[-np.inf, vol_q[0], vol_q[1], np.inf], 
                                     labels=['low_vol', 'mid_vol', 'high_vol'])
    test_df['vol_regime'] = pd.cut(test_df['market_vol'], bins=[-np.inf, vol_q[0], vol_q[1], np.inf], 
                                    labels=['low_vol', 'mid_vol', 'high_vol'])
    
    train_df['vol_regime'] = train_df['vol_regime'].astype(str)
    test_df['vol_regime'] = test_df['vol_regime'].astype(str)
    
    vol_dummies_train = pd.get_dummies(train_df['vol_regime'], prefix='regime')
    vol_dummies_test = pd.get_dummies(test_df['vol_regime'], prefix='regime')
    
    for col in vol_dummies_train.columns:
        train_df[col] = vol_dummies_train[col]
        test_df[col] = vol_dummies_test[col]
    
    regime_features = list(vol_dummies_train.columns) + ['market_return', 'market_vol', 'market_trend']
    print(f"  市场状态特征数: {len(regime_features)}")

print(f"  时间: {time.time() - start_time:.1f}秒")

# ============================================================
# 5. 傅里叶特征
# ============================================================
print("\n[5] 傅里叶特征...")

n_harmonics = 5
train_df['timeid_norm'] = train_df['timeid'].values / 239.0
test_df['timeid_norm'] = test_df['timeid'].values / 239.0

for k in range(1, n_harmonics + 1):
    train_df[f'fourier_sin{k}'] = np.sin(2 * np.pi * k * train_df['timeid_norm'])
    train_df[f'fourier_cos{k}'] = np.cos(2 * np.pi * k * train_df['timeid_norm'])
    test_df[f'fourier_sin{k}'] = np.sin(2 * np.pi * k * test_df['timeid_norm'])
    test_df[f'fourier_cos{k}'] = np.cos(2 * np.pi * k * test_df['timeid_norm'])

fourier_features = [f'fourier_sin{k}' for k in range(1, n_harmonics + 1)] + \
                   [f'fourier_cos{k}' for k in range(1, n_harmonics + 1)]

# ============================================================
# 6. 跨股票特征
# ============================================================
print("\n[6] 跨股票特征...")

CROSS_FEATURES = [f'f{i}' for i in range(50)]
CROSS_FEATURES = [f for f in CROSS_FEATURES if f in train_df.columns]

for col in CROSS_FEATURES[:30]:
    train_df[f'{col}_cross_mean'] = train_df.groupby(['dateid', 'timeid'])[col].transform('mean')
    test_df[f'{col}_cross_mean'] = test_df.groupby(['dateid', 'timeid'])[col].transform('mean')
    
    train_df[f'{col}_zscore'] = (train_df[col] - train_df[f'{col}_cross_mean']) / (train_df[f'{col}_cross_mean'].std() + 1e-8)
    test_df[f'{col}_zscore'] = (test_df[col] - test_df[f'{col}_cross_mean']) / (test_df[f'{col}_cross_mean'].std() + 1e-8)

if 'f298' in train_df.columns:
    train_df['market_avg_change'] = train_df.groupby(['dateid', 'timeid'])['price_change'].transform('mean')
    test_df['market_avg_change'] = test_df.groupby(['dateid', 'timeid'])['price_change'].transform('mean')
    train_df['market_volatility'] = train_df.groupby(['dateid', 'timeid'])['price_change'].transform('std')
    test_df['market_volatility'] = test_df.groupby(['dateid', 'timeid'])['price_change'].transform('std')

cross_features = [c for c in train_df.columns if '_cross_' in c or c in ['market_avg_change', 'market_volatility']]
gc.collect()

# ============================================================
# 7. 时间序列特征
# ============================================================
print("\n[7] 时间序列特征...")

key_cols = ['f0', 'f1', 'f2', 'f3']
key_cols = [c for c in key_cols if c in train_df.columns]

new_features = []

for col in key_cols:
    for window in [5, 10]:
        train_df[f'{col}_ma{window}'] = train_df.groupby('stockid')[col].transform(
            lambda x: x.rolling(window, min_periods=1).mean())
        test_df[f'{col}_ma{window}'] = test_df.groupby('stockid')[col].transform(
            lambda x: x.rolling(window, min_periods=1).mean())
        new_features.append(f'{col}_ma{window}')
        
        train_df[f'{col}_std{window}'] = train_df.groupby('stockid')[col].transform(
            lambda x: x.rolling(window, min_periods=1).std())
        test_df[f'{col}_std{window}'] = test_df.groupby('stockid')[col].transform(
            lambda x: x.rolling(window, min_periods=1).std())
        new_features.append(f'{col}_std{window}')
        
        train_df[f'{col}_ewm{window}'] = train_df.groupby('stockid')[col].transform(
            lambda x: x.ewm(span=window, min_periods=1).mean())
        test_df[f'{col}_ewm{window}'] = test_df.groupby('stockid')[col].transform(
            lambda x: x.ewm(span=window, min_periods=1).mean())
        new_features.append(f'{col}_ewm{window}')

for col in key_cols:
    for lag in [1, 3]:
        train_df[f'{col}_lag{lag}'] = train_df.groupby('stockid')[col].shift(lag)
        test_df[f'{col}_lag{lag}'] = test_df.groupby('stockid')[col].shift(lag)
        new_features.append(f'{col}_lag{lag}')

    train_df[f'{col}_diff1'] = train_df.groupby('stockid')[col].diff(1)
    test_df[f'{col}_diff1'] = test_df.groupby('stockid')[col].diff(1)
    new_features.append(f'{col}_diff1')

if 'f2' in key_cols and 'f3' in key_cols:
    train_df['obi'] = (train_df['f2'] - train_df['f3']) / (train_df['f2'] + train_df['f3'] + 1e-8)
    test_df['obi'] = (test_df['f2'] - test_df['f3']) / (test_df['f2'] + test_df['f3'] + 1e-8)
    new_features.append('obi')

gc.collect()

# ============================================================
# 8. 特征交互
# ============================================================
print("\n[8] 特征交互...")

TOP_6 = ['f298', 'f105', 'f128', 'f28', 'f46', 'f326']
TOP_6 = [f for f in TOP_6 if f in train_df.columns]

interaction_features = []
for i, f1 in enumerate(TOP_6):
    for f2 in TOP_6[i+1:]:
        train_df[f'{f1}_x_{f2}'] = train_df[f1] * train_df[f2]
        test_df[f'{f1}_x_{f2}'] = test_df[f1] * test_df[f2]
        interaction_features.append(f'{f1}_x_{f2}')

# ============================================================
# 9. 增强目标编码
# ============================================================
print("\n[9] 增强目标编码...")

train_df = train_df.sort_values(['stockid', 'dateid', 'timeid']).reset_index(drop=True)
unique_dates = sorted(train_df['dateid'].unique())
n_dates = len(unique_dates)

n_train_dates = int(n_dates * TRAIN_RATIO)
train_dates = unique_dates[:n_train_dates]
val_dates = unique_dates[n_train_dates:]

train_for_encoding = train_df[train_df['dateid'].isin(train_dates)]

stock_means_a = train_for_encoding.groupby('stockid')['LabelA'].mean()
stock_means_b = train_for_encoding.groupby('stockid')['LabelB'].mean()
stock_means_c = train_for_encoding.groupby('stockid')['LabelC'].mean()
global_mean = train_for_encoding['LabelA'].mean()

train_df['stock_target_enc_a'] = train_df['stockid'].map(stock_means_a).fillna(global_mean)
train_df['stock_target_enc_b'] = train_df['stockid'].map(stock_means_b).fillna(global_mean)
train_df['stock_target_enc_c'] = train_df['stockid'].map(stock_means_c).fillna(global_mean)
test_df['stock_target_enc_a'] = test_df['stockid'].map(stock_means_a).fillna(global_mean)
test_df['stock_target_enc_b'] = test_df['stockid'].map(stock_means_b).fillna(global_mean)
test_df['stock_target_enc_c'] = test_df['stockid'].map(stock_means_c).fillna(global_mean)
new_features.extend(['stock_target_enc_a', 'stock_target_enc_b', 'stock_target_enc_c'])

if 'exchangeid' in train_df.columns:
    exchange_means = train_for_encoding.groupby('exchangeid')['LabelA'].mean()
    train_df['exchange_target_enc'] = train_df['exchangeid'].map(exchange_means).fillna(global_mean)
    test_df['exchange_target_enc'] = test_df['exchangeid'].map(exchange_means).fillna(global_mean)
    new_features.append('exchange_target_enc')

time_means = train_for_encoding.groupby('timeid')['LabelA'].mean()
train_df['time_target_enc'] = train_df['timeid'].map(time_means).fillna(global_mean)
test_df['time_target_enc'] = test_df['timeid'].map(time_means).fillna(global_mean)
new_features.append('time_target_enc')

# ============================================================
# 10. 时间序列划分
# ============================================================
print("\n[10] 时间序列划分...")

train_data = train_df[train_df['dateid'].isin(train_dates)].copy()
val_data = train_df[train_df['dateid'].isin(val_dates)].copy()

print(f"训练: {len(train_data):,}, 验证: {len(val_data):,}")

del train_df
gc.collect()

# ============================================================
# 11. 样本权重
# ============================================================
print("\n[11] 样本权重计算...")

if 'price_change' in train_data.columns:
    train_data['market_vol'] = train_data.groupby(['dateid', 'timeid'])['price_change'].transform('std')
    val_data['market_vol'] = val_data.groupby(['dateid', 'timeid'])['price_change'].transform('std')
    
    vol_quantile = train_data['market_vol'].quantile(0.75)
    train_data['vol_weight'] = np.where(train_data['market_vol'] > vol_quantile, 1.5, 1.0)
    val_data['vol_weight'] = np.where(val_data['market_vol'] > vol_quantile, 1.5, 1.0)

train_data['time_weight'] = np.where(train_data['timeid'] < 229, 1.0, 0.1)
val_data['time_weight'] = np.where(val_data['timeid'] < 229, 1.0, 0.1)

train_data['sample_weight'] = train_data['vol_weight'] * train_data['time_weight']
val_data['sample_weight'] = np.ones(len(val_data))

sample_weights = train_data['sample_weight'].values

# ============================================================
# 12. 准备特征
# ============================================================
print("\n[12] 准备特征...")

exclude_cols = ID_COLS + TARGET_COLS + ['sample_weight', 'vol_weight', 'time_weight', 
                                        'market_vol', 'price_change', 'timeid_norm', 'vol_regime']
all_features = (FEATURE_COLS + cross_features + new_features + fourier_features + 
                interaction_features + pca_features + ica_features + regime_features)
all_features = [f for f in all_features if f in train_data.columns and f in test_df.columns]
all_features = list(set(all_features))

print(f"总特征数: {len(all_features)}")

train_data[all_features] = train_data[all_features].fillna(0)
val_data[all_features] = val_data[all_features].fillna(0)

X_train = train_data[all_features].values.astype('float32')
y_train_a = train_data['LabelA'].values.astype('float32')
y_train_b = train_data['LabelB'].values.astype('float32')
y_train_c = train_data['LabelC'].values.astype('float32')
X_val = val_data[all_features].values.astype('float32')
y_val = val_data['LabelA'].values.astype('float32')

print(f"X_train: {X_train.shape}")

# ============================================================
# 13. Optuna 150次试验
# ============================================================
print("\n[13] Optuna贝叶斯优化 (150次试验)...")
optuna_start = time.time()

def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 800, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.03),
        'max_depth': trial.suggest_int('max_depth', 8, 14),
        'num_leaves': trial.suggest_int('num_leaves', 63, 255),
        'min_child_samples': trial.suggest_int('min_child_samples', 30, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 1.5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 1.5),
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train_a, sample_weight=sample_weights)
    return r2_score(y_val, model.predict(X_val))

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=OPTUNA_TRIALS)

best_params = study.best_params
print(f"最佳参数: {best_params}")
print(f"最佳R²: {study.best_value:.6f}")
print(f"Optuna优化时间: {time.time() - optuna_start:.1f}秒")

# ============================================================
# 14. 多目标学习
# ============================================================
print("\n[14] 多目标学习...")

lgb_base_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1,
    **best_params
}

lgb_model_a = lgb.LGBMRegressor(**lgb_base_params)
lgb_model_a.fit(X_train, y_train_a, sample_weight=sample_weights)
lgb_val_pred_a = lgb_model_a.predict(X_val)
lgb_r2_a = r2_score(y_val, lgb_val_pred_a)
print(f"LabelA R²: {lgb_r2_a:.6f}")

lgb_model_b = lgb.LGBMRegressor(**{**lgb_base_params, 'n_estimators': 500})
lgb_model_b.fit(X_train, y_train_b, sample_weight=sample_weights)
lgb_val_pred_b = lgb_model_b.predict(X_val)

lgb_model_c = lgb.LGBMRegressor(**{**lgb_base_params, 'n_estimators': 500})
lgb_model_c.fit(X_train, y_train_c, sample_weight=sample_weights)
lgb_val_pred_c = lgb_model_c.predict(X_val)

multi_task_pred = 0.7 * lgb_val_pred_a + 0.15 * lgb_val_pred_b + 0.15 * lgb_val_pred_c
multi_task_r2 = r2_score(y_val, multi_task_pred)
print(f"多目标融合 R²: {multi_task_r2:.6f}")

# ============================================================
# 15. LightGBM + 5折CV + 15次Bagging
# ============================================================
print("\n[15] LightGBM + 5折CV + 15次Bagging...")
start_time = time.time()

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
lgb_cv_preds = np.zeros(len(y_val))
lgb_cv_models = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_vl = X_train[train_idx], X_train[val_idx]
    y_tr, y_vl = y_train_a[train_idx], y_train_a[val_idx]
    sw_tr = sample_weights[train_idx]
    
    model = lgb.LGBMRegressor(**lgb_base_params)
    model.fit(X_tr, y_tr, sample_weight=sw_tr)
    
    lgb_cv_preds[val_idx] = model.predict(X_vl)
    lgb_cv_models.append(model)
    print(f"  Fold {fold+1}/{N_FOLDS} 完成")

lgb_cv_r2 = r2_score(y_val, lgb_cv_preds)
print(f"LightGBM(5-Fold CV) R²: {lgb_cv_r2:.6f}")

# 15次Bagging
lgb_bag_preds = []
for i in range(N_BAGS):
    lgb_params_bag = {**lgb_base_params, 'random_state': 42 + i}
    model = lgb.LGBMRegressor(**lgb_params_bag)
    model.fit(
        X_train, y_train_a,
        eval_set=[(X_val, y_val)],
        sample_weight=sample_weights,
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    lgb_bag_preds.append(model.predict(X_val))
    print(f"  Bag {i+1}/{N_BAGS} 完成")

lgb_val_pred = np.mean(lgb_bag_preds, axis=0)
lgb_r2 = r2_score(y_val, lgb_val_pred)
print(f"LightGBM(15-Bagging) R²: {lgb_r2:.6f}")
print(f"训练时间: {time.time() - start_time:.1f}秒")

# 特征重要性
importance = pd.DataFrame({
    'feature': all_features,
    'importance': lgb_cv_models[0].feature_importances_
}).sort_values('importance', ascending=False)
print("\nTop 10 特征:")
print(importance.head(10))

# ============================================================
# 16. 特征选择
# ============================================================
print("\n[16] 特征选择...")

selected_features = importance[importance['importance'] > 0]['feature'].tolist()
print(f"选择后特征数: {len(selected_features)}")

X_train_selected = train_data[selected_features].fillna(0).values.astype('float32')
X_val_selected = val_data[selected_features].fillna(0).values.astype('float32')

# ============================================================
# 17. XGBoost
# ============================================================
print("\n[17] XGBoost...")
start_time = time.time()

xgb_base_params = {
    'objective': 'reg:squarederror',
    'random_state': 42,
    'tree_method': 'gpu_hist' if USE_GPU else 'hist',
    'gpu_id': 0,
    **best_params
}

xgb_bag_preds = []
for i in range(N_BAGS):
    xgb_params_bag = {**xgb_base_params, 'random_state': 42 + i}
    model = xgb.XGBRegressor(**xgb_params_bag)
    model.fit(
        X_train_selected, y_train_a,
        eval_set=[(X_val_selected, y_val)],
        sample_weight=sample_weights,
        verbose=0,
        early_stopping_rounds=50
    )
    xgb_bag_preds.append(model.predict(X_val_selected))
    print(f"  Bag {i+1}/{N_BAGS} 完成")

xgb_val_pred = np.mean(xgb_bag_preds, axis=0)
xgb_r2 = r2_score(y_val, xgb_val_pred)
print(f"XGBoost R²: {xgb_r2:.6f}")
print(f"训练时间: {time.time() - start_time:.1f}秒")

# ============================================================
# 18. CatBoost
# ============================================================
print("\n[18] CatBoost...")
start_time = time.time()

try:
    cat_model = CatBoostRegressor(
        iterations=best_params.get('n_estimators', 1000),
        learning_rate=best_params.get('learning_rate', 0.02),
        depth=best_params.get('max_depth', 10),
        l2_leaf_reg=best_params.get('reg_lambda', 0.5),
        random_seed=42,
        task_type='GPU' if USE_GPU else 'CPU',
        verbose=100
    )
    
    cat_model.fit(
        X_train_selected, y_train_a,
        eval_set=(X_val_selected, y_val),
        sample_weight=sample_weights,
        early_stopping_rounds=50
    )
    
    cat_val_pred = cat_model.predict(X_val_selected)
    cat_r2 = r2_score(y_val, cat_val_pred)
    print(f"CatBoost R²: {cat_r2:.6f}")
    use_cat = True
except Exception as e:
    print(f"CatBoost失败: {e}")
    cat_val_pred = lgb_val_pred
    cat_r2 = lgb_r2
    use_cat = False

print(f"训练时间: {time.time() - start_time:.1f}秒")

# ============================================================
# 19. MLP
# ============================================================
print("\n[19] MLP...")
start_time = time.time()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_val_scaled = scaler.transform(X_val_selected)

mlp_model = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    alpha=0.001,
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42,
    verbose=False
)
mlp_model.fit(X_train_scaled, y_train_a)
mlp_val_pred = mlp_model.predict(X_val_scaled)
mlp_r2 = r2_score(y_val, mlp_val_pred)
print(f"MLP R²: {mlp_r2:.6f}")
print(f"训练时间: {time.time() - start_time:.1f}秒")

# ============================================================
# 20. 线性模型
# ============================================================
print("\n[20] 线性模型...")

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train_a, sample_weight=sample_weights)
ridge_val_pred = ridge_model.predict(X_val_scaled)
ridge_r2 = r2_score(y_val, ridge_val_pred)
print(f"Ridge R²: {ridge_r2:.6f}")

# ============================================================
# 21. 3层Stacking
# ============================================================
print("\n[21] 3层Stacking...")

level1_pred = np.column_stack([
    lgb_val_pred, xgb_val_pred, cat_val_pred, mlp_val_pred, ridge_val_pred, multi_task_pred
])

lgb_l2_1 = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, max_depth=6, verbose=-1, random_state=42)
lgb_l2_1.fit(level1_pred, y_train_a, sample_weight=sample_weights)
l2_1_pred = lgb_l2_1.predict(level1_pred)

lgb_l2_2 = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, max_depth=4, verbose=-1, random_state=43)
lgb_l2_2.fit(level1_pred, y_train_a, sample_weight=sample_weights)
l2_2_pred = lgb_l2_2.predict(level1_pred)

ridge_l2 = Ridge(alpha=0.5)
ridge_l2.fit(level1_pred, y_train_a, sample_weight=sample_weights)
l2_ridge_pred = ridge_l2.predict(level1_pred)

level2_pred = np.column_stack([l2_1_pred, l2_2_pred, l2_ridge_pred])

lgb_l3 = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, verbose=-1, random_state=44)
lgb_l3.fit(level2_pred, y_train_a, sample_weight=sample_weights)
stack_l3_pred = lgb_l3.predict(level2_pred)
stack_l3_r2 = r2_score(y_val, stack_l3_pred)
print(f"3层Stacking R²: {stack_l3_r2:.6f}")

# ============================================================
# 22. 迭代伪标签
# ============================================================
print("\n[22] 迭代伪标签 (3轮)...")
start_time = time.time()

X_test = test_df[selected_features].fillna(0).values.astype('float32')
X_test_scaled = scaler.transform(X_test)

lgb_test_pred = np.mean([m.predict(X_test) for m in lgb_cv_models], axis=0)
xgb_test_pred = np.mean([m.predict(X_test) for m in xgb_bag_preds], axis=0)
ridge_test_pred = ridge_model.predict(X_test_scaled)
mlp_test_pred = mlp_model.predict(X_test_scaled)

if use_cat:
    cat_test_pred = cat_model.predict(X_test)
    test_pred = (0.25 * lgb_test_pred + 0.20 * xgb_test_pred + 0.15 * cat_test_pred + 
                0.15 * mlp_test_pred + 0.10 * ridge_test_pred + 0.15 * multi_task_pred)
else:
    test_pred = (0.30 * lgb_test_pred + 0.25 * xgb_test_pred + 
                0.15 * mlp_test_pred + 0.10 * ridge_test_pred + 0.20 * multi_task_pred)

for round_num in range(PSEUDO_ROUNDS):
    print(f"  伪标签轮次 {round_num + 1}/{PSEUDO_ROUNDS}")
    
    high_conf_mask = np.abs(test_pred) > np.percentile(np.abs(test_pred), 70)
    n_pseudo = high_conf_mask.sum()
    print(f"    高置信度样本: {n_pseudo:,}")
    
    if n_pseudo < 1000:
        break
    
    X_pseudo = X_test[high_conf_mask]
    y_pseudo = test_pred[high_conf_mask]
    
    X_augmented = np.vstack([X_train_selected, X_pseudo])
    y_augmented = np.concatenate([y_train_a, y_pseudo])
    sw_augmented = np.concatenate([sample_weights, np.ones(n_pseudo) * 0.5])
    
    lgb_pseudo = lgb.LGBMRegressor(**{**lgb_base_params, 'n_estimators': 800})
    lgb_pseudo.fit(X_augmented, y_augmented, sample_weight=sw_augmented)
    
    test_pred = lgb_pseudo.predict(X_test)
    
    lgb_val_pseudo = lgb_pseudo.predict(X_val_selected)
    pseudo_r2 = r2_score(y_val, lgb_val_pseudo)
    print(f"    伪标签后验证R²: {pseudo_r2:.6f}")

print(f"训练时间: {time.time() - start_time:.1f}秒")

# ============================================================
# 23. 最终预测
# ============================================================
print("\n[23] 最终预测...")

lgb_test_pred = np.mean([m.predict(X_test) for m in lgb_cv_models], axis=0)
xgb_test_pred = np.mean([m.predict(X_test) for m in xgb_bag_preds], axis=0)
ridge_test_pred = ridge_model.predict(X_test_scaled)
mlp_test_pred = mlp_model.predict(X_test_scaled)

lgb_model_a_test = lgb.LGBMRegressor(**lgb_base_params)
lgb_model_a_test.fit(X_train_selected, y_train_a, sample_weight=sample_weights)
multi_task_test = lgb_model_a_test.predict(X_test)

if use_cat:
    cat_test_pred = cat_model.predict(X_test)
    final_pred = (0.20 * lgb_test_pred + 0.18 * xgb_test_pred + 0.15 * cat_test_pred + 
                 0.12 * mlp_test_pred + 0.10 * ridge_test_pred + 0.15 * multi_task_test +
                 0.10 * test_pred)
else:
    final_pred = (0.25 * lgb_test_pred + 0.20 * xgb_test_pred + 
                 0.12 * mlp_test_pred + 0.10 * ridge_test_pred + 0.18 * multi_task_test +
                 0.15 * test_pred)

print(f"测试集预测完成: {len(final_pred):,}")

del X_test, test_df
gc.collect()

# ============================================================
# 24. 后处理校准
# ============================================================
print("\n[24] 后处理校准...")

weighted_pred = (0.20 * lgb_val_pred + 0.18 * xgb_val_pred + 0.14 * cat_val_pred + 
                0.12 * mlp_val_pred + 0.10 * ridge_val_pred + 0.14 * multi_task_pred +
                0.12 * stack_l3_pred)
weighted_r2 = r2_score(y_val, weighted_pred)
print(f"加权平均 R²: {weighted_r2:.6f}")

iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(weighted_pred, y_val)
calibrated_pred = iso.predict(weighted_pred)
calibrated_r2 = r2_score(y_val, calibrated_pred)
print(f"等渗校准 R²: {calibrated_r2:.6f}")

# 选择最佳
all_r2 = {
    'LightGBM_5FoldCV': lgb_cv_r2,
    'LightGBM_15Bag': lgb_r2,
    'XGBoost': xgb_r2,
    'CatBoost': cat_r2,
    'MLP': mlp_r2,
    'Ridge': ridge_r2,
    'Multi_Task': multi_task_r2,
    'Stack_3Layer': stack_l3_r2,
    'Weighted': weighted_r2,
    'Calibrated': calibrated_r2,
}

best_method = max(all_r2, key=all_r2.get)
best_r2 = all_r2[best_method]
print(f"\n最佳方法: {best_method}, R²: {best_r2:.6f}")

# ============================================================
# 25. 生成提交文件
# ============================================================
print("\n[25] 生成提交文件...")

test_df = pd.read_parquet(BASE_PATH + 'test.parquet', columns=['stockid', 'dateid', 'timeid'])

submission = pd.DataFrame({
    'Uid': test_df['stockid'].astype(str) + '|' + 
           test_df['dateid'].astype(str) + '|' + 
           test_df['timeid'].astype(str),
    'prediction': final_pred
})

submission.to_csv(OUTPUT_PATH + 'submission.csv', index=False)
print(f"提交文件保存: {OUTPUT_PATH}submission.csv")
print(f"样本数: {len(submission):,}")
print(submission.head())

print("\n" + "="*70)
print("✅ V23 完成！")
print("="*70)

total_time = time.time() - start_time
print(f"总耗时: {total_time/60:.1f} 分钟")

print(f"\n=== V23 核心优化 ===")
print(f"  1. ⭐⭐⭐⭐⭐ PCA特征 ({N_PCA}组件)")
print(f"  2. ⭐⭐⭐⭐⭐ ICA特征 ({N_ICA}组件)")
print(f"  3. ⭐⭐⭐⭐⭐ 市场状态分类")
print(f"  4. ⭐⭐⭐⭐⭐ Optuna {OPTUNA_TRIALS}次")
print(f"  5. ⭐⭐⭐⭐⭐ {N_BAGS}次Bagging")

print(f"\n=== 模型性能 ===")
for name, r2 in all_r2.items():
    print(f"  {name}: R² = {r2:.6f}")
print(f"  最佳: {best_method}, R² = {best_r2:.6f}")

print(f"\n🏆 预期R²: 0.012 → {best_r2:.4f}")
print("🎯 祝竞赛取得金牌!")
