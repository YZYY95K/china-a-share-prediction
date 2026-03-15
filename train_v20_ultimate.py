"""
China A-Share Market Microstructure Prediction
🏆 V20 终极版 - 极限优化天花板

包含所有优化:
1. ⭐⭐⭐⭐⭐ Optuna 50次试验
2. ⭐⭐⭐⭐⭐ 10次Bagging
3. ⭐⭐⭐⭐⭐ 全部训练数据
4. ⭐⭐⭐⭐ PyTorch GPU神经网络
5. ⭐⭐⭐⭐ 3层Stacking
6. ⭐⭐⭐ 协变量偏移适应
"""

print("="*70)
print("🏆 V20 终极版 - 极限优化天花板")
print("="*70)

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit
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
    print("PyTorch未安装，使用sklearn MLP")

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

# V20核心优化配置
N_CHUNKS = None  # 全部数据!
BATCH_SIZE = 500000
USE_GPU = True
TRAIN_RATIO = 0.85
OPTUNA_TRIALS = 50  # 增加到50!
N_BAGS = 10  # 增加到10!

# ============================================================
# 1. 数据加载 (全部数据!)
# ============================================================
print("\n[1] 数据加载 (全部数据!)...")
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
# 2. 傅里叶特征
# ============================================================
print("\n[2] 傅里叶特征...")

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
# 3. 跨股票特征 (精简版，加速)
# ============================================================
print("\n[3] 跨股票特征 (精简版)...")

CROSS_FEATURES = [f'f{i}' for i in range(50)]  # 精简到50个
CROSS_FEATURES = [f for f in CROSS_FEATURES if f in train_df.columns]

for col in CROSS_FEATURES[:30]:
    train_df[f'{col}_cross_mean'] = train_df.groupby(['dateid', 'timeid'])[col].transform('mean')
    test_df[f'{col}_cross_mean'] = test_df.groupby(['dateid', 'timeid'])[col].transform('mean')
    
    train_df[f'{col}_zscore'] = (train_df[col] - train_df[f'{col}_cross_mean']) / (train_df[f'{col}_cross_mean'].std() + 1e-8)
    test_df[f'{col}_zscore'] = (test_df[col] - test_df[f'{col}_cross_mean']) / (test_df[f'{col}_cross_mean'].std() + 1e-8)

if 'f298' in train_df.columns:
    train_df['price_change'] = train_df.groupby('stockid')['f298'].diff()
    test_df['price_change'] = test_df.groupby('stockid')['f298'].diff()
    
    train_df['market_avg_change'] = train_df.groupby(['dateid', 'timeid'])['price_change'].transform('mean')
    test_df['market_avg_change'] = test_df.groupby(['dateid', 'timeid'])['price_change'].transform('mean')
    train_df['market_volatility'] = train_df.groupby(['dateid', 'timeid'])['price_change'].transform('std')
    test_df['market_volatility'] = test_df.groupby(['dateid', 'timeid'])['price_change'].transform('std')

cross_features = [c for c in train_df.columns if '_cross_' in c or c in ['market_avg_change', 'market_volatility']]
gc.collect()

# ============================================================
# 4. 时间序列特征
# ============================================================
print("\n[4] 时间序列特征...")

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
# 5. 特征交互
# ============================================================
print("\n[5] 特征交互...")

TOP_6 = ['f298', 'f105', 'f128', 'f28', 'f46', 'f326']
TOP_6 = [f for f in TOP_6 if f in train_df.columns]

interaction_features = []
for i, f1 in enumerate(TOP_6):
    for f2 in TOP_6[i+1:]:
        train_df[f'{f1}_x_{f2}'] = train_df[f1] * train_df[f2]
        test_df[f'{f1}_x_{f2}'] = test_df[f1] * test_df[f2]
        interaction_features.append(f'{f1}_x_{f2}')

# ============================================================
# 6. 增强目标编码
# ============================================================
print("\n[6] 增强目标编码...")

train_df = train_df.sort_values(['stockid', 'dateid', 'timeid']).reset_index(drop=True)
unique_dates = sorted(train_df['dateid'].unique())
n_dates = len(unique_dates)

n_train_dates = int(n_dates * TRAIN_RATIO)
train_dates = unique_dates[:n_train_dates]
val_dates = unique_dates[n_train_dates:]

train_for_encoding = train_df[train_df['dateid'].isin(train_dates)]

stock_means = train_for_encoding.groupby('stockid')['LabelA'].mean()
global_mean = train_for_encoding['LabelA'].mean()

train_df['stock_target_enc'] = train_df['stockid'].map(stock_means).fillna(global_mean)
test_df['stock_target_enc'] = test_df['stockid'].map(stock_means).fillna(global_mean)
new_features.append('stock_target_enc')

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
# 7. 时间序列划分
# ============================================================
print("\n[7] 时间序列划分...")

train_data = train_df[train_df['dateid'].isin(train_dates)].copy()
val_data = train_df[train_df['dateid'].isin(val_dates)].copy()

print(f"训练: {len(train_data):,}, 验证: {len(val_data):,}")

del train_df
gc.collect()

# ============================================================
# 8. 样本权重
# ============================================================
print("\n[8] 样本权重计算...")

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
# 9. 准备特征
# ============================================================
print("\n[9] 准备特征...")

exclude_cols = ID_COLS + TARGET_COLS + ['sample_weight', 'vol_weight', 'time_weight', 
                                        'market_vol', 'price_change', 'timeid_norm']
all_features = FEATURE_COLS + cross_features + new_features + fourier_features + interaction_features
all_features = [f for f in all_features if f in train_data.columns and f in test_df.columns]
all_features = list(set(all_features))

print(f"总特征数: {len(all_features)}")

train_data[all_features] = train_data[all_features].fillna(0)
val_data[all_features] = val_data[all_features].fillna(0)

X_train = train_data[all_features].values.astype('float32')
y_train = train_data['LabelA'].values.astype('float32')
X_val = val_data[all_features].values.astype('float32')
y_val = val_data['LabelA'].values.astype('float32')

print(f"X_train: {X_train.shape}")

# ============================================================
# 10. Optuna 50次试验
# ============================================================
print("\n[10] Optuna贝叶斯优化 (50次试验)...")
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
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return r2_score(y_val, model.predict(X_val))

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=OPTUNA_TRIALS)

best_params = study.best_params
print(f"最佳参数: {best_params}")
print(f"最佳R²: {study.best_value:.6f}")
print(f"Optuna优化时间: {time.time() - optuna_start:.1f}秒")

# ============================================================
# 11. LightGBM + 10次Bagging
# ============================================================
print("\n[11] LightGBM + 10次Bagging...")
start_time = time.time()

lgb_base_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1,
    **best_params
}

lgb_bag_preds = []
lgb_bag_models = []

for i in range(N_BAGS):
    lgb_params_bag = {**lgb_base_params, 'random_state': 42 + i}
    model = lgb.LGBMRegressor(**lgb_params_bag)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        sample_weight=sample_weights,
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    lgb_bag_preds.append(model.predict(X_val))
    lgb_bag_models.append(model)
    print(f"  Bag {i+1}/{N_BAGS} 完成")

lgb_val_pred = np.mean(lgb_bag_preds, axis=0)
lgb_r2 = r2_score(y_val, lgb_val_pred)
print(f"LightGBM(10-Bagging) R²: {lgb_r2:.6f}")
print(f"训练时间: {time.time() - start_time:.1f}秒")

# 特征重要性
importance = pd.DataFrame({
    'feature': all_features,
    'importance': lgb_bag_models[0].feature_importances_
}).sort_values('importance', ascending=False)
print("\nTop 10 特征:")
print(importance.head(10))

# ============================================================
# 12. 特征选择
# ============================================================
print("\n[12] 特征选择...")

selected_features = importance[importance['importance'] > 0]['feature'].tolist()
print(f"选择后特征数: {len(selected_features)}")

X_train_selected = train_data[selected_features].fillna(0).values.astype('float32')
X_val_selected = val_data[selected_features].fillna(0).values.astype('float32')

# ============================================================
# 13. XGBoost + 10次Bagging
# ============================================================
print("\n[13] XGBoost + 10次Bagging...")
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
        X_train_selected, y_train,
        eval_set=[(X_val_selected, y_val)],
        sample_weight=sample_weights,
        verbose=0,
        early_stopping_rounds=50
    )
    xgb_bag_preds.append(model.predict(X_val_selected))
    print(f"  Bag {i+1}/{N_BAGS} 完成")

xgb_val_pred = np.mean(xgb_bag_preds, axis=0)
xgb_r2 = r2_score(y_val, xgb_val_pred)
print(f"XGBoost(10-Bagging) R²: {xgb_r2:.6f}")
print(f"训练时间: {time.time() - start_time:.1f}秒")

# ============================================================
# 14. CatBoost
# ============================================================
print("\n[14] CatBoost...")
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
        X_train_selected, y_train,
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
# 15. PyTorch GPU神经网络 (新增!)
# ============================================================
print("\n[15] PyTorch GPU神经网络...")
start_time = time.time()

if HAS_TORCH:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_val_scaled = scaler.transform(X_val_selected)
    
    class MLPModel(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
        
        def forward(self, x):
            return self.net(x).squeeze()
    
    # 训练
    X_train_t = torch.FloatTensor(X_train_scaled).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val_scaled).to(device)
    
    model = MLPModel(X_train_selected.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=4096, shuffle=True)
    
    best_val_r2 = -999
    patience = 10
    patience_counter = 0
    
    for epoch in range(50):
        model.train()
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t).cpu().numpy()
            val_r2 = r2_score(y_val, val_pred)
        
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        torch_val_pred = model(X_val_t).cpu().numpy()
    
    torch_r2 = r2_score(y_val, torch_val_pred)
    print(f"PyTorch MLP R²: {torch_r2:.6f}")
    mlp_val_pred = torch_val_pred
    use_torch = True
else:
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
    mlp_model.fit(X_train_scaled, y_train)
    mlp_val_pred = mlp_model.predict(X_val_scaled)
    torch_r2 = r2_score(y_val, mlp_val_pred)
    print(f"Sklearn MLP R²: {torch_r2:.6f}")
    use_torch = False

print(f"训练时间: {time.time() - start_time:.1f}秒")

# ============================================================
# 16. 线性模型
# ============================================================
print("\n[16] 线性模型...")

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
ridge_val_pred = ridge_model.predict(X_val_scaled)
ridge_r2 = r2_score(y_val, ridge_val_pred)
print(f"Ridge R²: {ridge_r2:.6f}")

lasso_model = Lasso(alpha=0.001, random_state=42)
lasso_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
lasso_val_pred = lasso_model.predict(X_val_scaled)
lasso_r2 = r2_score(y_val, lasso_val_pred)
print(f"Lasso R²: {lasso_r2:.6f}")

# ============================================================
# 17. 3层Stacking (新增!)
# ============================================================
print("\n[17] 3层Stacking...")

# 第1层: 6个模型
level1_pred = np.column_stack([
    lgb_val_pred, xgb_val_pred, cat_val_pred, mlp_val_pred, ridge_val_pred, lasso_val_pred
])

# 第2层: 3个模型
lgb_l2_1 = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, max_depth=6, verbose=-1, random_state=42)
lgb_l2_1.fit(level1_pred, y_train, sample_weight=sample_weights)
l2_1_pred = lgb_l2_1.predict(level1_pred)

lgb_l2_2 = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, max_depth=4, verbose=-1, random_state=43)
lgb_l2_2.fit(level1_pred, y_train, sample_weight=sample_weights)
l2_2_pred = lgb_l2_2.predict(level1_pred)

ridge_l2 = Ridge(alpha=0.5)
ridge_l2.fit(level1_pred, y_train, sample_weight=sample_weights)
l2_ridge_pred = ridge_l2.predict(level1_pred)

level2_pred = np.column_stack([l2_1_pred, l2_2_pred, l2_ridge_pred])

# 第3层: 最终集成
lgb_l3 = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, verbose=-1, random_state=44)
lgb_l3.fit(level2_pred, y_train, sample_weight=sample_weights)
stack_l3_pred = lgb_l3.predict(level2_pred)
stack_l3_r2 = r2_score(y_val, stack_l3_pred)
print(f"3层Stacking R²: {stack_l3_r2:.6f}")

# ============================================================
# 18. 协变量偏移适应 (新增!)
# ============================================================
print("\n[18] 协变量偏移适应...")

# 计算训练集和验证集的分布差异
train_mean = X_train_selected.mean(axis=0)
train_std = X_train_selected.std(axis=0) + 1e-8
val_mean = X_val_selected.mean(axis=0)
val_std = X_val_selected.std(axis=0) + 1e-8

# 标准化后重新训练
X_train_norm = (X_train_selected - val_mean) / val_std
X_val_norm = (X_val_selected - val_mean) / val_std

lgb_shift = lgb.LGBMRegressor(**{**lgb_base_params, 'n_estimators': 500})
lgb_shift.fit(X_train_norm, y_train, sample_weight=sample_weights)
shift_pred = lgb_shift.predict(X_val_norm)
shift_r2 = r2_score(y_val, shift_pred)
print(f"协变量偏移适应 R²: {shift_r2:.6f}")

# ============================================================
# 19. 集成预测
# ============================================================
print("\n[19] 集成预测...")

weighted_pred = (0.20 * lgb_val_pred + 0.18 * xgb_val_pred + 0.15 * cat_val_pred + 
                0.15 * mlp_val_pred + 0.10 * ridge_val_pred + 0.10 * lasso_val_pred +
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
    'LightGBM_10Bag': lgb_r2,
    'XGBoost_10Bag': xgb_r2,
    'CatBoost': cat_r2,
    'PyTorch_MLP': torch_r2,
    'Ridge': ridge_r2,
    'Lasso': lasso_r2,
    'Stack_3Layer': stack_l3_r2,
    'Covariate_Shift': shift_r2,
    'Weighted': weighted_r2,
    'Calibrated': calibrated_r2,
}
best_method = max(all_r2, key=all_r2.get)
best_r2 = all_r2[best_method]
print(f"\n最佳方法: {best_method}, R²: {best_r2:.6f}")

del X_train, y_train, X_val, y_val
gc.collect()

# ============================================================
# 20. 伪标签
# ============================================================
print("\n[20] 伪标签...")

X_test = test_df[selected_features].fillna(0).values.astype('float32')

if use_torch:
    X_test_scaled = scaler.transform(X_test)
    X_test_t = torch.FloatTensor(X_test_scaled).to(device)
    with torch.no_grad():
        mlp_test_pred = model(X_test_t).cpu().numpy()
else:
    X_test_scaled = scaler.transform(X_test)
    mlp_test_pred = mlp_model.predict(X_test_scaled)

lgb_test_pred = np.mean([m.predict(X_test) for m in lgb_bag_models], axis=0)
xgb_test_pred = np.mean([m.predict(X_test) for m in xgb_bag_preds], axis=0)
ridge_test_pred = ridge_model.predict(X_test_scaled)
lasso_test_pred = lasso_model.predict(X_test_scaled)

if use_cat:
    cat_test_pred = cat_model.predict(X_test)
    test_pred = (0.20 * lgb_test_pred + 0.18 * xgb_test_pred + 0.15 * cat_test_pred + 
                0.15 * mlp_test_pred + 0.10 * ridge_test_pred + 0.10 * lasso_test_pred)
else:
    test_pred = (0.25 * lgb_test_pred + 0.20 * xgb_test_pred + 
                0.15 * mlp_test_pred + 0.15 * ridge_test_pred + 0.10 * lasso_test_pred)

if calibrated_r2 > weighted_r2:
    test_pred = iso.predict(test_pred)

high_conf_mask = np.abs(test_pred) > np.percentile(np.abs(test_pred), 75)
n_pseudo = high_conf_mask.sum()
print(f"高置信度样本: {n_pseudo:,}")

# 最终预测
lgb_test_pred = np.mean([m.predict(X_test) for m in lgb_bag_models], axis=0)
xgb_test_pred = np.mean([m.predict(X_test) for m in xgb_bag_preds], axis=0)

if use_cat:
    final_pred = (0.20 * lgb_test_pred + 0.18 * xgb_test_pred + 0.15 * cat_test_pred + 
                 0.15 * mlp_test_pred + 0.10 * ridge_test_pred + 0.10 * lasso_test_pred +
                 0.12 * stack_l3_pred)
else:
    final_pred = (0.25 * lgb_test_pred + 0.20 * xgb_test_pred + 
                 0.15 * mlp_test_pred + 0.15 * ridge_test_pred + 0.10 * lasso_test_pred +
                 0.15 * stack_l3_pred)

if calibrated_r2 > best_r2:
    final_pred = iso.predict(final_pred)

print(f"测试集预测完成: {len(final_pred):,}")

del X_test, test_df
gc.collect()

# ============================================================
# 21. 生成提交文件
# ============================================================
print("\n[21] 生成提交文件...")

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
print("✅ V20 终极版完成！")
print("="*70)

total_time = time.time() - start_time
print(f"总耗时: {total_time/60:.1f} 分钟")

print(f"\n=== V20 核心优化 ===")
print(f"  1. ⭐⭐⭐⭐⭐ Optuna: {OPTUNA_TRIALS}次试验")
print(f"  2. ⭐⭐⭐⭐⭐ Bagging: {N_BAGS}次")
print(f"  3. ⭐⭐⭐⭐⭐ 全部训练数据")
print(f"  4. ⭐⭐⭐⭐ PyTorch GPU神经网络")
print(f"  5. ⭐⭐⭐⭐ 3层Stacking")
print(f"  6. ⭐⭐⭐ 协变量偏移适应")

print(f"\n=== 模型性能 ===")
for name, r2 in all_r2.items():
    print(f"  {name}: R² = {r2:.6f}")
print(f"  最佳: {best_method}, R² = {best_r2:.6f}")

print(f"\n🏆 预期R²: 0.012 → {best_r2:.4f}")
print("🎯 祝竞赛取得金牌!")
