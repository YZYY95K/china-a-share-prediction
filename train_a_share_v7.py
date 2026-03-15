"""
China A-Share Market Microstructure Prediction
🏆 终极版 V7 - A股特性增强 + Ridge基线 + 加权优化

增强内容:
1. 交易所效应特征 (exchangeid)
2. 细粒度时段特征 (竞价/早盘/午盘/尾盘)
3. 涨跌停挖掘特征 (从匿名特征)
4. Ridge基线对比
5. 加权集成优化
"""

print("="*70)
print("🏆 终极版 V7 - A股特性增强 + 加权集成")
print("="*70)

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import warnings
import gc
import time

warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================
BASE_PATH = '/root/autodl-tmp/data/'
OUTPUT_PATH = '/root/autodl-tmp/'

ID_COLS = ['stockid', 'dateid', 'timeid', 'exchangeid']
TARGET_COLS = ['LabelA', 'LabelB', 'LabelC']
FEATURE_COLS = [f'f{i}' for i in range(384)]

TRAIN_RATIO = 0.85

# ============================================================
# 1. 数据加载
# ============================================================
print("\n[1] 数据加载...")
start_time = time.time()

train_chunks = []
total_rows = 0
for chunk in pd.read_parquet(BASE_PATH + 'train.parquet', chunksize=500000):
    train_chunks.append(chunk)
    total_rows += len(chunk)
    print(f"  已加载: {total_rows:,}")
    if len(train_chunks) >= 6:
        break

train_df = pd.concat(train_chunks, ignore_index=True)
del train_chunks
gc.collect()

test_df = pd.read_parquet(BASE_PATH + 'test.parquet')
print(f"训练: {train_df.shape}, 测试: {test_df.shape}")

# 内存优化
for col in train_df.columns:
    if col not in ID_COLS and train_df[col].dtype == 'float64':
        train_df[col] = train_df[col].astype('float32')
for col in test_df.columns:
    if col not in ID_COLS and test_df[col].dtype == 'float64':
        test_df[col] = test_df[col].astype('float32')
gc.collect()

# ============================================================
# 2. A股特性增强特征工程
# ============================================================
print("\n[2] A股特性增强特征工程...")

def create_a_share_features(df, feature_cols):
    """A股特性增强特征工程"""
    new_features = []
    key_cols = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5']
    key_cols = [c for c in key_cols if c in df.columns]
    
    # ========== A. 基础滞后特征 ==========
    for col in key_cols[:4]:
        for lag in [1, 5, 10]:
            df[f'{col}_lag{lag}'] = df.groupby('stockid')[col].shift(lag)
            new_features.append(f'{col}_lag{lag}')
    
    # ========== B. 滚动特征 ==========
    for col in key_cols[:4]:
        for window in [5, 10]:
            df[f'{col}_mean{window}'] = df.groupby('stockid')[col].transform(
                lambda x: x.rolling(window, min_periods=1).mean())
            df[f'{col}_std{window}'] = df.groupby('stockid')[col].transform(
                lambda x: x.rolling(window, min_periods=1).std())
            new_features.extend([f'{col}_mean{window}', f'{col}_std{window}'])
    
    # ========== C. 差分特征 ==========
    for col in key_cols[:3]:
        df[f'{col}_diff1'] = df.groupby('stockid')[col].diff(1)
        new_features.append(f'{col}_diff1')
    
    # ========== D. 订单流不平衡 ==========
    if 'f2' in key_cols and 'f3' in key_cols:
        df['obi'] = (df['f2'] - df['f3']) / (df['f2'] + df['f3'] + 1e-8)
        new_features.append('obi')
    
    # ========== E. 资金流 ==========
    if 'f4' in key_cols and 'f5' in key_cols:
        df['fund_flow'] = df['f4'] - df['f5']
        df['fund_flow_ratio'] = df['fund_flow'] / (df['f4'] + df['f5'] + 1e-8)
        new_features.extend(['fund_flow', 'fund_flow_ratio'])
    
    # ========== F. 波动率 ==========
    for col in key_cols[:2]:
        rolling_mean = df.groupby('stockid')[col].transform(
            lambda x: x.rolling(10, min_periods=1).mean())
        rolling_std = df.groupby('stockid')[col].transform(
            lambda x: x.rolling(10, min_periods=1).std())
        df[f'{col}_cv'] = rolling_std / (rolling_mean.abs() + 1e-8)
        new_features.append(f'{col}_cv')
    
    # ========== G. 价差特征 ==========
    if 'f0' in key_cols and 'f1' in key_cols:
        mid_price = (df['f0'] + df['f1']) / 2
        df['spread'] = df['f0'] - df['f1']
        df['spread_ratio'] = df['spread'] / (mid_price + 1e-8)
        new_features.extend(['spread', 'spread_ratio'])
        
        # 涨跌停挖掘特征 (基于价格和交易量异常)
        # 假设f0/f1是价格，f2/f3是交易量
        price_range = df.groupby('stockid')[key_cols[0]].transform(
            lambda x: x.max() - x.min() + 1e-8)
        df['price_range_norm'] = (df[key_cols[0]] - df[key_cols[0]].min()) / (price_range + 1e-8)
        new_features.append('price_range_norm')
        
        # 交易量异常检测 (潜在涨跌停)
        vol_ma5 = df.groupby('stockid')[key_cols[2]].transform(
            lambda x: x.rolling(5, min_periods=1).mean())
        df['vol_ratio'] = df[key_cols[2]] / (vol_ma5 + 1e-8)
        new_features.append('vol_ratio')
        
        # 价格跳跃检测
        price_diff = df.groupby('stockid')[key_cols[0]].diff(5)
        price_std = df.groupby('stockid')[key_cols[0]].transform(
            lambda x: x.rolling(10, min_periods=1).std())
        df['price_jump'] = np.abs(price_diff) / (price_std + 1e-8)
        new_features.append('price_jump')
    
    # ========== H. 截面特征 ==========
    for col in key_cols[:3]:
        df[f'{col}_market'] = df.groupby(['dateid', 'timeid'])[col].transform('mean')
        df[f'{col}_vs_market'] = df[col] / (df[f'{col}_market'] + 1e-8) - 1
        new_features.extend([f'{col}_market', f'{col}_vs_market'])
    
    # ========== I. 细粒度时段特征 (A股特性) ==========
    df['hour'] = df['timeid'] // 60
    df['minute'] = df['timeid'] % 60
    
    # 更细的时间段划分
    # 0-14: 集合竞价
    # 15-59: 早盘
    # 60-89: 午间休市 (无数据，假设timeid不含这段)
    # 90-119: 下午开盘
    # 120-239: 尾盘
    
    # 简化版时段
    df['is_auction'] = (df['timeid'] <= 14).astype('int8')
    df['is_morning'] = ((df['timeid'] >= 15) & (df['timeid'] < 120)).astype('int8')
    df['is_afternoon'] = (df['timeid'] >= 120).astype('int8')
    df['is_open'] = (df['timeid'] < 15).astype('int8')
    df['is_close'] = (df['timeid'] > 210).astype('int8')
    
    # 时段交互
    df['is_open_auction'] = df['is_auction'] * df['is_open']
    df['is_close_afternoon'] = df['is_afternoon'] * df['is_close']
    
    new_features.extend(['hour', 'minute', 'is_auction', 'is_morning', 'is_afternoon', 
                        'is_open', 'is_close', 'is_open_auction', 'is_close_afternoon'])
    
    # ========== J. 交易所效应 (A股特性) ==========
    if 'exchangeid' in df.columns:
        # 交易所分组统计
        for col in key_cols[:3]:
            df[f'{col}_exchange_mean'] = df.groupby('exchangeid')[col].transform('mean')
            df[f'{col}_vs_exchange'] = df[col] / (df[f'{col}_exchange_mean'] + 1e-8) - 1
            new_features.extend([f'{col}_exchange_mean', f'{col}_vs_exchange'])
        
        # 交易所订单流差异
        if 'f2' in key_cols and 'f3' in key_cols:
            df['obi_exchange'] = df.groupby('exchangeid')['obi'].transform('mean')
            df['obi_vs_exchange'] = df['obi'] - df['obi_exchange']
            new_features.extend(['obi_exchange', 'obi_vs_exchange'])
        
        # 交易所交易量差异
        if 'f4' in key_cols and 'f5' in key_cols:
            df['fund_flow_exchange'] = df.groupby('exchangeid')['fund_flow'].transform('mean')
            new_features.append('fund_flow_exchange')
        
        # 交易所编码 (转为数值)
        df['exchange_encoded'] = df['exchangeid'].astype('category').cat.codes.astype('int8')
        new_features.append('exchange_encoded')
    
    # ========== K. 交易密度特征 ==========
    # 每个时间点的交易密度
    time_density = df.groupby(['dateid', 'timeid']).size().reset_index(name='n_stocks')
    df = df.merge(time_density, on=['dateid', 'timeid'], how='left')
    new_features.append('n_stocks')
    
    return df, new_features

train_df, _ = create_a_share_features(train_df, FEATURE_COLS)
test_df, _ = create_a_share_features(test_df, FEATURE_COLS)
gc.collect()

# ============================================================
# 3. 时间序列划分
# ============================================================
print("\n[3] 时间序列划分...")

train_df = train_df.sort_values(['stockid', 'dateid', 'timeid']).reset_index(drop=True)
unique_dates = sorted(train_df['dateid'].unique())
n_dates = len(unique_dates)
print(f"总日期数: {n_dates}")

n_train_dates = int(n_dates * TRAIN_RATIO)
train_dates = unique_dates[:n_train_dates]
val_dates = unique_dates[n_train_dates:]

print(f"训练日期: {len(train_dates)}, 验证日期: {len(val_dates)}")

train_data = train_df[train_df['dateid'].isin(train_dates)].copy()
val_data = train_df[train_df['dateid'].isin(val_dates)].copy()

print(f"训练: {len(train_data):,}, 验证: {len(val_data):,}")

del train_df
gc.collect()

# ============================================================
# 4. 准备特征
# ============================================================
print("\n[4] 准备特征...")

exclude_cols = ID_COLS + TARGET_COLS + ['hour', 'minute', 'n_stocks']
all_features = FEATURE_COLS + [c for c in train_data.columns if c not in exclude_cols]
all_features = [f for f in all_features if f in train_data.columns]
print(f"特征数: {len(all_features)}")

# 填充
for col in all_features:
    if col in train_data.columns:
        train_data[col] = train_data[col].fillna(0).replace([np.inf, -np.inf], 0)
    if col in val_data.columns:
        val_data[col] = val_data[col].fillna(0).replace([np.inf, -np.inf], 0)
    if col in test_df.columns:
        test_df[col] = test_df[col].fillna(0).replace([np.inf, -np.inf], 0)

X_train = train_data[all_features].values.astype('float32')
y_train = train_data[TARGET_COLS].values

X_val = val_data[all_features].values.astype('float32')
y_val = val_data[TARGET_COLS].values

X_test = test_df[all_features].values.astype('float32')

gc.collect()

# ============================================================
# 5. 模型训练 - Ridge基线对比
# ============================================================
print("\n[5] 模型训练 - Ridge基线 + LGB + XGB...")
print("="*50)

target_idx = 0
y_tr = y_train[:, target_idx]
y_vl = y_val[:, target_idx]

y_tr_clean = np.clip(y_tr, np.percentile(y_tr, 1), np.percentile(y_tr, 99))

# 5.1 Ridge基线
print("\n[5.1] Ridge线性基线...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_tr)
ridge_val_pred = ridge.predict(X_val_scaled)
ridge_r2 = r2_score(y_vl, ridge_val_pred)
print(f"Ridge R²: {ridge_r2:.6f}")

# 5.2 LightGBM
print("\n[5.2] LightGBM...")

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'max_depth': 6,
    'learning_rate': 0.02,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.6,
    'bagging_freq': 5,
    'min_data_in_leaf': 500,
    'lambda_l1': 0.5,
    'lambda_l2': 1.0,
    'verbose': -1,
    'seed': 42,
}

lgb_train = lgb.Dataset(X_train, label=y_tr_clean)
lgb_val = lgb.Dataset(X_val, label=y_vl, reference=lgb_train)

model_lgb = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=2000,
    valid_sets=[lgb_val],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)]
)

lgb_val_pred = model_lgb.predict(X_val)
lgb_r2 = r2_score(y_vl, lgb_val_pred)
print(f"LightGBM R²: {lgb_r2:.6f}")

# 5.3 XGBoost
print("\n[5.3] XGBoost...")
dtrain = xgb.DMatrix(X_train, label=y_tr_clean)
dval = xgb.DMatrix(X_val, label=y_vl)
dtest = xgb.DMatrix(X_test)

xgb_params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'learning_rate': 0.02,
    'subsample': 0.6,
    'colsample_bytree': 0.5,
    'min_child_weight': 500,
    'reg_alpha': 0.5,
    'reg_lambda': 1.0,
    'seed': 42,
    'tree_method': 'hist',
    'device': 'cuda',
}

model_xgb = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=2000,
    evals=[(dval, 'valid')],
    early_stopping_rounds=100,
    verbose_eval=500
)

xgb_val_pred = model_xgb.predict(dval)
xgb_r2 = r2_score(y_vl, xgb_val_pred)
print(f"XGBoost R²: {xgb_r2:.6f}")

gc.collect()

# ============================================================
# 6. 加权集成优化
# ============================================================
print("\n[6] 加权集成优化...")
print("="*50)

# 模型预测汇总
val_predictions = {
    'ridge': ridge_val_pred,
    'lightgbm': lgb_val_pred,
    'xgboost': xgb_val_pred
}

# 优化权重 (最小化MSE)
def optimize_weights(predictions_dict, y_true):
    model_names = list(predictions_dict.keys())
    n_models = len(model_names)
    
    def objective(weights):
        weights = np.array(weights)
        weights = weights / weights.sum()
        ensemble = sum(predictions_dict[name] * w for name, w in zip(model_names, weights))
        return np.mean((y_true - ensemble) ** 2)
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in range(n_models)]
    
    result = minimize(objective, x0=np.ones(n_models) / n_models,
                     method='SLSQP', bounds=bounds, constraints=constraints)
    
    return {name: w for name, w in zip(model_names, result.x)}

# 优化权重
optimal_weights = optimize_weights(val_predictions, y_vl)
print("\n优化后的权重:")
for name, w in optimal_weights.items():
    print(f"  {name}: {w:.4f}")

# 加权集成
weighted_val_pred = sum(val_predictions[name] * optimal_weights[name] 
                       for name in val_predictions.keys())
weighted_r2 = r2_score(y_vl, weighted_val_pred)
print(f"\n加权集成 R²: {weighted_r2:.6f}")

# 简单平均
simple_avg_pred = (lgb_val_pred + xgb_val_pred) / 2
simple_r2 = r2_score(y_vl, simple_avg_pred)
print(f"简单平均 R²: {simple_r2:.6f}")

# 选择最佳方法
methods = {
    'ridge': ridge_r2,
    'lightgbm': lgb_r2,
    'xgboost': xgb_r2,
    'simple_avg': simple_r2,
    'weighted': weighted_r2
}
best_method = max(methods, key=methods.get)
best_r2 = methods[best_method]
print(f"\n最佳方法: {best_method}, R² = {best_r2:.6f}")

# ============================================================
# 7. 生成提交
# ============================================================
print("\n[7] 生成提交...")

ridge_test_pred = ridge.predict(X_test_scaled)
lgb_test_pred = model_lgb.predict(X_test)
xgb_test_pred = model_xgb.predict(dtest)

test_predictions = {
    'ridge': ridge_test_pred,
    'lightgbm': lgb_test_pred,
    'xgboost': xgb_test_pred
}

if best_method == 'weighted':
    final_pred = sum(test_predictions[name] * optimal_weights[name] 
                    for name in test_predictions.keys())
elif best_method == 'simple_avg':
    final_pred = (lgb_test_pred + xgb_test_pred) / 2
elif best_method == 'lightgbm':
    final_pred = lgb_test_pred
elif best_method == 'xgboost':
    final_pred = xgb_test_pred
else:
    final_pred = lgb_test_pred

# 预测截断 (后处理)
lower_bound = np.percentile(y_tr, 1)
upper_bound = np.percentile(y_tr, 99)
final_pred = np.clip(final_pred, lower_bound, upper_bound)

test_df['Uid'] = (
    test_df['stockid'].astype(str) + '|' +
    test_df['dateid'].astype(str) + '|' +
    test_df['timeid'].astype(str)
)

submission = pd.DataFrame({
    'Uid': test_df['Uid'],
    'prediction': final_pred
})

submission.to_csv(OUTPUT_PATH + 'submission.csv', index=False)

print(f"\n提交文件: {OUTPUT_PATH}submission.csv")
print(f"提交行数: {len(submission):,}")

# ============================================================
# 完成
# ============================================================
print("\n" + "="*70)
print("🏆 训练完成!")
print("="*70)

total_time = time.time() - start_time
print(f"耗时: {total_time/60:.1f} 分钟")

print(f"\n=== A股特性增强 ===")
print(f"  特征数: {len(all_features)}")
print(f"  交易所效应: ✅")
print(f"  细粒度时段: ✅")
print(f"  涨跌停挖掘: ✅")

print(f"\n=== 模型性能对比 ===")
print(f"  Ridge:      R² = {ridge_r2:.6f} (基线)")
print(f"  LightGBM:   R² = {lgb_r2:.6f}")
print(f"  XGBoost:    R² = {xgb_r2:.6f}")
print(f"  简单平均:   R² = {simple_r2:.6f}")
print(f"  加权集成:   R² = {weighted_r2:.6f}")
print(f"  最佳:       {best_method}, R² = {best_r2:.6f}")

print(f"\n=== 集成权重 ===")
for name, w in optimal_weights.items():
    print(f"  {name}: {w:.4f}")

print("\n✅ 完成!")
