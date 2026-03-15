"""
China A-Share Market Microstructure Prediction
🏆 优化版 V11 - GPU + 内存优化 + 跨股票特征 + 树模型

优化重点：
1. 内存优化 - 分批处理、float32
2. 跨股票特征 - 核心提升点
3. 树模型 - LightGBM + XGBoost
4. 移除神经网络 - 不适合
"""

print("="*70)
print("🏆 V11 - GPU + 内存优化 + 跨股票特征 + 树模型")
print("="*70)

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
import warnings
import gc
import time

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================
# 配置 - Kaggle P100 30GB 优化
# ============================================================
BASE_PATH = '/kaggle/input/competitions/china-a-share-market-microstructure-prediction/'
OUTPUT_PATH = '/kaggle/working/'

print(f"数据路径: {BASE_PATH}")
print(f"输出路径: {OUTPUT_PATH}")

ID_COLS = ['stockid', 'dateid', 'timeid', 'exchangeid']
TARGET_COLS = ['LabelA', 'LabelB', 'LabelC']
FEATURE_COLS = [f'f{i}' for i in range(384)]

# P100 30GB 优化配置
BATCH_SIZE = 300000
USE_GPU = True

# ============================================================
# 1. 数据加载（内存优化）
# ============================================================
print("\n[1] 数据加载（内存优化）...")
start_time = time.time()

# 分批加载训练数据
train_chunks = []
total_rows = 0
for chunk in pd.read_parquet(BASE_PATH + 'train.parquet', chunksize=BATCH_SIZE):
    # 内存优化：转换为float32
    for col in chunk.columns:
        if col not in ID_COLS and chunk[col].dtype == 'float64':
            chunk[col] = chunk[col].astype('float32')
    
    train_chunks.append(chunk)
    total_rows += len(chunk)
    print(f"  已加载: {total_rows:,}")
    
    # 只加载前4个chunk（约200万行）
    if len(train_chunks) >= 4:
        break

train_df = pd.concat(train_chunks, ignore_index=True)
del train_chunks
gc.collect()

# 加载测试数据
test_df = pd.read_parquet(BASE_PATH + 'test.parquet')
for col in test_df.columns:
    if col not in ID_COLS and test_df[col].dtype == 'float64':
        test_df[col] = test_df[col].astype('float32')

print(f"训练: {train_df.shape}, 测试: {test_df.shape}")
print(f"加载时间: {time.time() - start_time:.1f}秒")

# ============================================================
# 2. 跨股票特征工程（核心！）
# ============================================================
print("\n[2] 跨股票特征工程（核心）...")
start_time = time.time()

# 选择Top重要特征（根据之前分析）
TOP_FEATURES = ['f298', 'f105', 'f128', 'f28', 'f46', 'f326', 'f124', 
                'f314', 'f259', 'f334', 'f171', 'f350', 'f176', 'f170', 
                'f319', 'f126', 'f182', 'f349', 'f372', 'f0', 'f1', 'f2', 'f3', 'f4', 'f5']

# 确保特征存在
TOP_FEATURES = [f for f in TOP_FEATURES if f in train_df.columns]
print(f"使用Top特征数: {len(TOP_FEATURES)}")

def create_cross_stock_features(df, features):
    """跨股票特征工程"""
    new_cols = []
    
    # 对每个重要特征计算截面统计
    for col in features:
        # 截面均值
        df[f'{col}_cross_mean'] = df.groupby(['dateid', 'timeid'])[col].transform('mean')
        new_cols.append(f'{col}_cross_mean')
        
        # 截面标准差
        df[f'{col}_cross_std'] = df.groupby(['dateid', 'timeid'])[col].transform('std')
        new_cols.append(f'{col}_cross_std')
        
        # Z-score（相对价值）
        df[f'{col}_zscore'] = (df[col] - df[f'{col}_cross_mean']) / (df[f'{col}_cross_std'] + 1e-8)
        new_cols.append(f'{col}_zscore')
        
        # 排名百分位
        df[f'{col}_rank_pct'] = df.groupby(['dateid', 'timeid'])[col].rank(pct=True)
        new_cols.append(f'{col}_rank_pct')
    
    # 计算价格变化（用于市场情绪）
    if 'f298' in df.columns:
        df['price_change'] = df.groupby('stockid').groupby('dateid')['f298'].diff()
        
        # 市场情绪指标
        df['market_up_ratio'] = df.groupby(['dateid', 'timeid'])['price_change'].apply(
            lambda x: (x > 0).mean()
        ).reset_index(level=[0,1], drop=True)
        new_cols.append('market_up_ratio')
        
        df['market_avg_change'] = df.groupby(['dateid', 'timeid'])['price_change'].transform('mean')
        new_cols.append('market_avg_change')
        
        df['market_volatility'] = df.groupby(['dateid', 'timeid'])['price_change'].transform('std')
        new_cols.append('market_volatility')
    
    gc.collect()
    return df, new_cols

# 应用跨股票特征
print("  处理训练集...")
train_df, cross_features = create_cross_stock_features(train_df, TOP_FEATURES)
print(f"  跨股票特征数: {len(cross_features)}")

print("  处理测试集...")
test_df, _ = create_cross_stock_features(test_df, TOP_FEATURES)

print(f"  特征工程时间: {time.time() - start_time:.1f}秒")

# ============================================================
# 3. 基础时间序列特征
# ============================================================
print("\n[3] 基础时间序列特征...")

key_cols = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5']
key_cols = [c for c in key_cols if c in train_df.columns]

new_features = []

# 滞后特征
for col in key_cols[:4]:
    for lag in [1, 5]:
        train_df[f'{col}_lag{lag}'] = train_df.groupby('stockid')[col].shift(lag)
        test_df[f'{col}_lag{lag}'] = test_df.groupby('stockid')[col].shift(lag)
        new_features.extend([f'{col}_lag{lag}'])

# 滚动特征
for col in key_cols[:3]:
    train_df[f'{col}_mean5'] = train_df.groupby('stockid')[col].transform(
        lambda x: x.rolling(5, min_periods=1).mean())
    test_df[f'{col}_mean5'] = test_df.groupby('stockid')[col].transform(
        lambda x: x.rolling(5, min_periods=1).mean())
    new_features.append(f'{col}_mean5')

gc.collect()

# ============================================================
# 4. 时间序列划分
# ============================================================
print("\n[4] 时间序列划分...")

train_df = train_df.sort_values(['stockid', 'dateid', 'timeid']).reset_index(drop=True)
unique_dates = sorted(train_df['dateid'].unique())
n_dates = len(unique_dates)
print(f"总日期数: {n_dates}")

# 85%训练，15%验证
n_train_dates = int(n_dates * 0.85)
train_dates = unique_dates[:n_train_dates]
val_dates = unique_dates[n_train_dates:]

print(f"训练日期: {len(train_dates)}, 验证日期: {len(val_dates)}")

train_data = train_df[train_df['dateid'].isin(train_dates)].copy()
val_data = train_df[train_df['dateid'].isin(val_dates)].copy()

print(f"训练: {len(train_data):,}, 验证: {len(val_data):,}")

del train_df
gc.collect()

# ============================================================
# 5. 准备特征
# ============================================================
print("\n[5] 准备特征...")

exclude_cols = ID_COLS + TARGET_COLS
all_features = FEATURE_COLS + cross_features + new_features
all_features = [f for f in all_features if f in train_data.columns and f in test_df.columns]
all_features = list(set(all_features))

print(f"总特征数: {len(all_features)}")

# 填充NaN
train_data[all_features] = train_data[all_features].fillna(0)
val_data[all_features] = val_data[all_features].fillna(0)

X_train = train_data[all_features].values.astype('float32')
y_train = train_data['LabelA'].values.astype('float32')
X_val = val_data[all_features].values.astype('float32')
y_val = val_data['LabelA'].values.astype('float32')

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")

del train_data, val_data
gc.collect()

# ============================================================
# 6. LightGBM 模型
# ============================================================
print("\n[6] LightGBM 模型...")
start_time = time.time()

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 8,
    'num_leaves': 63,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1,
}

lgb_model = lgb.LGBMRegressor(**lgb_params)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False)]
)

lgb_val_pred = lgb_model.predict(X_val)
lgb_r2 = r2_score(y_val, lgb_val_pred)
print(f"LightGBM R²: {lgb_r2:.6f}")
print(f"训练时间: {time.time() - start_time:.1f}秒")

# 特征重要性
importance = pd.DataFrame({
    'feature': all_features,
    'importance': lgb_model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nTop 10 特征:")
print(importance.head(10))

# ============================================================
# 7. XGBoost 模型
# ============================================================
print("\n[7] XGBoost 模型...")
start_time = time.time()

xgb_params = {
    'objective': 'reg:squarederror',
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 8,
    'min_child_weight': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'tree_method': 'gpu_hist' if USE_GPU else 'hist',
    'gpu_id': 0,
}

xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50,
    early_stopping_rounds=50
)

xgb_val_pred = xgb_model.predict(X_val)
xgb_r2 = r2_score(y_val, xgb_val_pred)
print(f"XGBoost R²: {xgb_r2:.6f}")
print(f"训练时间: {time.time() - start_time:.1f}秒")

# ============================================================
# 8. 集成预测
# ============================================================
print("\n[8] 集成预测...")

# 简单平均
ensemble_pred = (lgb_val_pred + xgb_val_pred) / 2
ensemble_r2 = r2_score(y_val, ensemble_pred)
print(f"集成 R²: {ensemble_r2:.6f}")

# 选择最佳模型
best_r2 = max(lgb_r2, xgb_r2, ensemble_r2)
if best_r2 == lgb_r2:
    print("最佳模型: LightGBM")
    best_model = 'lgb'
elif best_r2 == xgb_r2:
    print("最佳模型: XGBoost")
    best_model = 'xgb'
else:
    print("最佳模型: 集成")
    best_model = 'ensemble'

del X_train, y_train, X_val, y_val
gc.collect()

# ============================================================
# 9. 测试集预测
# ============================================================
print("\n[9] 测试集预测...")

# 准备测试数据
X_test = test_df[all_features].fillna(0).values.astype('float32')

# 预测
lgb_test_pred = lgb_model.predict(X_test)
xgb_test_pred = xgb_model.predict(X_test)

if best_model == 'lgb':
    final_pred = lgb_test_pred
elif best_model == 'xgb':
    final_pred = xgb_test_pred
else:
    final_pred = (lgb_test_pred + xgb_test_pred) / 2

print(f"测试集预测完成: {len(final_pred):,}")

del X_test, test_df
gc.collect()

# ============================================================
# 10. 生成提交文件
# ============================================================
print("\n[10] 生成提交文件...")

# 重新加载测试数据获取ID
test_df = pd.read_parquet(BASE_PATH + 'test.parquet', columns=['stockid', 'dateid', 'timeid'])

# 创建提交文件
submission = pd.DataFrame({
    'Uid': test_df['stockid'].astype(str) + '|' + 
           test_df['dateid'].astype(str) + '|' + 
           test_df['timeid'].astype(str),
    'prediction': final_pred
})

# 保存
submission.to_csv(OUTPUT_PATH + 'submission.csv', index=False)
print(f"提交文件保存: {OUTPUT_PATH}submission.csv")
print(f"样本数: {len(submission):,}")
print(submission.head())

print("\n" + "="*70)
print("✅ 完成！")
print(f"最佳验证R²: {best_r2:.6f}")
print("="*70)
