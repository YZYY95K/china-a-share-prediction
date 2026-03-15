"""
China A-Share Market Microstructure Prediction
🏆 终极版 V9 - 全优化集成 + 深度学习

增强内容:
1. 交易所效应特征 (exchangeid)
2. 细粒度时段特征 (竞价/早盘/午盘/尾盘)
3. 涨跌停核心特征 (价格位置、封板强度)
4. VPIN订单流毒性特征
5. Ridge基线对比
6. 加权集成优化
7. 截面标准化后处理 (关键！)
8. 样本权重优化 (时间+涨跌停)
9. Stacking集成
10. EMA平滑后处理
11. LSTM深度学习模型
"""

print("="*70)
print("🏆 终极版 V9 - 全优化集成 + 深度学习")
print("="*70)

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor
from scipy.optimize import minimize
import warnings
import gc
import time

# PyTorch for LSTM
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

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

# ============================================================
# 2.1 涨跌停核心特征 (V8新增)
# ============================================================
def add_limit_features(df):
    """
    涨跌停核心特征 - A股市场关键特征
    包括：价格位置、接近涨跌停标识、封板强度
    """
    key_cols = ['f0', 'f1', 'f2', 'f3']
    key_cols = [c for c in key_cols if c in df.columns]
    
    if len(key_cols) >= 2:
        # 计算日内价格位置 (0=最低, 1=最高)
        daily_stats = df.groupby(['stockid', 'dateid'])[key_cols[0]].agg(['min', 'max'])
        daily_stats.columns = ['daily_min', 'daily_max']
        df = df.merge(daily_stats, left_on=['stockid', 'dateid'], 
                     right_index=True, how='left')
        
        # 价格位置 (0-1之间)
        df['price_position'] = (df[key_cols[0]] - df['daily_min']) / \
                               (df['daily_max'] - df['daily_min'] + 1e-8)
        
        # 接近涨停 (>95%位置)
        df['near_limit_up'] = (df['price_position'] > 0.95).astype('int8')
        # 接近跌停 (<5%位置)
        df['near_limit_down'] = (df['price_position'] < 0.05).astype('int8')
        # 任一涨跌停状态
        df['near_limit'] = df['near_limit_up'] | df['near_limit_down']
        
        # 封板强度 (关键特征)
        # 涨停时看买盘力量，跌停时看卖盘力量
        df['limit_strength'] = 0.0
        
        if 'f2' in df.columns and 'f3' in df.columns:
            # 接近涨停：买盘/卖盘比值
            mask_up = df['near_limit_up'] == 1
            df.loc[mask_up, 'limit_strength'] = df.loc[mask_up, 'f2'] / \
                                                (df.loc[mask_up, 'f3'] + 1e-8)
            
            # 接近跌停：-卖盘/买盘比值
            mask_down = df['near_limit_down'] == 1
            df.loc[mask_down, 'limit_strength'] = -df.loc[mask_down, 'f3'] / \
                                                  (df.loc[mask_down, 'f2'] + 1e-8)
        
        # 连续涨跌停计数
        df['consecutive_up'] = df.groupby('stockid')['near_limit_up'].cumsum()
        df['consecutive_down'] = df.groupby('stockid')['near_limit_down'].cumsum()
        
        # 价格动量 (向涨跌停方向的速度)
        df['price_velocity'] = df.groupby('stockid')['price_position'].diff()
        df['price_acceleration'] = df.groupby('stockid')['price_velocity'].diff()
    
    return df

# ============================================================
# 2.2 VPIN订单流毒性特征 (V8新增)
# ============================================================
def add_vpin_features(df, window=50):
    """
    VPIN (Volume-synchronized Probability of Informed Trading)
    订单流毒性指标 - 衡量知情交易者的存在
    """
    key_cols = ['f0', 'f4']  # 价格和成交量
    key_cols = [c for c in key_cols if c in df.columns]
    
    if len(key_cols) >= 2:
        # 计算价格变动方向
        df['price_diff'] = df.groupby('stockid')[key_cols[0]].diff()
        df['trade_direction'] = np.sign(df['price_diff'])
        
        # 有方向的成交量
        df['signed_volume'] = df['trade_direction'] * df[key_cols[1]]
        
        # VPIN计算：滚动窗口内的订单流不平衡
        df['vpin'] = df.groupby('stockid')['signed_volume'].transform(
            lambda x: x.rolling(window, min_periods=10).apply(
                lambda y: np.abs(y).sum() / (np.abs(y).sum() + 1e-8), raw=True
            )
        )
        
        # 订单流不平衡绝对值
        df['volume_imbalance'] = df.groupby('stockid')['signed_volume'].transform(
            lambda x: x.rolling(window, min_periods=10).mean()
        )
        
        # 买卖压力差
        df['buy_pressure'] = df.groupby('stockid').apply(
            lambda x: x[x['signed_volume'] > 0]['signed_volume'].rolling(window, min_periods=1).sum()
        ).reset_index(level=0, drop=True).fillna(0)
        
        df['sell_pressure'] = df.groupby('stockid').apply(
            lambda x: x[x['signed_volume'] < 0]['signed_volume'].abs().rolling(window, min_periods=1).sum()
        ).reset_index(level=0, drop=True).fillna(0)
        
        df['pressure_ratio'] = df['buy_pressure'] / (df['sell_pressure'] + 1e-8)
    
    return df

# ============================================================
# 2.3 截面标准化后处理 (V8关键新增)
# ============================================================
def cross_sectional_standardize(predictions, stock_ids, date_ids, time_ids):
    """
    截面标准化 - 对每个时间点的预测值进行标准化
    这对Panel R²评估至关重要！
    
    Panel R² = 1 - Σ(y_true - y_pred)² / Σ(y_true - y_mean)²
    截面标准化使预测值在每个时间点内具有可比性
    """
    df = pd.DataFrame({
        'pred': predictions,
        'stock_id': stock_ids,
        'date': date_ids,
        'time': time_ids
    })
    
    # 按时间点分组标准化 (z-score)
    df['pred_std'] = df.groupby(['date', 'time'])['pred'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )
    
    # 如果标准差为0，保持原值
    df['pred_std'] = df['pred_std'].fillna(df['pred'])
    
    return df['pred_std'].values

# ============================================================
# 2.4 样本权重计算 (V9新增)
# ============================================================
def compute_sample_weights(df, time_col='dateid', limit_col='near_limit'):
    """
    计算样本权重 - 时间权重 + 涨跌停降权
    """
    weights = np.ones(len(df))
    
    # 1. 时间权重：近期样本更高权重
    if time_col in df.columns:
        unique_times = df[time_col].unique()
        n_times = len(unique_times)
        # 线性递增权重 0.8 -> 1.0
        time_weights = np.linspace(0.8, 1.0, n_times)
        time_weight_map = dict(zip(unique_times, time_weights))
        weights *= df[time_col].map(time_weight_map).values
    
    # 2. 涨跌停降权：限制样本降低权重
    if limit_col in df.columns:
        limit_mask = df[limit_col].values if df[limit_col].dtype == bool else df[limit_col].values > 0
        weights[limit_mask] *= 0.7  # 涨跌停样本权重降为0.7
    
    # 3. 极端收益降权
    if 'LabelA' in df.columns:
        returns = df['LabelA'].values
        extreme_threshold = np.percentile(np.abs(returns), 95)
        extreme_mask = np.abs(returns) > extreme_threshold
        weights[extreme_mask] *= 0.8
    
    return weights

# ============================================================
# 2.5 LSTM深度学习模型 (V9新增)
# ============================================================
class TimeSeriesDataset(Dataset):
    """时序数据集"""
    def __init__(self, X, y=None, seq_len=20):
        self.X = X
        self.y = y
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.X) - self.seq_len
    
    def __getitem__(self, idx):
        x = self.X[idx:idx+self.seq_len]
        if self.y is not None:
            y = self.y[idx+self.seq_len-1]
            return torch.FloatTensor(x), torch.FloatTensor([y])
        return torch.FloatTensor(x)

class LSTMModel(nn.Module):
    """LSTM时序预测模型"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, (hidden, cell) = self.lstm(x)
        # 取最后一个时间步
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out.squeeze()

def train_lstm(X_train, y_train, X_val, y_val, input_size, seq_len=20, 
               epochs=50, batch_size=256, lr=0.001):
    """训练LSTM模型"""
    print(f"\n[5.4] LSTM深度学习模型...")
    print(f"  序列长度: {seq_len}, 隐藏层: 64, 层数: 2")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  使用设备: {device}")
    
    # 创建数据集
    train_dataset = TimeSeriesDataset(X_train, y_train, seq_len)
    val_dataset = TimeSeriesDataset(X_val, y_val, seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    model = LSTMModel(input_size=input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    patience = 10
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证
        model.eval()
        val_loss = 0
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y.squeeze())
                val_loss += loss.item()
                
                val_preds.extend(outputs.cpu().numpy())
                val_true.extend(batch_y.squeeze().cpu().numpy())
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_r2 = r2_score(val_true, val_preds)
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val R²: {val_r2:.6f}")
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  早停于 Epoch {epoch+1}")
                break
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # 生成验证集预测
    model.eval()
    val_preds_final = []
    with torch.no_grad():
        for batch_x, _ in val_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            val_preds_final.extend(outputs.cpu().numpy())
    
    return model, np.array(val_preds_final)

def predict_lstm(model, X_test, seq_len=20, batch_size=256):
    """使用LSTM生成测试集预测"""
    device = next(model.parameters()).device
    
    # 为测试集创建序列（使用前seq_len个训练样本）
    # 简化处理：直接使用最后seq_len个特征
    X_test_seq = []
    for i in range(len(X_test)):
        if i < seq_len:
            # 前面不足seq_len的，用第一个特征补齐
            pad = np.tile(X_test[0], (seq_len - i - 1, 1))
            seq = np.vstack([pad, X_test[:i+1]])
        else:
            seq = X_test[i-seq_len+1:i+1]
        X_test_seq.append(seq)
    
    X_test_seq = np.array(X_test_seq)
    test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_test_seq))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    test_preds = []
    with torch.no_grad():
        for (batch_x,) in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            test_preds.extend(outputs.cpu().numpy())
    
    return np.array(test_preds)

# 应用特征工程
print("\n[2] A股特性增强特征工程...")
train_df, _ = create_a_share_features(train_df, FEATURE_COLS)
test_df, _ = create_a_share_features(test_df, FEATURE_COLS)

print("\n[2.1] 添加涨跌停核心特征...")
train_df = add_limit_features(train_df)
test_df = add_limit_features(test_df)

print("\n[2.2] 添加VPIN订单流毒性特征...")
train_df = add_vpin_features(train_df, window=50)
test_df = add_vpin_features(test_df, window=50)

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

# ============================================================
# 4.1 计算样本权重 (V9新增)
# ============================================================
print("\n[4.1] 计算样本权重...")
train_weights = compute_sample_weights(train_data)
print(f"  权重范围: [{train_weights.min():.3f}, {train_weights.max():.3f}]")
print(f"  平均权重: {train_weights.mean():.3f}")

gc.collect()

# ============================================================
# 5. 模型训练 - Ridge基线 + LGB + XGB + LSTM
# ============================================================
print("\n[5] 模型训练 - Ridge基线 + LGB + XGB + LSTM...")
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
ridge.fit(X_train_scaled, y_tr, sample_weight=train_weights)
ridge_val_pred = ridge.predict(X_val_scaled)
ridge_r2 = r2_score(y_vl, ridge_val_pred)
print(f"Ridge R²: {ridge_r2:.6f}")

# 5.2 LightGBM (带样本权重)
print("\n[5.2] LightGBM (带样本权重)...")

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

lgb_train = lgb.Dataset(X_train, label=y_tr_clean, weight=train_weights)
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

# 5.4 LSTM深度学习 (V9新增)
print("\n[5.4] LSTM深度学习模型...")
try:
    # 只对前100个特征训练LSTM（避免维度灾难）
    n_lstm_features = min(100, X_train.shape[1])
    X_train_lstm = X_train[:, :n_lstm_features]
    X_val_lstm = X_val[:, :n_lstm_features]
    
    model_lstm, lstm_val_pred = train_lstm(
        X_train_lstm, y_tr_clean, 
        X_val_lstm, y_vl,
        input_size=n_lstm_features,
        seq_len=20,
        epochs=30,  # 减少轮数加快训练
        batch_size=512
    )
    lstm_r2 = r2_score(y_vl, lstm_val_pred)
    print(f"LSTM R²: {lstm_r2:.6f}")
    use_lstm = True
except Exception as e:
    print(f"  LSTM训练失败: {e}")
    print("  跳过LSTM，继续使用树模型")
    lstm_val_pred = None
    lstm_r2 = -999
    use_lstm = False

gc.collect()

# ============================================================
# 6. 集成优化 - 加权 + Stacking
# ============================================================
print("\n[6] 集成优化 - 加权 + Stacking...")
print("="*50)

# 模型预测汇总
val_predictions = {
    'ridge': ridge_val_pred,
    'lightgbm': lgb_val_pred,
    'xgboost': xgb_val_pred
}

# 如果LSTM成功，加入集成
if use_lstm:
    val_predictions['lstm'] = lstm_val_pred
    print(f"  包含LSTM，共{len(val_predictions)}个模型")
else:
    print(f"  不包含LSTM，共{len(val_predictions)}个模型")

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
if use_lstm:
    simple_avg_pred = (lgb_val_pred + xgb_val_pred + lstm_val_pred) / 3
else:
    simple_avg_pred = (lgb_val_pred + xgb_val_pred) / 2
simple_r2 = r2_score(y_vl, simple_avg_pred)
print(f"简单平均 R²: {simple_r2:.6f}")

# 6.2 Stacking集成 (V9新增)
print("\n[6.2] Stacking集成...")
try:
    from sklearn.linear_model import Ridge as RidgeMeta
    
    # 构建元特征
    meta_features = np.column_stack([pred for pred in val_predictions.values()])
    
    # 元学习器
    meta_model = RidgeMeta(alpha=1.0)
    meta_model.fit(meta_features, y_vl)
    
    stacking_val_pred = meta_model.predict(meta_features)
    stacking_r2 = r2_score(y_vl, stacking_val_pred)
    print(f"Stacking R²: {stacking_r2:.6f}")
    print(f"  元学习器权重: {dict(zip(val_predictions.keys(), meta_model.coef_))}")
    use_stacking = True
except Exception as e:
    print(f"  Stacking失败: {e}")
    stacking_r2 = -999
    use_stacking = False

# 选择最佳方法
methods = {
    'ridge': ridge_r2,
    'lightgbm': lgb_r2,
    'xgboost': xgb_r2,
    'simple_avg': simple_r2,
    'weighted': weighted_r2,
}
if use_lstm:
    methods['lstm'] = lstm_r2
if use_stacking:
    methods['stacking'] = stacking_r2

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

# LSTM预测
if use_lstm:
    print("\n[7.0] LSTM预测...")
    X_test_lstm = X_test[:, :n_lstm_features]
    lstm_test_pred = predict_lstm(model_lstm, X_test_lstm, seq_len=20)
    test_predictions['lstm'] = lstm_test_pred
    print(f"  LSTM预测完成")

# 根据最佳方法生成预测
if best_method == 'weighted':
    final_pred = sum(test_predictions[name] * optimal_weights[name] 
                    for name in test_predictions.keys())
elif best_method == 'simple_avg':
    if use_lstm:
        final_pred = (lgb_test_pred + xgb_test_pred + lstm_test_pred) / 3
    else:
        final_pred = (lgb_test_pred + xgb_test_pred) / 2
elif best_method == 'stacking' and use_stacking:
    # Stacking预测
    meta_features_test = np.column_stack([pred for pred in test_predictions.values()])
    final_pred = meta_model.predict(meta_features_test)
elif best_method == 'lstm' and use_lstm:
    final_pred = lstm_test_pred
elif best_method == 'lightgbm':
    final_pred = lgb_test_pred
elif best_method == 'xgboost':
    final_pred = xgb_test_pred
else:
    final_pred = lgb_test_pred

# ============================================================
# 7.1 后处理优化
# ============================================================
print("\n[7.1] 后处理优化...")

# 预测截断
lower_bound = np.percentile(y_tr, 1)
upper_bound = np.percentile(y_tr, 99)
final_pred = np.clip(final_pred, lower_bound, upper_bound)
print(f"  截断后范围: [{lower_bound:.4f}, {upper_bound:.4f}]")

# ============================================================
# 7.2 截面标准化 - 关键优化！
# ============================================================
print("\n[7.2] 截面标准化 (关键优化)...")
print("  对每个时间点的预测值进行标准化...")

final_pred_std = cross_sectional_standardize(
    final_pred,
    test_df['stockid'].values,
    test_df['dateid'].values,
    test_df['timeid'].values
)

print(f"  标准化前: mean={np.mean(final_pred):.4f}, std={np.std(final_pred):.4f}")
print(f"  标准化后: mean={np.mean(final_pred_std):.4f}, std={np.std(final_pred_std):.4f}")

# 使用标准化后的预测
final_pred = final_pred_std

# ============================================================
# 7.3 EMA平滑 (V9新增)
# ============================================================
print("\n[7.3] EMA平滑...")
final_pred_smooth = pd.Series(final_pred).ewm(span=3).mean().values
print(f"  平滑前 std: {np.std(final_pred):.4f}")
print(f"  平滑后 std: {np.std(final_pred_smooth):.4f}")
final_pred = final_pred_smooth

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

print(f"\n=== A股特性增强 (V9) ===")
print(f"  特征数: {len(all_features)}")
print(f"  交易所效应: ✅")
print(f"  细粒度时段: ✅")
print(f"  涨跌停核心特征: ✅ (价格位置、封板强度)")
print(f"  VPIN订单流毒性: ✅")
print(f"  样本权重优化: ✅ (时间+涨跌停)")
print(f"  截面标准化: ✅ (关键优化)")
print(f"  EMA平滑: ✅")

print(f"\n=== 模型性能对比 ===")
print(f"  Ridge:      R² = {ridge_r2:.6f} (基线)")
print(f"  LightGBM:   R² = {lgb_r2:.6f} (带样本权重)")
print(f"  XGBoost:    R² = {xgb_r2:.6f}")
if use_lstm:
    print(f"  LSTM:       R² = {lstm_r2:.6f} (深度学习)")
print(f"  简单平均:   R² = {simple_r2:.6f}")
print(f"  加权集成:   R² = {weighted_r2:.6f}")
if use_stacking:
    print(f"  Stacking:   R² = {stacking_r2:.6f}")
print(f"  最佳:       {best_method}, R² = {best_r2:.6f}")

print(f"\n=== 集成权重 ===")
for name, w in optimal_weights.items():
    print(f"  {name}: {w:.4f}")

if use_stacking:
    print(f"\n=== Stacking元学习器权重 ===")
    for name, coef in zip(val_predictions.keys(), meta_model.coef_):
        print(f"  {name}: {coef:.4f}")

print("\n" + "="*70)
print("🚀 V9 全优化版本完成!")
print("="*70)
print("\n✅ 所有优化已应用:")
print("  1. 涨跌停核心特征")
print("  2. VPIN订单流毒性")
print("  3. 样本权重优化")
print("  4. LSTM深度学习")
print("  5. Stacking集成")
print("  6. 截面标准化")
print("  7. EMA平滑")
print("\n🏆 祝竞赛取得好成绩!")
