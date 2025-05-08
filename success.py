"""
金融波动率预测系统
该系统结合传统统计方法、机器学习和深度学习技术，对股票和指数的波动率进行多窗口预测和分析。
"""

import pandas as pd
import numpy as np
import matplotlib
# 设置使用Agg非交互式后端，而不是TkAgg
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yfinance as yf
import os
import json
from typing import List, Dict, Tuple, Union, Optional, Any
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math

# 设置中文字体支持
import platform
system = platform.system()
if system == 'Windows':
    try:
        # 尝试使用微软雅黑
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    except:
        # 如果没有中文字体，就使用英文
        pass
elif system == 'Linux':
    try:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
    except:
        pass
elif system == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC']

# 设置正常显示负号
plt.rcParams['axes.unicode_minus'] = False

# 传统时间序列模型
import statsmodels.api as sm
# 引入GARCH模型所需的包
from arch import arch_model
from arch.univariate import ConstantMean, GARCH, Normal

# 机器学习模型 - 替换SVR为RandomForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 超参数优化
import optuna

#------------------------------------------------------------------------------
# 数据处理模块
#------------------------------------------------------------------------------

class FinancialDataProcessor:
    """
    金融数据处理模块，负责获取、清洗、预处理和可视化用于波动率预测的金融数据。
    """

    def __init__(self, output_dir: str = 'output'):
        """
        初始化金融数据处理器。

        参数:
            output_dir (str): 保存处理后数据和可视化结果的目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)

    def fetch_data(self,
                  symbol: str,
                  start_date: str,
                  end_date: str,
                  interval: str = '1d') -> pd.DataFrame:
        """
        从Yahoo Finance获取历史金融数据。

        参数:
            symbol (str): 股票/指数代码 (例如 "^GSPC", "000300.SS")
            start_date (str): 起始日期，格式为"YYYY-MM-DD"
            end_date (str): 结束日期，格式为"YYYY-MM-DD"
            interval (str): 数据间隔 ('1d', '1wk', '1mo')

        返回:
            pd.DataFrame: 历史价格数据
        """
        print(f"获取{symbol}从{start_date}到{end_date}的数据...")
        data = yf.download(symbol, start=start_date, end=end_date, interval=interval, auto_adjust=True)
        print(f"获取了{len(data)}条记录。")

        # 确保列名统一
        if 'Adj Close' not in data.columns and 'Close' in data.columns:
            data['Adj Close'] = data['Close']  # 如果使用auto_adjust=True，Close已经是调整后的价格

        return data

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        清洗金融数据，处理缺失值。

        参数:
            data (pd.DataFrame): 原始金融数据

        返回:
            pd.DataFrame: 清洗后的金融数据
        """
        # 检查缺失值
        missing_values = data.isnull().sum()
        print(f"清洗前的缺失值:\n{missing_values}")

        # 使用前向填充法处理缺失值（使用前一天的值）
        data_cleaned = data.ffill()

        # 如果开始部分仍有缺失值，使用后向填充
        data_cleaned = data_cleaned.bfill()

        missing_values_after = data_cleaned.isnull().sum()
        print(f"清洗后的缺失值:\n{missing_values_after}")

        return data_cleaned

    def calculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算日收益率和对数收益率。

        参数:
            data (pd.DataFrame): 清洗后的价格数据

        返回:
            pd.DataFrame: 包含日收益率和对数收益率的数据
        """
        # 保留原始数据的副本
        df = data.copy()

        # 计算简单日收益率
        df['daily_return'] = df['Adj Close'].pct_change()

        # 计算对数收益率
        df['log_return'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))

        # 删除包含NaN值的第一行
        df = df.dropna()

        return df

    def calculate_volatility(self,
                            data: pd.DataFrame,
                            windows: List[int] = [7, 14, 21, 30]) -> pd.DataFrame:
        """
        计算指定窗口的已实现波动率。

        参数:
            data (pd.DataFrame): 包含收益率的数据
            windows (List[int]): 用于计算波动率的窗口大小列表

        返回:
            pd.DataFrame: 包含波动率列的数据
        """
        df = data.copy()

        # 计算每个窗口的已实现波动率
        for window in windows:
            # 对数收益率的标准差 * sqrt(252) 用于年化
            df[f'volatility_{window}d'] = df['log_return'].rolling(window=window).std() * np.sqrt(252)

        # 删除包含NaN波动率值的行
        df = df.dropna()

        return df

    def normalize_data(self,
                      data: pd.DataFrame,
                      columns: List[str] = None,
                      method: str = 'z-score') -> Tuple[pd.DataFrame, Dict]:
        """
        使用z-score或min-max标准化数据。

        参数:
            data (pd.DataFrame): 要标准化的数据
            columns (List[str]): 要标准化的列（如果为None，则标准化所有数值列）
            method (str): 标准化方法 ('z-score' 或 'min-max')

        返回:
            Tuple[pd.DataFrame, Dict]: 标准化后的数据和标准化参数
        """
        df = data.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        norm_params = {}

        for col in columns:
            if method == 'z-score':
                mean = df[col].mean()
                std = df[col].std()
                df[f"{col}_norm"] = (df[col] - mean) / std
                norm_params[col] = {'mean': mean, 'std': std}

            elif method == 'min-max':
                min_val = df[col].min()
                max_val = df[col].max()
                df[f"{col}_norm"] = (df[col] - min_val) / (max_val - min_val)
                norm_params[col] = {'min': min_val, 'max': max_val}

        return df, norm_params

    def visualize_data(self,
                       data: pd.DataFrame,
                       symbol: str,
                       save: bool = True,
                       show: bool = False) -> None:
        """
        可视化金融数据和计算的指标。

        参数:
            data (pd.DataFrame): 处理后的数据
            symbol (str): 金融资产的代码
            save (bool): 是否保存可视化结果
            show (bool): 是否显示可视化结果
        """
        # 创建包含3个子图的图表
        fig = plt.figure(figsize=(14, 18))

        # 处理股票代码中的特殊字符
        safe_symbol = symbol.replace('^', '_')


        # 图1: 价格
        ax1 = plt.subplot(3, 1, 1)
        price_line, = ax1.plot(data.index, data['Adj Close'])
        ax1.set_title(f'{symbol} Price', fontsize=14)
        ax1.set_ylabel('Price')
        price_line.set_label('Adjusted Close')
        ax1.legend()
        ax1.grid(True)

        # 图2: 收益率
        ax2 = plt.subplot(3, 1, 2)
        daily_line, = ax2.plot(data.index, data['daily_return'], alpha=0.7)
        log_line, = ax2.plot(data.index, data['log_return'], alpha=0.7)
        ax2.set_title(f'{symbol} Returns', fontsize=14)
        ax2.set_ylabel('Return')
        daily_line.set_label('Daily Return')
        log_line.set_label('Log Return')
        ax2.legend()
        ax2.grid(True)

        # 图3: 波动率
        ax3 = plt.subplot(3, 1, 3)
        # 修复: 使用数据框的索引作为x轴，而不是尝试访问'Date'列
        volatility7_line, = ax3.plot(data.index, data['volatility_7d'], alpha=0.7)
        volatility14_line, = ax3.plot(data.index, data['volatility_14d'], alpha=0.7)
        volatility21_line, = ax3.plot(data.index, data['volatility_21d'], alpha=0.7)
        volatility30_line, = ax3.plot(data.index, data['volatility_30d'], alpha=0.7)
        ax3.set_title(f'{symbol} Volatility', fontsize=14)
        ax3.set_ylabel('Annualized Volatility')
        volatility7_line.set_label('volatility_7days')
        volatility14_line.set_label('volatility_14days')
        volatility21_line.set_label('volatility_21days')
        volatility30_line.set_label('volatility_30days')
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
        if save:
            # Replace special characters like '^' with '_'
            safe_symbol = symbol.replace('^', '_')
            file_path = os.path.join(self.output_dir, 'visualizations', f'{safe_symbol}_visualization.png')
            plt.savefig(file_path, dpi=100)
            print(f"Visualization saved to {file_path}")


    def save_data(self, data: pd.DataFrame, symbol: str) -> str:
        """
        保存处理后的数据到CSV文件。

        参数:
            data (pd.DataFrame): 要保存的数据
            symbol (str): 金融资产代码

        返回:
            str: 保存文件的路径
        """


        file_path = os.path.join(self.output_dir, 'data', f'{symbol}_processed.csv')
        data.to_csv(file_path)
        print(f"数据已保存到 {file_path}")
        return file_path

    def process_asset(self,
                     symbol: str,
                     start_date: str,
                     end_date: str,
                     windows: List[int] = [7, 14, 21, 30],
                     visualize: bool = True) -> pd.DataFrame:
        """
        处理单个金融资产的完整流程。

        参数:
            symbol (str): 股票/指数代码
            start_date (str): 起始日期
            end_date (str): 结束日期
            windows (List[int]): 波动率窗口大小列表
            visualize (bool): 是否可视化处理后的数据

        返回:
            pd.DataFrame: 处理后的数据
        """
        # 获取数据
        data = self.fetch_data(symbol, start_date, end_date)

        # 清洗数据
        data_cleaned = self.clean_data(data)

        # 计算收益率
        data_with_returns = self.calculate_returns(data_cleaned)

        # 计算波动率
        data_with_volatility = self.calculate_volatility(data_with_returns, windows)

        # 可视化
        if visualize:
            self.visualize_data(data_with_volatility, symbol, show=False)

        # 保存数据
        self.save_data(data_with_volatility, symbol)

        return data_with_volatility

#------------------------------------------------------------------------------
# 深度学习支持模块
#------------------------------------------------------------------------------

class TimeSeriesDataset(Dataset):
    """
    用于时间序列数据的PyTorch数据集类。
    """

    def __init__(self,
                 features: np.ndarray,
                 targets: np.ndarray,
                 seq_length: int):
        """
        初始化时间序列数据集。

        参数:
            features (np.ndarray): 特征数据，形状为 [n_samples, n_features]
            targets (np.ndarray): 目标数据，形状为 [n_samples]
            seq_length (int): 序列长度（时间步数）
        """
        self.features = features
        self.targets = targets
        self.seq_length = seq_length

    def __len__(self) -> int:
        """
        返回数据集中的样本数。

        返回:
            int: 样本数
        """
        return len(self.features) - self.seq_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个样本。

        参数:
            idx (int): 样本索引

        返回:
            Tuple[torch.Tensor, torch.Tensor]: 特征序列和目标值
        """
        # 提取序列
        x = self.features[idx:idx + self.seq_length]

        # 目标是序列之后的下一个值
        y = self.targets[idx + self.seq_length]

        return torch.FloatTensor(x), torch.FloatTensor([y])


class LSTMModel(nn.Module):
    """
    用于时间序列预测的LSTM模型。
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 output_dim: int,
                 dropout: float = 0.2):
        """
        初始化LSTM模型。

        参数:
            input_dim (int): 输入特征维度
            hidden_dim (int): LSTM隐藏层维度
            num_layers (int): LSTM层数
            output_dim (int): 输出维度
            dropout (float): Dropout率
        """
        super(LSTMModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM层
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 全连接输出层
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
            x (torch.Tensor): 输入数据，形状为 [batch_size, seq_length, input_dim]

        返回:
            torch.Tensor: 预测结果，形状为 [batch_size, output_dim]
        """
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)

        # 只使用最后一个时间步的输出
        last_time_step = lstm_out[:, -1, :]

        # 应用dropout
        out = self.dropout(last_time_step)

        # 通过全连接层
        out = self.fc(out)

        return out


class PositionalEncoding(nn.Module):
    """
    Transformer模型的位置编码。
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        """
        初始化位置编码。

        参数:
            d_model (int): 嵌入维度
            max_len (int): 最大序列长度
        """
        super(PositionalEncoding, self).__init__()

        # 创建一个长度为max_len，维度为d_model的0矩阵
        pe = torch.zeros(max_len, d_model)

        # 创建一个形状为[max_len, 1]的位置矩阵
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 计算分母部分，使用指数函数
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 计算位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加批次维度 [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # 注册为非训练参数
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，添加位置编码。

        参数:
            x (torch.Tensor): 输入嵌入，形状为 [batch_size, seq_length, d_model]

        返回:
            torch.Tensor: 添加位置编码后的嵌入
        """
        return x + self.pe[:, :x.size(1), :]


class TransformerModel(nn.Module):
    """
    用于时间序列预测的Transformer模型。
    """

    def __init__(self,
                 input_dim: int,
                 d_model: int,
                 nhead: int,
                 num_encoder_layers: int,
                 dim_feedforward: int,
                 dropout: float,
                 output_dim: int):
        """
        初始化Transformer模型。

        参数:
            input_dim (int): 输入特征维度
            d_model (int): 模型维度
            nhead (int): 多头注意力的头数
            num_encoder_layers (int): 编码器层数
            dim_feedforward (int): 前馈网络维度
            dropout (float): Dropout率
            output_dim (int): 输出维度
        """
        super(TransformerModel, self).__init__()

        # 特征投影层
        self.input_projection = nn.Linear(input_dim, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer编码器层
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Transformer编码器
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_encoder_layers
        )

        # 输出层
        self.output_layer = nn.Linear(d_model, output_dim)

        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
            src (torch.Tensor): 输入数据，形状为 [batch_size, seq_length, input_dim]

        返回:
            torch.Tensor: 预测结果，形状为 [batch_size, output_dim]
        """
        # 输入投影到d_model维度
        src = self.input_projection(src)

        # 添加位置编码
        src = self.pos_encoder(src)

        # 通过Transformer编码器
        output = self.transformer_encoder(src)

        # 取最后一个时间步的输出
        output = output[:, -1, :]

        # 应用dropout
        output = self.dropout(output)

        # 通过输出层
        output = self.output_layer(output)

        return output


def setup_device() -> torch.device:
    """
    设置计算设备（CPU或GPU）。

    返回:
        torch.device: 计算设备
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("GPU不可用，使用CPU")

    return device

#------------------------------------------------------------------------------
# 模型构建与评估模块
#------------------------------------------------------------------------------

class VolatilityPredictionModels:
    """
    波动率预测模型的构建、训练和评估。
    """

    def __init__(self,
                 output_dir: str = 'output',
                 device: torch.device = None):
        """
        初始化波动率预测模型类。

        参数:
            output_dir (str): 输出目录
            device (torch.device): 计算设备
        """
        self.output_dir = output_dir
        os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'evaluations'), exist_ok=True)

        # 设置计算设备
        self.device = device if device is not None else setup_device()

    def create_features(self,
                       data: pd.DataFrame,
                       target_col: str,
                       lag_periods: List[int] = [1, 2, 3, 5, 7, 14, 21]) -> pd.DataFrame:
        """
        为时间序列数据创建特征，包括滞后特征和技术指标。

        参数:
            data (pd.DataFrame): 输入数据
            target_col (str): 目标列名
            lag_periods (List[int]): 滞后期列表

        返回:
            pd.DataFrame: 包含特征的数据
        """
        df = data.copy()

        # 创建滞后特征
        for lag in lag_periods:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)

        # 创建移动平均特征
        for window in [5, 10, 20, 30]:
            df[f'ma_{window}'] = df['Adj Close'].rolling(window=window).mean()
            df[f'{target_col}_ma_{window}'] = df[target_col].rolling(window=window).mean()

        # 创建移动标准差特征
        for window in [5, 10, 20, 30]:
            df[f'{target_col}_std_{window}'] = df[target_col].rolling(window=window).std()

        # 相对强弱指标 (RSI)
        delta = df['Adj Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()

        rs = avg_gain / avg_loss
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # MACD (移动平均线收敛/发散)
        exp12 = df['Adj Close'].ewm(span=12, adjust=False).mean()
        exp26 = df['Adj Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp12 - exp26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # 布林带
        for window in [20]:
            df[f'bb_middle_{window}'] = df['Adj Close'].rolling(window=window).mean()
            df[f'bb_std_{window}'] = df['Adj Close'].rolling(window=window).std()
            df[f'bb_upper_{window}'] = df[f'bb_middle_{window}'] + 2 * df[f'bb_std_{window}']
            df[f'bb_lower_{window}'] = df[f'bb_middle_{window}'] - 2 * df[f'bb_std_{window}']
            df[f'bb_width_{window}'] = (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']) / df[f'bb_middle_{window}']

        # 交易量特征（如果有交易量数据）
        if 'Volume' in df.columns:
            df['volume_change'] = df['Volume'].pct_change()
            df['volume_ma_5'] = df['Volume'].rolling(window=5).mean()
            df['volume_ma_10'] = df['Volume'].rolling(window=10).mean()
            df['volume_ma_ratio'] = df['volume_ma_5'] / df['volume_ma_10']

        # 删除包含NaN值的行
        df = df.dropna()

        # 重置索引
        df = df.reset_index()



        return df

    def train_test_split(self,
                        data: pd.DataFrame,
                        target_col: str,
                        test_size: float = 0.2,
                        feature_cols: List[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        将数据分割为训练集和测试集。

        参数:
            data (pd.DataFrame): 输入数据
            target_col (str): 目标列名
            test_size (float): 测试集比例
            feature_cols (List[str]): 特征列名列表（如果为None，则使用所有数值列）

        返回:
            Tuple: X_train, X_test, y_train, y_test, feature_columns
        """
        if feature_cols is None:
            # 排除目标列和非数值列
            feature_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in feature_cols if col != target_col and col != 'Date']

        # 按时间顺序分割
        train_size = int(len(data) * (1 - test_size))
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]

        # 提取特征和目标
        X_train = train_data[feature_cols].values
        y_train = train_data[target_col].values

        X_test = test_data[feature_cols].values
        y_test = test_data[target_col].values

        print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
        print(f"特征数量: {len(feature_cols)}")

        return X_train, X_test, y_train, y_test, feature_cols

    def scale_data(self,
                  X_train: np.ndarray,
                  X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
        """
        标准化特征数据。

        参数:
            X_train (np.ndarray): 训练集特征
            X_test (np.ndarray): 测试集特征

        返回:
            Tuple: 标准化后的特征和缩放器
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, scaler

    def evaluate_model(self,
                      y_true: np.ndarray,
                      y_pred: np.ndarray,
                      model_name: str) -> Dict[str, float]:
        """
        评估模型性能。

        参数:
            y_true (np.ndarray): 真实值
            y_pred (np.ndarray): 预测值
            model_name (str): 模型名称

        返回:
            Dict[str, float]: 包含评估指标的字典
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        metrics = {
            'model': model_name,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }

        print(f"{model_name} Evaluation Results:")
        print(f"  MAE: {mae:.6f}")
        print(f"  MSE: {mse:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  R²: {r2:.6f}")

        return metrics

    def save_evaluation(self,
                       metrics: Dict[str, float],
                       symbol: str,
                       target_col: str) -> None:
        """
        保存评估结果。

        参数:
            metrics (Dict[str, float]): 评估指标
            symbol (str): 金融资产代码
            target_col (str): 目标列名
        """
        file_path = os.path.join(self.output_dir, 'evaluations',
                                f'{symbol}_{metrics["model"]}_{target_col}_eval.json')

        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        print(f"Evaluation results saved to {file_path}")

    def save_predictions(self,
                        dates: pd.DatetimeIndex,
                        y_true: np.ndarray,
                        y_pred: np.ndarray,
                        model_name: str,
                        symbol: str,
                        target_col: str) -> str:
        """
        保存预测结果。

        参数:
            dates (pd.DatetimeIndex): 日期索引
            y_true (np.ndarray): 真实值
            y_pred (np.ndarray): 预测值
            model_name (str): 模型名称
            symbol (str): 金融资产代码
            target_col (str): 目标列名

        返回:
            str: 保存文件的路径
        """
        # 创建包含日期、真实值和预测值的DataFrame
        predictions_df = pd.DataFrame({
            'date': dates,
            'actual': y_true,
            'predicted': y_pred
        })

        # 保存到CSV
        file_path = os.path.join(self.output_dir, 'predictions',
                               f'{symbol}_{model_name}_{target_col}_predictions.csv')
        predictions_df.to_csv(file_path, index=False)

        print(f"Prediction results saved to {file_path}")
        return file_path

    def visualize_predictions(self,
                            dates: pd.DatetimeIndex,
                            y_true: np.ndarray,
                            predictions: Dict[str, np.ndarray],
                            symbol: str,
                            target_col: str,
                            save: bool = True,
                            show: bool = False) -> None:
        """
        可视化预测结果。

        参数:
            dates (pd.DatetimeIndex): 日期索引
            y_true (np.ndarray): 真实值
            predictions (Dict[str, np.ndarray]): 不同模型的预测值
            symbol (str): 金融资产代码
            target_col (str): 目标列名
            save (bool): 是否保存可视化结果
            show (bool): 是否显示可视化结果
        """
        fig = plt.figure(figsize=(14, 8))

        # 绘制真实值
        actual_line, = plt.plot(dates, y_true, linewidth=2, color='black')
        actual_line.set_label('Actual')

        # 绘制预测值
        colors = ['blue', 'red', 'green', 'purple']
        for i, (model_name, y_pred) in enumerate(predictions.items()):
            pred_line, = plt.plot(dates, y_pred, linewidth=1.5, color=colors[i % len(colors)])
            pred_line.set_label(f'{model_name} Prediction')

        # 添加标题和标签
        plt.title(f'{symbol} {target_col} Prediction Comparison', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel(target_col, fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 保存图表
        if save:
            file_path = os.path.join(self.output_dir, 'visualizations',
                                   f'{symbol}_{target_col}_predictions_comparison.png')
            plt.savefig(file_path, dpi=100)
            print(f"Prediction comparison chart saved to {file_path}")

        # 关闭图表，避免内存泄漏
        plt.close(fig)

    def visualize_model_comparison(self,
                                   evaluations: List[Dict[str, float]],
                                   symbol: str,
                                   target_col: str,
                                   metric: str = 'rmse',
                                   save: bool = True,
                                   show: bool = False) -> None:
        """
        可视化不同模型性能比较。

        参数:
            evaluations (List[Dict[str, float]]): 评估指标列表
            symbol (str): 金融资产代码
            target_col (str): 目标列名
            metric (str): 用于比较的指标
            save (bool): 是否保存可视化结果
            show (bool): 是否显示可视化结果
        """
        models = [eval_dict['model'] for eval_dict in evaluations]
        metric_values = [eval_dict[metric] for eval_dict in evaluations]

        # 为RMSE和MAE设置较小值更好，为R²设置较大值更好
        if metric in ['rmse', 'mse', 'mae']:
            best_idx = np.argmin(metric_values)
            colors = ['lightblue' if i != best_idx else 'green' for i in range(len(models))]
            title_prefix = "Lower is better"
        else:  # R²
            best_idx = np.argmax(metric_values)
            colors = ['lightblue' if i != best_idx else 'green' for i in range(len(models))]
            title_prefix = "Higher is better"

        fig = plt.figure(figsize=(12, 6))

        # 决定是否使用对数刻度
        use_log_scale = False
        if metric in ['rmse', 'mse', 'mae']:
            # 检查值的范围，如果最大值和最小值比例超过100，使用对数刻度
            max_val = max(metric_values)
            min_val = min(metric_values)
            if max_val / (min_val + 1e-10) > 100:  # 防止除以0
                use_log_scale = True

        # 创建条形图
        bars = plt.bar(models, metric_values, color=colors)

        # 在每个条形上方添加数值，使用科学计数法和更高精度
        for i, bar in enumerate(bars):
            height = bar.get_height()

            # 根据数值大小选择合适的格式化方式
            if metric_values[i] < 0.001:
                # 非常小的值使用科学计数法
                value_text = f'{metric_values[i]:.10e}'
            else:
                # 较大的值使用10位小数
                value_text = f'{metric_values[i]:.10f}'

            # 计算文本位置
            if use_log_scale:
                # 对数刻度下，文本位置需要特别处理
                text_y = height * 1.1  # 在条形上方10%位置
            else:
                # 线性刻度下，固定在条形上方一定距离
                text_y = height + (max(metric_values) * 0.02)  # 最大值的2%

            plt.text(bar.get_x() + bar.get_width() / 2., text_y,
                     value_text,
                     ha='center', va='bottom', rotation=45, fontsize=9)  # 旋转45度提高可读性

        # 设置Y轴刻度
        if use_log_scale:
            plt.yscale('log')  # 使用对数刻度
            plt.title(f'{symbol} {target_col} {metric.upper()} Comparison (Log Scale, {title_prefix})', fontsize=14)
        else:
            # 确保Y轴有合适的最小值，让所有条形可见
            plt.ylim(0, max(metric_values) * 1.2)  # 上限设为最大值的1.2倍
            plt.title(f'{symbol} {target_col} {metric.upper()} Comparison ({title_prefix})', fontsize=14)

        plt.ylabel(f'{metric.upper()}', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        # 保存图表
        if save:
            file_path = os.path.join(self.output_dir, 'visualizations',
                                     f'{symbol}_{target_col}_{metric}_comparison.png')
            plt.savefig(file_path, dpi=100)
            print(f"Model comparison chart saved to {file_path}")

        # 关闭图表，避免内存泄漏
        plt.close(fig)

    #--------------------------------------------------------------------------
    # GARCH模型 (替换原来的ARIMA模型)
    #--------------------------------------------------------------------------

    def train_auto_garch(self,
                        train_data: pd.Series,
                        max_p: int = 2,
                        max_q: int = 2) -> Dict[str, Any]:
        """
        使用网格搜索找到最佳GARCH参数并训练模型。

        参数:
            train_data (pd.Series): 训练数据（收益率序列）
            max_p (int): 最大p参数
            max_q (int): 最大q参数

        返回:
            Dict[str, Any]: 包含模型和参数的字典
        """
        print("训练GARCH模型...")

        best_aic = float('inf')
        best_params = None
        best_model = None

        # 网格搜索最佳参数
        for p in range(1, max_p + 1):
            for q in range(1, max_q + 1):
                try:
                    # 使用常数均值模型 + GARCH波动率模型
                    model = arch_model(
                        train_data,
                        mean='Constant',
                        vol='GARCH',
                        p=p,
                        q=q,
                        rescale=False
                    )

                    # 拟合模型
                    fitted_model = model.fit(disp='off')

                    # 获取AIC
                    aic = fitted_model.aic

                    # 更新最佳模型
                    if aic < best_aic:
                        best_aic = aic
                        best_params = (p, q)
                        best_model = fitted_model

                    print(f"GARCH({p},{q}) AIC: {aic:.4f}")

                except Exception as e:
                    print(f"GARCH({p},{q}) 拟合失败: {str(e)}")
                    continue

        if best_model is None:
            print("所有GARCH模型拟合失败，使用默认GARCH(1,1)")
            model = arch_model(train_data, mean='Constant', vol='GARCH', p=1, q=1)
            best_model = model.fit(disp='off')
            best_params = (1, 1)

        print(f"最佳GARCH参数: GARCH{best_params}")
        print(f"最佳GARCH模型AIC: {best_aic:.4f}")

        return {
            'model': best_model,
            'order': best_params
        }

    def predict_garch(self,
                     model_dict: Dict[str, Any],
                     steps: int,
                     last_obs: pd.Series = None,
                     annualize: bool = True) -> np.ndarray:
        """
        使用训练好的GARCH模型进行预测。

        参数:
            model_dict (Dict[str, Any]): 包含模型和参数的字典
            steps (int): 预测步数
            last_obs (pd.Series): 最后观测的数据，用于滚动预测
            annualize (bool): 是否将波动率年化

        返回:
            np.ndarray: 预测的波动率
        """
        # 获取模型
        model = model_dict['model']

        # 执行预测
        forecast = model.forecast(horizon=steps, reindex=False)

        # 提取波动率预测（标准差）
        volatility_forecast = np.sqrt(forecast.variance.values[-1])

        # 年化波动率（如果需要）
        if annualize:
            # 假设使用的是日收益率数据，年化因子为sqrt(252)
            volatility_forecast = volatility_forecast * np.sqrt(252)

        return volatility_forecast

    def optimize_garch_with_optuna(self,
                                 train_data: pd.Series,
                                 n_trials: int = 20) -> Dict[str, Any]:
        """
        使用Optuna优化GARCH模型参数。

        参数:
            train_data (pd.Series): 训练数据（收益率序列）
            n_trials (int): Optuna试验次数

        返回:
            Dict[str, Any]: 最佳参数和模型
        """
        print("使用Optuna优化GARCH模型参数...")

        def objective(trial):
            # 定义超参数搜索空间
            p = trial.suggest_int('p', 1, 3)
            q = trial.suggest_int('q', 1, 3)

            # 可以添加更多的参数
            # power = trial.suggest_float('power', 1.0, 2.0)  # 用于GJR-GARCH的非对称性

            try:
                # 创建并拟合模型
                model = arch_model(
                    train_data,
                    mean='Constant',
                    vol='GARCH',
                    p=p,
                    q=q,
                    rescale=True
                )

                fitted_model = model.fit(disp='off')

                # 返回AIC作为优化目标
                return fitted_model.aic

            except Exception as e:
                # 如果拟合失败，返回一个很大的数
                print(f"Error fitting GARCH({p},{q}): {str(e)}")
                return float('inf')

        # 创建Optuna研究
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        print(f"最佳GARCH参数: {best_params}")

        # 使用最佳参数拟合最终模型
        try:
            final_model = arch_model(
                train_data,
                mean='Constant',
                vol='GARCH',
                p=best_params['p'],
                q=best_params['q'],
                rescale=False
            )

            fitted_model = final_model.fit(disp='off')

            print(f"最终模型AIC: {fitted_model.aic:.4f}")

            return {
                'model': fitted_model,
                'order': (best_params['p'], best_params['q'])
            }

        except Exception as e:
            print(f"拟合最终模型时出错: {str(e)}")
            # 回退到简单的GARCH(1,1)模型
            model = arch_model(train_data, mean='Constant', vol='GARCH', p=1, q=1)
            fitted_model = model.fit(disp='off')

            return {
                'model': fitted_model,
                'order': (1, 1)
            }

    def rolling_garch_forecast(self,
                               returns: pd.Series,
                               train_size: int,
                               forecast_horizon: int = 1,
                               p: int = 1,
                               q: int = 1,
                               annualize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Using rolling window method with GARCH model to predict volatility.
        """
        # Number of predictions
        n_predictions = len(returns) - train_size

        # Store prediction results
        predictions = np.zeros(n_predictions)
        actuals = np.zeros(n_predictions)

        # For each time point
        for i in range(n_predictions):
            # Use data up to this point for training
            train_returns = returns.iloc[i:i + train_size]

            # Scale the returns to improve model fit (convert to percentage)
            scaled_returns = train_returns * 100

            try:
                # Create and fit GARCH model
                model = arch_model(scaled_returns, mean='Constant', vol='GARCH', p=p, q=q, rescale=False)
                fitted_model = model.fit(disp='off', show_warning=False)

                # Forecast next time point volatility
                forecast = fitted_model.forecast(horizon=forecast_horizon)
                # Extract predicted conditional volatility (standard deviation) and convert back to original scale
                volatility = np.sqrt(forecast.variance.iloc[-1, 0]) / 100

                # Annualize volatility if needed
                if annualize:
                    volatility = volatility * np.sqrt(252)

                predictions[i] = volatility

            except Exception as e:
                print(f"Prediction for time point {i} failed: {str(e)}")
                # If prediction fails, use historical volatility as fallback
                hist_vol = train_returns.std()
                if annualize:
                    hist_vol = hist_vol * np.sqrt(252)
                predictions[i] = hist_vol

            # Store actual volatility value
            actuals[i] = returns.iloc[i + train_size]

            # Show progress
            if i % 10 == 0:
                print(f"Completed {i}/{n_predictions} predictions")

        return actuals, predictions

    #--------------------------------------------------------------------------
    # 随机森林模型 (替换SVR模型)
    #--------------------------------------------------------------------------

    def optimize_random_forest(self,
                           X_train: np.ndarray,
                           y_train: np.ndarray) -> Dict[str, Any]:
        """
        优化随机森林模型参数。

        参数:
            X_train (np.ndarray): 训练集特征
            y_train (np.ndarray): 训练集目标

        返回:
            Dict[str, Any]: 包含最佳参数的字典和最佳模型
        """
        print("优化随机森林模型参数...")

        # 参数网格
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['log2', 'sqrt']
        }

        # 创建网格搜索
        tscv = TimeSeriesSplit(n_splits=5)
        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            verbose=1,
            n_jobs=-1
        )

        # 执行网格搜索
        grid_search.fit(X_train, y_train)

        print(f"最佳随机森林参数: {grid_search.best_params_}")
        print(f"最佳随机森林得分: {-grid_search.best_score_:.6f} (MSE)")

        # 返回最佳参数和最佳模型
        return grid_search.best_params_, grid_search.best_estimator_

    def train_random_forest(self,
                        X_train: np.ndarray,
                        y_train: np.ndarray,
                        params: Dict[str, Any] = None) -> RandomForestRegressor:
        """
        训练随机森林模型。

        参数:
            X_train (np.ndarray): 训练集特征
            y_train (np.ndarray): 训练集目标
            params (Dict[str, Any]): 模型参数

        返回:
            RandomForestRegressor: 训练好的随机森林模型
        """
        print("训练随机森林模型...")

        if params is None:
            # 默认参数
            params = {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt'
            }

        # 创建随机森林模型
        model = RandomForestRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            max_features=params['max_features'],
            random_state=42
        )

        # 训练模型
        model.fit(X_train, y_train)

        print("随机森林模型训练完成")

        return model

    def predict_random_forest(self,
                          model: RandomForestRegressor,
                          X_test: np.ndarray) -> np.ndarray:
        """
        使用训练好的随机森林模型进行预测。

        参数:
            model (RandomForestRegressor): 训练好的随机森林模型
            X_test (np.ndarray): 测试集特征

        返回:
            np.ndarray: 预测结果
        """
        # 进行预测
        predictions = model.predict(X_test)

        return predictions

    def feature_importance_analysis(self,
                                 model: RandomForestRegressor,
                                 feature_names: List[str],
                                 symbol: str,
                                 target_col: str) -> None:
        """
        分析随机森林模型的特征重要性并可视化。

        参数:
            model (RandomForestRegressor): 训练好的随机森林模型
            feature_names (List[str]): 特征名称列表
            symbol (str): 金融资产代码
            target_col (str): 目标列名
        """
        # 获取特征重要性
        importances = model.feature_importances_

        # 创建包含特征名称和重要性的DataFrame
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })

        # 按重要性降序排序
        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

        # 取前20个最重要的特征进行可视化
        top_n = min(20, len(feature_importance_df))
        top_features = feature_importance_df[:top_n]

        # 创建水平条形图
        fig, ax = plt.figure(figsize=(10, 8)), plt.axes()

        # 绘制条形图
        bars = ax.barh(top_features['Feature'], top_features['Importance'], color='skyblue')

        # 设置标题和标签
        ax.set_title(f'Top {top_n} Feature Importances for {symbol} {target_col}', fontsize=14)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)

        # 在条形上添加数值
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.002, bar.get_y() + bar.get_height()/2,
                   f'{width:.4f}', ha='left', va='center')

        # 保存图表
        file_path = os.path.join(self.output_dir, 'visualizations',
                               f'{symbol}_{target_col}_feature_importance.png')
        plt.tight_layout()
        plt.savefig(file_path, dpi=100)
        plt.close()

        print(f"Feature importance visualization saved to {file_path}")

        # 保存特征重要性数据
        csv_path = os.path.join(self.output_dir, 'evaluations',
                              f'{symbol}_{target_col}_feature_importance.csv')
        feature_importance_df.to_csv(csv_path, index=False)
        print(f"Feature importance data saved to {csv_path}")

    #--------------------------------------------------------------------------
    # LSTM模型
    #--------------------------------------------------------------------------

    def optimize_lstm_hyperparams(self,
                                X_train: np.ndarray,
                                y_train: np.ndarray,
                                X_val: np.ndarray,
                                y_val: np.ndarray,
                                n_trials: int = 20,
                                seq_length: int = 20) -> Dict[str, Any]:
        """
        使用Optuna优化LSTM模型超参数。

        参数:
            X_train (np.ndarray): 训练集特征
            y_train (np.ndarray): 训练集目标
            X_val (np.ndarray): 验证集特征
            y_val (np.ndarray): 验证集目标
            n_trials (int): Optuna试验次数
            seq_length (int): 序列长度

        返回:
            Dict[str, Any]: 最佳超参数
        """
        print("优化LSTM模型超参数...")

        # 准备数据集
        train_dataset = TimeSeriesDataset(X_train, y_train, seq_length)
        val_dataset = TimeSeriesDataset(X_val, y_val, seq_length)

        def objective(trial):
            # 定义超参数搜索空间
            hidden_dim = trial.suggest_int('hidden_dim', 32, 256)
            num_layers = trial.suggest_int('num_layers', 1, 3)
            dropout = trial.suggest_float('dropout', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

            # 创建数据加载器
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            # 创建模型
            input_dim = X_train.shape[1]
            model = LSTMModel(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                output_dim=1,
                dropout=dropout
            ).to(self.device)

            # 定义损失函数和优化器
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # 训练过程
            n_epochs = 50
            best_val_loss = float('inf')
            patience = 10
            counter = 0

            for epoch in range(n_epochs):
                # 训练模式
                model.train()
                train_loss = 0.0

                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                    # 前向传播
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)

                    # 反向传播和优化
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                # 验证模式
                model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)

                        val_loss += loss.item()

                # 早停
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    counter = 0
                else:
                    counter += 1

                if counter >= patience:
                    break

            return best_val_loss

        # 创建Optuna研究
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        print(f"最佳LSTM参数: {best_params}")
        print(f"最佳LSTM损失: {study.best_value:.6f}")

        return best_params

    def train_lstm(self,
                  X_train: np.ndarray,
                  y_train: np.ndarray,
                  X_val: np.ndarray,
                  y_val: np.ndarray,
                  params: Dict[str, Any] = None,
                  seq_length: int = 20,
                  epochs: int = 100) -> Tuple[LSTMModel, Dict[str, Any]]:
        """
        训练LSTM模型。

        参数:
            X_train (np.ndarray): 训练集特征
            y_train (np.ndarray): 训练集目标
            X_val (np.ndarray): 验证集特征
            y_val (np.ndarray): 验证集目标
            params (Dict[str, Any]): 模型参数
            seq_length (int): 序列长度
            epochs (int): 最大训练轮数

        返回:
            Tuple[LSTMModel, Dict[str, Any]]: 训练好的模型和训练历史
        """
        print("Training LSTM model...")

        if params is None:
            # 默认参数
            params = {
                'hidden_dim': 128,
                'num_layers': 2,
                'dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 64
            }

        # 准备数据集
        train_dataset = TimeSeriesDataset(X_train, y_train, seq_length)
        val_dataset = TimeSeriesDataset(X_val, y_val, seq_length)

        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'])

        # 创建模型
        input_dim = X_train.shape[1]
        model = LSTMModel(
            input_dim=input_dim,
            hidden_dim=params['hidden_dim'],
            num_layers=params['num_layers'],
            output_dim=1,
            dropout=params['dropout']
        ).to(self.device)

        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

        # 定义早停
        patience = 20
        counter = 0
        best_val_loss = float('inf')
        best_model_state = None

        # 训练历史
        history = {
            'train_loss': [],
            'val_loss': []
        }

        # 训练循环
        for epoch in range(epochs):
            # 训练模式
            model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                # 前向传播
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # 计算平均训练损失
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)

            # 验证模式
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)

                    val_loss += loss.item()

            # 计算平均验证损失
            avg_val_loss = val_loss / len(val_loader)
            history['val_loss'].append(avg_val_loss)

            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')

            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
                best_model_state = model.state_dict().copy()
            else:
                counter += 1

            if counter >= patience:
                print(f'Early stopping triggered at epoch {epoch+1}')
                break

        # 恢复最佳模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        print(f"LSTM training completed, best validation loss: {best_val_loss:.6f}")

        return model, history

    def predict_lstm(self,
                    model: LSTMModel,
                    X_test: np.ndarray,
                    seq_length: int = 20) -> np.ndarray:
        """
        使用训练好的LSTM模型进行预测。

        参数:
            model (LSTMModel): 训练好的LSTM模型
            X_test (np.ndarray): 测试集特征
            seq_length (int): 序列长度

        返回:
            np.ndarray: 预测结果
        """
        # 创建数据集和加载器
        test_dataset = torch.FloatTensor(X_test)

        # 设置为评估模式
        model.eval()

        predictions = []

        # 对每个时间点进行预测
        with torch.no_grad():
            for i in range(len(X_test) - seq_length):
                # 准备序列
                sequence = test_dataset[i:i+seq_length].unsqueeze(0).to(self.device)

                # 预测
                output = model(sequence)
                predictions.append(output.item())

        return np.array(predictions)

    def save_lstm_model(self,
                       model: LSTMModel,
                       params: Dict[str, Any],
                       history: Dict[str, List[float]],
                       symbol: str,
                       target_col: str) -> str:
        """
        保存训练好的LSTM模型。

        参数:
            model (LSTMModel): 训练好的LSTM模型
            params (Dict[str, Any]): 模型参数
            history (Dict[str, List[float]]): 训练历史
            symbol (str): 金融资产代码
            target_col (str): 目标列名

        返回:
            str: 保存路径
        """
        # 创建保存目录
        save_dir = os.path.join(self.output_dir, 'models', f'{symbol}_lstm_{target_col}')
        os.makedirs(save_dir, exist_ok=True)

        # 保存模型状态
        model_path = os.path.join(save_dir, 'model.pth')
        torch.save(model.state_dict(), model_path)

        # 保存参数和历史
        info = {
            'params': params,
            'history': history
        }
        info_path = os.path.join(save_dir, 'info.json')
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)

        print(f"LSTM模型已保存到 {save_dir}")
        return save_dir

    #--------------------------------------------------------------------------
    # Transformer模型
    #--------------------------------------------------------------------------

    def optimize_transformer_hyperparams(self,
                                       X_train: np.ndarray,
                                       y_train: np.ndarray,
                                       X_val: np.ndarray,
                                       y_val: np.ndarray,
                                       n_trials: int = 20,
                                       seq_length: int = 20) -> Dict[str, Any]:
        """
        使用Optuna优化Transformer模型超参数。

        参数:
            X_train (np.ndarray): 训练集特征
            y_train (np.ndarray): 训练集目标
            X_val (np.ndarray): 验证集特征
            y_val (np.ndarray): 验证集目标
            n_trials (int): Optuna试验次数
            seq_length (int): 序列长度

        返回:
            Dict[str, Any]: 最佳超参数
        """
        print("优化Transformer模型超参数...")

        # 准备数据集
        train_dataset = TimeSeriesDataset(X_train, y_train, seq_length)
        val_dataset = TimeSeriesDataset(X_val, y_val, seq_length)

        def objective(trial):
            # 定义超参数搜索空间
            d_model = trial.suggest_categorical('d_model', [32, 64, 128, 256])
            nhead = trial.suggest_categorical('nhead', [2, 4, 8,])
            num_encoder_layers = trial.suggest_int('num_encoder_layers', 1, 6)
            dim_feedforward = trial.suggest_categorical('dim_feedforward', [128, 256, 512])
            dropout = trial.suggest_float('dropout', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

            # 确保nhead能整除d_model
            while d_model % nhead != 0:
                nhead = trial.suggest_categorical('nhead', [2, 4, 8])

            # 创建数据加载器
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            # 创建模型
            input_dim = X_train.shape[1]
            model = TransformerModel(
                input_dim=input_dim,
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                output_dim=1
            ).to(self.device)

            # 定义损失函数和优化器
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # 训练过程
            n_epochs = 50
            best_val_loss = float('inf')
            patience = 10
            counter = 0

            for epoch in range(n_epochs):
                # 训练模式
                model.train()
                train_loss = 0.0

                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                    # 前向传播
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)

                    # 反向传播和优化
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                # 验证模式
                model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)

                        val_loss += loss.item()

                # 早停
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    counter = 0
                else:
                    counter += 1

                if counter >= patience:
                    break

            return best_val_loss

        # 创建Optuna研究
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        print(f"最佳Transformer参数: {best_params}")
        print(f"最佳Transformer损失: {study.best_value:.6f}")

        return best_params

    def train_transformer(self,
                        X_train: np.ndarray,
                        y_train: np.ndarray,
                        X_val: np.ndarray,
                        y_val: np.ndarray,
                        params: Dict[str, Any] = None,
                        seq_length: int = 20,
                        epochs: int = 100) -> Tuple[TransformerModel, Dict[str, Any]]:
        """
        训练Transformer模型。

        参数:
            X_train (np.ndarray): 训练集特征
            y_train (np.ndarray): 训练集目标
            X_val (np.ndarray): 验证集特征
            y_val (np.ndarray): 验证集目标
            params (Dict[str, Any]): 模型参数
            seq_length (int): 序列长度
            epochs (int): 最大训练轮数

        返回:
            Tuple[TransformerModel, Dict[str, Any]]: 训练好的模型和训练历史
        """
        print("Training Transformer model...")

        if params is None:
            # 默认参数
            params = {
                'd_model': 128,
                'nhead': 4,
                'num_encoder_layers': 2,
                'dim_feedforward': 512,
                'dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32
            }

        # 准备数据集
        train_dataset = TimeSeriesDataset(X_train, y_train, seq_length)
        val_dataset = TimeSeriesDataset(X_val, y_val, seq_length)

        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'])

        # 创建模型
        input_dim = X_train.shape[1]
        model = TransformerModel(
            input_dim=input_dim,
            d_model=params['d_model'],
            nhead=params['nhead'],
            num_encoder_layers=params['num_encoder_layers'],
            dim_feedforward=params['dim_feedforward'],
            dropout=params['dropout'],
            output_dim=1
        ).to(self.device)

        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

        # 定义早停
        patience = 20
        counter = 0
        best_val_loss = float('inf')
        best_model_state = None

        # 训练历史
        history = {
            'train_loss': [],
            'val_loss': []
        }

        # 训练循环
        for epoch in range(epochs):
            # 训练模式
            model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                # 前向传播
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # 计算平均训练损失
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)

            # 验证模式
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)

                    val_loss += loss.item()

            # 计算平均验证损失
            avg_val_loss = val_loss / len(val_loader)
            history['val_loss'].append(avg_val_loss)

            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')

            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
                best_model_state = model.state_dict().copy()
            else:
                counter += 1

            if counter >= patience:
                print(f'Early stopping triggered at epoch {epoch+1}')
                break

        # 恢复最佳模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        print(f"Transformer training completed, best validation loss: {best_val_loss:.6f}")

        return model, history

    def predict_transformer(self,
                          model: TransformerModel,
                          X_test: np.ndarray,
                          seq_length: int = 20) -> np.ndarray:
        """
        使用训练好的Transformer模型进行预测。

        参数:
            model (TransformerModel): 训练好的Transformer模型
            X_test (np.ndarray): 测试集特征
            seq_length (int): 序列长度

        返回:
            np.ndarray: 预测结果
        """
        # 创建数据集和加载器
        test_dataset = torch.FloatTensor(X_test)

        # 设置为评估模式
        model.eval()

        predictions = []

        # 对每个时间点进行预测
        with torch.no_grad():
            for i in range(len(X_test) - seq_length):
                # 准备序列
                sequence = test_dataset[i:i+seq_length].unsqueeze(0).to(self.device)

                # 预测
                output = model(sequence)
                predictions.append(output.item())

        return np.array(predictions)

    def save_transformer_model(self,
                             model: TransformerModel,
                             params: Dict[str, Any],
                             history: Dict[str, List[float]],
                             symbol: str,
                             target_col: str) -> str:
        """
        保存训练好的Transformer模型。

        参数:
            model (TransformerModel): 训练好的Transformer模型
            params (Dict[str, Any]): 模型参数
            history (Dict[str, List[float]]): 训练历史
            symbol (str): 金融资产代码
            target_col (str): 目标列名

        返回:
            str: 保存路径
        """
        # 创建保存目录
        save_dir = os.path.join(self.output_dir, 'models', f'{symbol}_transformer_{target_col}')
        os.makedirs(save_dir, exist_ok=True)

        # 保存模型状态
        model_path = os.path.join(save_dir, 'model.pth')
        torch.save(model.state_dict(), model_path)

        # 保存参数和历史
        info = {
            'params': params,
            'history': history
        }
        info_path = os.path.join(save_dir, 'info.json')
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)

        print(f"Transformer模型已保存到 {save_dir}")
        return save_dir

#------------------------------------------------------------------------------
# 主函数和工作流程
#------------------------------------------------------------------------------

def run_volatility_prediction(
    symbol: str,
    target_volatility_window: int,
    start_date: str,
    end_date: str,
    test_size: float = 0.2,
    seq_length: int = 20,
    output_dir: str = 'output',
    optimize_params: bool = False
) -> Dict[str, Any]:
    """
    运行完整的波动率预测工作流程。

    参数:
        symbol (str): 金融资产代码
        target_volatility_window (int): 目标波动率窗口
        start_date (str): 起始日期
        end_date (str): 结束日期
        test_size (float): 测试集比例
        seq_length (int): 序列长度
        output_dir (str): 输出目录
        optimize_params (bool): 是否优化模型参数

    返回:
        Dict[str, Any]: 包含预测结果和评估指标的字典
    """
    print(f"Starting volatility prediction workflow for {symbol}...")

    # 1. 数据处理
    data_processor = FinancialDataProcessor(output_dir)
    data = data_processor.process_asset(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        windows=[7, 14, 21, 30],
        visualize=True
    )

    # 2. 特征工程和模型训练
    target_col = f'volatility_{target_volatility_window}d'
    print(f"Target column: {target_col}")

    model_builder = VolatilityPredictionModels(output_dir)

    # 创建特征
    data_with_features = model_builder.create_features(data, target_col)

    data_with_features.to_csv(os.path.join(output_dir, 'data', f'{symbol}_with_features.csv'), index=False)
    print(f"Feature engineering data saved to {os.path.join(output_dir, 'data', f'{symbol}_with_features.csv')}")

    # 分割数据
    X_train, X_test, y_train, y_test, feature_cols = model_builder.train_test_split(
        data_with_features,
        target_col,
        test_size=test_size
    )

    # 标准化特征
    X_train_scaled, X_test_scaled, scaler = model_builder.scale_data(X_train, X_test)

    # 获取测试集日期
    test_dates = data_with_features.iloc[-len(y_test):]['Date'].values

    # 保存预测结果和评估指标
    predictions = {}
    evaluations = []

    # 3. 训练和评估GARCH模型
    try:
        # Get original return data for GARCH
        train_returns = data.iloc[:int(len(data) * (1 - test_size))]['log_return']
        test_returns = data.iloc[int(len(data) * (1 - test_size)):]['log_return']

        # Get actual volatility values from the target column
        test_vol = data.iloc[int(len(data) * (1 - test_size)):][target_col].values

        # Train GARCH model
        garch_model_dict = model_builder.train_auto_garch(
            train_returns,
            max_p=2,
            max_q=2
        )

        # Execute rolling prediction - make sure this returns arrays of same length
        _, garch_pred = model_builder.rolling_garch_forecast(
            pd.Series(np.concatenate([train_returns[-252:], test_returns])),
            train_size=252,
            p=garch_model_dict['order'][0],
            q=garch_model_dict['order'][1]
        )

        # Make sure arrays are same length before evaluation and visualization
        if len(garch_pred) == len(test_vol):
            predictions['GARCH'] = garch_pred
            garch_metrics = model_builder.evaluate_model(test_vol, garch_pred, 'GARCH')
            evaluations.append(garch_metrics)
        else:
            # If lengths don't match, align them
            min_len = min(len(garch_pred), len(test_vol))
            predictions['GARCH'] = garch_pred[:min_len]
            garch_metrics = model_builder.evaluate_model(test_vol[:min_len], garch_pred[:min_len], 'GARCH')
            evaluations.append(garch_metrics)

    except Exception as e:
        print(f"Error in GARCH model: {e}")

    # 4. 训练和评估随机森林模型 (替代SVR)
    try:
        if optimize_params:
            rf_params, rf_model = model_builder.optimize_random_forest(X_train_scaled, y_train)
            # 直接使用已训练的最佳模型
        else:
            # 使用默认参数
            rf_params = {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt'
            }
            rf_model = model_builder.train_random_forest(X_train_scaled, y_train, rf_params)

        rf_pred = model_builder.predict_random_forest(rf_model, X_test_scaled)

        predictions['RandomForest'] = rf_pred
        rf_metrics = model_builder.evaluate_model(y_test, rf_pred, 'RandomForest')
        evaluations.append(rf_metrics)
        model_builder.save_evaluation(rf_metrics, symbol, target_col)
        model_builder.save_predictions(test_dates, y_test, rf_pred, 'RandomForest', symbol, target_col)

        # 分析并可视化特征重要性
        model_builder.feature_importance_analysis(rf_model, feature_cols, symbol, target_col)

        # 保存随机森林模型
        joblib.dump(rf_model, os.path.join(output_dir, 'models', f'{symbol}_randomforest_{target_col}.pkl'))
    except Exception as e:
        print(f"Error in Random Forest model: {e}")

    # 5. 准备深度学习的验证集
    val_size = int(len(X_train_scaled) * 0.2)
    X_val_scaled = X_train_scaled[-val_size:]
    y_val = y_train[-val_size:]
    X_train_dl = X_train_scaled[:-val_size]
    y_train_dl = y_train[:-val_size]

    # 6. 训练和评估LSTM模型
    try:
        if optimize_params:
            lstm_params = model_builder.optimize_lstm_hyperparams(
                X_train_dl, y_train_dl, X_val_scaled, y_val,
                n_trials=10, seq_length=seq_length
            )
        else:
            # 使用默认参数
            lstm_params = {
                'hidden_dim': 128,
                'num_layers': 2,
                'dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 64
            }

        lstm_model, lstm_history = model_builder.train_lstm(
            X_train_dl, y_train_dl, X_val_scaled, y_val,
            params=lstm_params, seq_length=seq_length
        )

        lstm_pred = model_builder.predict_lstm(lstm_model, X_test_scaled, seq_length)

        # 调整预测结果长度以匹配测试集
        if len(lstm_pred) < len(y_test):
            y_test_lstm = y_test[seq_length:]
            test_dates_lstm = test_dates[seq_length:]
        else:
            y_test_lstm = y_test
            test_dates_lstm = test_dates
            lstm_pred = lstm_pred[:len(y_test)]

        predictions['LSTM'] = lstm_pred
        lstm_metrics = model_builder.evaluate_model(y_test_lstm, lstm_pred, 'LSTM')
        evaluations.append(lstm_metrics)
        model_builder.save_evaluation(lstm_metrics, symbol, target_col)
        model_builder.save_predictions(test_dates_lstm, y_test_lstm, lstm_pred, 'LSTM', symbol, target_col)

        # 保存LSTM模型
        model_builder.save_lstm_model(lstm_model, lstm_params, lstm_history, symbol, target_col)
    except Exception as e:
        print(f"Error in LSTM model: {e}")

    # 7. 训练和评估Transformer模型
    try:
        if optimize_params:
            transformer_params = model_builder.optimize_transformer_hyperparams(
                X_train_dl, y_train_dl, X_val_scaled, y_val,
                n_trials=10, seq_length=seq_length
            )
        else:
            # 使用默认参数
            transformer_params = {
                'd_model': 128,
                'nhead': 4,
                'num_encoder_layers': 2,
                'dim_feedforward': 512,
                'dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32
            }

        transformer_model, transformer_history = model_builder.train_transformer(
            X_train_dl, y_train_dl, X_val_scaled, y_val,
            params=transformer_params, seq_length=seq_length
        )

        transformer_pred = model_builder.predict_transformer(transformer_model, X_test_scaled, seq_length)

        # 调整预测结果长度
        if len(transformer_pred) < len(y_test):
            y_test_transformer = y_test[seq_length:]
            test_dates_transformer = test_dates[seq_length:]
        else:
            y_test_transformer = y_test
            test_dates_transformer = test_dates
            transformer_pred = transformer_pred[:len(y_test)]

        predictions['Transformer'] = transformer_pred
        transformer_metrics = model_builder.evaluate_model(y_test_transformer, transformer_pred, 'Transformer')
        evaluations.append(transformer_metrics)
        model_builder.save_evaluation(transformer_metrics, symbol, target_col)
        model_builder.save_predictions(test_dates_transformer, y_test_transformer, transformer_pred, 'Transformer', symbol, target_col)

        # 保存Transformer模型
        model_builder.save_transformer_model(transformer_model, transformer_params, transformer_history, symbol, target_col)
    except Exception as e:
        print(f"Error in Transformer model: {e}")

    # 8. 可视化模型比较
    if len(predictions) >= 2:  # 至少有两个模型成功训练
        # 调整预测字典，确保所有预测长度一致
        aligned_predictions = {}

        # 找出共同的预测长度（以最短的为准）
        if 'LSTM' in predictions:
            aligned_y_test = y_test_lstm
            aligned_test_dates = test_dates_lstm
        elif 'Transformer' in predictions:
            aligned_y_test = y_test_transformer
            aligned_test_dates = test_dates_transformer
        else:
            aligned_y_test = y_test
            aligned_test_dates = test_dates

        for model_name, preds in predictions.items():
            if model_name in ['LSTM', 'Transformer']:
                # 深度学习模型预测已经处理过
                if len(preds) == len(aligned_y_test):
                    aligned_predictions[model_name] = preds
                else:
                    print(f"Skipping model {model_name} (prediction length mismatch)")
            else:
                # 处理其他模型预测
                if len(preds) > len(aligned_y_test):
                    aligned_predictions[model_name] = preds[-len(aligned_y_test):]
                elif len(preds) == len(aligned_y_test):
                    aligned_predictions[model_name] = preds
                else:
                    # 如果预测较短，则不包含在比较中
                    print(f"Skipping model {model_name} (prediction length mismatch)")

        # 可视化预测结果
        if len(aligned_predictions) >= 2:
            model_builder.visualize_predictions(
                aligned_test_dates, aligned_y_test, aligned_predictions,
                symbol, target_col, save=True, show=False
            )

        # 可视化模型比较
        model_builder.visualize_model_comparison(
            evaluations, symbol, target_col,
            metric='rmse', save=True, show=False
        )

        # 可视化R²比较
        model_builder.visualize_model_comparison(
            evaluations, symbol, target_col,
            metric='r2', save=True, show=False
        )

        # 可视化MAE比较
        model_builder.visualize_model_comparison(
            evaluations, symbol, target_col,
            metric='mae', save=True, show=False
        )

        # 可视化MSE比较
        model_builder.visualize_model_comparison(
            evaluations, symbol, target_col,
            metric='mse', save=True, show=False
        )

    print(f"Volatility prediction workflow completed for {symbol}!")

    return {
        'predictions': predictions,
        'evaluations': evaluations,
        'target_col': target_col,
        'feature_cols': feature_cols
    }

def main():
    """
    主函数，运行多个资产和不同波动率窗口的预测。
    """
    # 确保使用非交互式后端
    matplotlib.use('Agg')  # 必须在导入pyplot之前设置
    plt.ioff()  # 关闭交互模式

    # 输出目录
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # 设置参数
    start_date = '2015-01-01'
    end_date = '2023-12-31'

    # 目标资产
    assets = [
        '^GSPC',       # 标普500
        '000300.SS'    # 沪深300
    ]

    # 波动率窗口
    windows = [7, 14, 21, 30]

    # 执行预测
    results = {}

    for symbol in assets:
        results[symbol] = {}
        for window in windows:
            print(f"\n{'='*80}")
            print(f"Processing {symbol} with {window}-day volatility window")
            print(f"{'='*80}\n")

            try:
                results[symbol][window] = run_volatility_prediction(
                    symbol=symbol,
                    target_volatility_window=window,
                    start_date=start_date,
                    end_date=end_date,
                    test_size=0.2,
                    seq_length=20,
                    output_dir=output_dir,
                    optimize_params=True  # 设置为True进行参数优化（更耗时）
                )
            except Exception as e:
                print(f"Error processing {symbol} with {window}-day window: {e}")
                continue

    print("\nAll prediction tasks completed!")
    return results

if __name__ == "__main__":
    # 确保所有图形对象都被关闭
    plt.close('all')

    try:
        main()
    finally:
        # 确保所有图形对象都被关闭
        plt.close('all')

if __name__ == "__main__":
    main()