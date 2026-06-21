import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import time
import random
import os
from sklearn.preprocessing import StandardScaler

# Tắt cảnh báo TensorFlow để tránh làm bẩn log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras import backend as K
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

# --- CẤU HÌNH TRANG & CSS PREMIUM ---
st.set_page_config(
    page_title="Tối ưu hóa Danh mục Đầu tư LSTM-GRU",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Nhúng CSS tùy chỉnh để làm giao diện đẹp và hiện đại hơn
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Giao diện màu tối cao cấp */
    .main {
        background-color: #0F121C;
        color: #E2E8F0;
    }
    
    /* Cấu hình các khối thông tin (Cards) */
    .card-box {
        background-color: #181D2D;
        border: 1px solid #232D42;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s, border-color 0.2s;
    }
    
    .card-box:hover {
        transform: translateY(-2px);
        border-color: #3b82f6;
    }
    
    .card-title {
        font-size: 15px;
        color: #94A3B8;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 500;
    }
    
    .card-value {
        font-size: 32px;
        font-weight: 700;
        color: #3B82F6;
    }
    
    .card-subtitle {
        font-size: 13px;
        color: #64748B;
        margin-top: 5px;
    }

    /* Tùy chỉnh Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #181D2D;
        padding: 10px 10px 0px 10px;
        border-radius: 12px 12px 0 0;
        border-bottom: 1px solid #232D42;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: transparent;
        border-radius: 8px 8px 0px 0px;
        color: #94A3B8;
        font-weight: 500;
        font-size: 15px;
        transition: background-color 0.2s, color 0.2s;
        border: none;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #3B82F6;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #232D42 !important;
        color: #3B82F6 !important;
        border-bottom: 2px solid #3B82F6 !important;
    }
    
    /* Tùy chỉnh nút bấm */
    div.stButton > button:first-child {
        background-image: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 28px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 10px rgba(59, 130, 246, 0.2);
        width: 100%;
    }
    
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.4);
    }
    
    /* Thiết kế sidebar */
    section[data-testid="stSidebar"] {
        background-color: #090B11;
        border-right: 1px solid #181D2D;
    }
    
    .sidebar-header {
        font-size: 20px;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 20px;
        border-bottom: 1px solid #232D42;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD TICKERS TỪ FILE HOẶC FALLBACK ---
try:
    from industry_tickers import INDUSTRY_TICKERS
except ImportError:
    # Fallback nếu file industry_tickers.py không tồn tại
    INDUSTRY_TICKERS = {
        "Thép": ['BVG','TTS','DTL','CBI','VDT','VGS','TMG','VGL','HLA','BCA','TVN','HPG','HMC','HMG','HSG','ITQ','KMT','KVC','MEL','MHL','POM','NKG','KKC','SMC','SHA','SSM','TDS','PAS','SHI','VCA','TTH','TLH','TNA','TNB','TNS','TNI','GDA','KTL','HSV','DFC','TIS'],
        "Bán buôn": ['CMC', 'DGW', 'FID', 'GEL', 'HHS', 'HMC', 'HTL', 'JVC', 'KDM', 'KMT', 'MCF', 'MEL', 'NO1', 'PCT', 'PET', 'PIT', 'PLX', 'PMG', 'PPT', 'PPY', 'PSC', 'PSD', 'PSE', 'PTX', 'SDA', 'SHN', 'SMA', 'SMC', 'SRA', 'ST8', 'TDG', 'THS', 'TLH', 'TNI', 'TSC', 'TTH', 'UNI', 'VCM', 'VFG', 'VID', 'VMD', 'VPG', 'VTV', 'VVS'],
        "Bán lẻ": ['BTT', 'CCI', 'CMV', 'COM', 'CTF', 'FRT', 'GMA', 'HAX', 'HTC', 'MWG', 'PNC', 'PNJ', 'SFC', 'SVC', 'TMC'],
        "Bảo hiểm": ['BIC', 'BMI', 'BVH', 'MIG', 'PGI', 'PRE', 'PTI', 'PVI', 'VNR'],
        "Bất động sản": ['AAV', 'AGG', 'API', 'BAX', 'BCE', 'BCM', 'CCL', 'CDC', 'CEO', 'CIG', 'CKG', 'CRE', 'CRV', 'CSC', 'D11', 'D2D', 'DIG', 'DRH', 'DTA', 'DTD', 'DXG', 'DXS', 'EVG', 'FDC', 'FIR', 'HAR', 'HDC', 'HDG', 'HLD', 'HPX', 'HQC', 'HU1', 'ICG', 'IDJ', 'IDV', 'IJC', 'ITC', 'KBC', 'KDH', 'KHG', 'KOS', 'KSF', 'L14', 'LDG', 'LGL', 'LHG', 'NBB', 'NDN', 'NHA', 'NLG', 'NRC', 'NTC', 'NTL', 'NVL', 'PDR', 'PTL', 'PV2', 'QCG', 'RCL', 'SCR', 'SDU', 'SGR', 'SJS', 'SZB', 'SZC', 'SZL', 'TAL', 'TCH', 'TDC', 'TDH', 'TIG', 'TIP', 'TIX', 'TN1', 'V21', 'VC3', 'VC7', 'VHM', 'VIC', 'VPH', 'VPI', 'VRC', 'VRE', 'VTJ'],
        "Ngân hàng": ['ACB', 'BAB', 'BID', 'CTG', 'EIB', 'EVF', 'HDB', 'KLB', 'LPB', 'MBB', 'MSB', 'NAB', 'NVB', 'OCB', 'SHB', 'SSB', 'STB', 'TCB', 'TPB', 'VAB', 'VCB', 'VIB', 'VPB'],
        "Chứng khoán": ['AGR', 'APG', 'APS', 'BSI', 'BVS', 'CTS', 'DSC', 'DSE', 'EVS', 'FTS', 'HBS', 'HCM', 'IVS', 'MBS', 'ORS', 'PSI', 'SHS', 'SSI', 'TCI', 'TCX', 'TVB', 'TVS', 'VCI', 'VCK', 'VDS', 'VFS', 'VIG', 'VIX', 'VND', 'VPX', 'WSS'],
        "Công nghệ và thông tin": ['ABR', 'ADC', 'BED', 'CKV', 'CMG', 'CTR', 'DAD', 'DAE', 'DST', 'EBS', 'ECI', 'EID', 'ELC', 'FPT', 'GLT', 'HEV', 'ICT', 'ITD', 'KST', 'LBE', 'ONE', 'PIA', 'QST', 'SED', 'SGD', 'SGT', 'SMN', 'STC', 'VTC']
    }

# --- CẤU HÌNH SIDEBAR ---
st.sidebar.markdown('<div class="sidebar-header">📈 Cấu Hình Tham Số</div>', unsafe_allow_html=True)

# 1. Chọn nhóm ngành hoặc upload tệp CSV
data_source = st.sidebar.radio("Nguồn dữ liệu:", ["Chọn ngành có sẵn", "Tải tệp CSV lên"])

selected_tickers = []
uploaded_file = None
selected_industry = "Thép"

if data_source == "Chọn ngành có sẵn":
    industries = list(INDUSTRY_TICKERS.keys())
    selected_industry = st.sidebar.selectbox("Chọn ngành phân tích:", industries, index=industries.index("Thép") if "Thép" in industries else 0)
    selected_tickers = sorted(list(set(INDUSTRY_TICKERS[selected_industry])))
else:
    uploaded_file = st.sidebar.file_uploader("Tải tệp CSV dữ liệu giá (Các cột: Date, Ticker1, Ticker2,...):", type=["csv"])

# 2. Chọn mốc thời gian
col_dates = st.sidebar.columns(2)
with col_dates[0]:
    train_start_date = st.date_input("Train Start:", pd.to_datetime("2015-01-01"))
    test_start_date = st.date_input("Test Start:", pd.to_datetime("2025-01-01"))
with col_dates[1]:
    train_end_date = st.date_input("Train End:", pd.to_datetime("2024-12-31"))
    test_end_date = st.date_input("Test End:", pd.to_datetime("2025-12-31"))

# 3. Tham số tài chính
st.sidebar.markdown("---")
st.sidebar.markdown("**Tham Số Tài Chính**")
rf_annual = st.sidebar.number_input("Lãi suất phi rủi ro năm (Rf):", value=0.045, step=0.005, format="%.3f")
trading_days = st.sidebar.number_input("Số ngày giao dịch/năm:", value=252, step=1)
top_n = st.sidebar.slider("Số cổ phiếu Sharpe cao nhất (N):", min_value=2, max_value=20, value=10)

# 4. Tham số mô hình LSTM-GRU
st.sidebar.markdown("---")
st.sidebar.markdown("**Tham Số Học Máy**")
epochs = st.sidebar.slider("Số lượng Epochs:", min_value=5, max_value=150, value=50, step=5)
batch_size = st.sidebar.selectbox("Batch Size:", [16, 32, 64, 128], index=1)
window_size = st.sidebar.slider("Độ dài Window (ngày):", min_value=5, max_value=60, value=30, step=5)
horizon = st.sidebar.slider("Horizon dự báo (ngày):", min_value=1, max_value=20, value=5, step=1)
lstm_units = st.sidebar.number_input("LSTM Units:", value=96, step=8)
gru_units = st.sidebar.number_input("GRU Units:", value=48, step=8)

# Danh sách hạt giống để huấn luyện
all_seeds = [7, 21, 42, 99, 123]
selected_seeds = st.sidebar.multiselect("Hạt giống (Seeds) huấn luyện:", all_seeds, default=[42])

if not selected_seeds:
    st.sidebar.error("Vui lòng chọn ít nhất 1 hạt giống (Seed).")

# Hằng số và cấu hình huấn luyện
LAMBDA_ENTROPY = 0.01

# --- CÁC HÀM XỬ LÝ DỮ LIỆU & HUẤN LUYỆN ---
@st.cache_data
def download_stock_data(tickers, start_date, end_date):
    """Tải dữ liệu chứng khoán thời gian thực từ vnstock."""
    from vnstock import Vnstock
    all_data = []
    failed_tickers = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, ticker in enumerate(tickers):
        status_text.text(f"Đang tải {ticker} ({idx+1}/{len(tickers)})...")
        success = False
        retry_count = 0
        max_retry = 3
        
        while not success and retry_count < max_retry:
            try:
                stock = Vnstock().stock(symbol=ticker, source="KBS")
                df = stock.quote.history(
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                    interval="1D"
                )
                
                if df is not None and not df.empty:
                    df = df.copy()
                    df["ticker"] = ticker
                    
                    if "time" not in df.columns:
                        if "date" in df.columns:
                            df["time"] = df["date"]
                        elif "datetime" in df.columns:
                            df["time"] = df["datetime"]
                            
                    keep_cols = [c for c in ["time", "open", "high", "low", "close", "volume", "ticker"] if c in df.columns]
                    df = df[keep_cols]
                    all_data.append(df)
                    success = True
                else:
                    retry_count += 1
            except Exception as e:
                retry_count += 1
                time.sleep(1)
        
        if not success:
            failed_tickers.append(ticker)
        progress_bar.progress((idx + 1) / len(tickers))
        
    progress_bar.empty()
    status_text.empty()
    
    if all_data:
        raw_data = pd.concat(all_data, ignore_index=True)
        return raw_data, failed_tickers
    else:
        return None, failed_tickers

def compute_rsi(price_df, period=14):
    delta = price_df.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def build_features(price_df, return_df):
    common_idx = price_df.index.intersection(return_df.index)
    price_df = price_df.loc[common_idx].copy()
    return_df = return_df.loc[common_idx].copy()

    price_df = price_df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return_df = return_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    feat_list = []

    ret_1 = return_df.copy()
    ret_1.columns = [f"{c}_ret1" for c in ret_1.columns]
    feat_list.append(ret_1)

    ret_5 = price_df.pct_change(5)
    ret_5.columns = [f"{c}_ret5" for c in ret_5.columns]
    feat_list.append(ret_5)

    ret_10 = price_df.pct_change(10)
    ret_10.columns = [f"{c}_ret10" for c in ret_10.columns]
    feat_list.append(ret_10)

    ma5_ratio = price_df / (price_df.rolling(5, min_periods=5).mean() + 1e-9) - 1
    ma5_ratio.columns = [f"{c}_ma5_ratio" for c in ma5_ratio.columns]
    feat_list.append(ma5_ratio)

    ma10_ratio = price_df / (price_df.rolling(10, min_periods=10).mean() + 1e-9) - 1
    ma10_ratio.columns = [f"{c}_ma10_ratio" for c in ma10_ratio.columns]
    feat_list.append(ma10_ratio)

    vol5 = return_df.rolling(5, min_periods=5).std()
    vol5.columns = [f"{c}_vol5" for c in vol5.columns]
    feat_list.append(vol5)

    vol10 = return_df.rolling(10, min_periods=10).std()
    vol10.columns = [f"{c}_vol10" for c in vol10.columns]
    feat_list.append(vol10)

    mom5 = price_df.pct_change(5)
    mom5.columns = [f"{c}_mom5" for c in mom5.columns]
    feat_list.append(mom5)

    rsi14 = compute_rsi(price_df, period=14) / 100.0
    rsi14.columns = [f"{c}_rsi14" for c in rsi14.columns]
    feat_list.append(rsi14)

    features = pd.concat(feat_list, axis=1)
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna(axis=0, how="any")
    return features

def create_sequences_and_targets(features_df, target_returns_df, window_size, horizon=5):
    X, y, dates = [], [], []
    feat_values = features_df.values.astype(np.float32)
    target_values = target_returns_df.values.astype(np.float32)
    idx = features_df.index

    for i in range(len(features_df) - window_size - horizon + 1):
        X.append(feat_values[i:i + window_size])
        y.append(target_values[i + window_size:i + window_size + horizon].mean(axis=0))
        dates.append(idx[i + window_size + horizon - 1])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), pd.Index(dates)

# --- KHAI BÁO CÁC HÀM LIÊN QUAN TỚI TENSORFLOW (NẾU CÓ) ---
if HAS_TENSORFLOW:
    def sharpe_loss(y_true, y_pred):
        portfolio_returns = K.sum(y_true * y_pred, axis=1)
        portfolio_returns = portfolio_returns - (rf_annual / trading_days)
        mean_returns = K.mean(portfolio_returns)
        std_returns = K.std(portfolio_returns)
        sharpe = mean_returns / (std_returns + 1e-9)
        entropy = -K.sum(y_pred * K.log(y_pred + 1e-9), axis=1)
        entropy = K.mean(entropy)
        return -sharpe - LAMBDA_ENTROPY * entropy

    def build_lstm_gru_model(timesteps, n_features, n_assets):
        model = Sequential([
            Input(shape=(timesteps, n_features)),
            LSTM(lstm_units, return_sequences=True, activation="tanh", recurrent_activation="sigmoid"),
            Dropout(0.2),
            GRU(gru_units, return_sequences=False, activation="tanh", recurrent_activation="sigmoid"),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dropout(0.1),
            Dense(n_assets, activation="softmax")
        ])
        return model

    def set_seed(seed=42):
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    class StreamlitCallback(tf.keras.callbacks.Callback):
        def __init__(self, status_text, progress_bar, epochs_val, seed_val):
            self.status_text = status_text
            self.progress_bar = progress_bar
            self.epochs_val = epochs_val
            self.seed_val = seed_val

        def on_epoch_end(self, epoch, logs=None):
            loss = logs.get('loss', 0.0)
            val_loss = logs.get('val_loss', 0.0)
            percent = (epoch + 1) / self.epochs_val
            self.progress_bar.progress(percent)
            self.status_text.text(f"Hạt giống {self.seed_val} | Epoch {epoch + 1}/{self.epochs_val} | Loss: {loss:.4f} | Val Loss: {val_loss:.4f}")

    def train_one_run(seed, X_train, y_train_target, status_text, progress_bar):
        set_seed(seed)
        model = build_lstm_gru_model(
            timesteps=X_train.shape[1],
            n_features=X_train.shape[2],
            n_assets=y_train_target.shape[1]
        )
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss=sharpe_loss
        )
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
            StreamlitCallback(status_text, progress_bar, epochs, seed)
        ]
        history = model.fit(
            X_train,
            y_train_target,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=False,
            verbose=0,
            validation_split=0.2,
            callbacks=callbacks
        )
        return model, history

# --- CÁC HÀM TÍNH TOÁN HIỆU QUẢ ---
def port_char(weights_df, returns_df, annualize=True, freq=252):
    er = returns_df.mean().reset_index()
    er.columns = ["Asset", "Er"]
    weights_merged = pd.merge(weights_df, er, on="Asset", how="left")
    weights_merged["Er"] = weights_merged["Er"].fillna(0.0)
    portfolio_er_daily = np.dot(weights_merged["Weight"], weights_merged["Er"])
    
    cov_matrix = returns_df.cov()
    asset_order = weights_merged["Asset"].tolist()
    cov_matrix = cov_matrix.loc[asset_order, asset_order]
    
    w = weights_merged["Weight"].values
    portfolio_std_daily = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
    
    if annualize:
        portfolio_er = portfolio_er_daily * freq
        portfolio_std_dev = portfolio_std_daily * np.sqrt(freq)
    else:
        portfolio_er = portfolio_er_daily
        portfolio_std_dev = portfolio_std_daily
        
    return portfolio_er, portfolio_std_dev

def port_char_from_series(portfolio_return_series, annualize=True, freq=252):
    portfolio_return_series = pd.Series(portfolio_return_series).dropna()
    er_daily = portfolio_return_series.mean()
    std_daily = portfolio_return_series.std()
    
    if annualize:
        er = er_daily * freq
        std = std_daily * np.sqrt(freq)
    else:
        er = er_daily
        std = std_daily
        
    return er, std

def sharpe_port(weights_df, returns_df, rf=0.045, freq=252):
    portfolio_er, portfolio_std_dev = port_char(weights_df, returns_df, annualize=True, freq=freq)
    sharpe_ratio = (portfolio_er - rf) / (portfolio_std_dev + 1e-12)
    return sharpe_ratio

def sharpe_from_series(portfolio_return_series, rf=0.045, freq=252):
    portfolio_er, portfolio_std_dev = port_char_from_series(portfolio_return_series, annualize=True, freq=freq)
    sharpe_ratio = (portfolio_er - rf) / (portfolio_std_dev + 1e-12)
    return sharpe_ratio

def build_allocation_80_20(train_returns, rf_ann, days_freq):
    rf_daily = rf_ann / days_freq
    mean_ret = train_returns.mean()
    std_ret = train_returns.std().replace(0, np.nan)
    sharpe_train = ((mean_ret - rf_daily) / std_ret).dropna().sort_values(ascending=False)
    
    ranked = sharpe_train.reset_index()
    ranked.columns = ["Asset", "Score"]
    
    n_assets = len(ranked)
    top_count = max(1, int(np.ceil(0.2 * n_assets)))
    bottom_count = n_assets - top_count
    
    top_weights = [0.8 / top_count] * top_count
    bottom_weights = [0.2 / bottom_count] * bottom_count if bottom_count > 0 else []
    
    ranked["Weight"] = top_weights + bottom_weights
    return ranked[["Asset", "Weight"]]

# --- TIÊU ĐỀ ỨNG DỤNG ---
st.markdown('<h1 style="color: #F8FAFC; text-align: center;">📈 Tối Ưu Hóa Danh Mục Đầu Tư Bằng LSTM-GRU</h1>', unsafe_allow_html=True)
st.markdown('<p style="color: #94A3B8; text-align: center; font-size: 16px;">Phát triển bởi chuyên gia AI & Định lượng Tài chính. Mô hình hóa Sharpe tối ưu động bằng Deep Learning lai.</p>', unsafe_allow_html=True)
st.markdown('---')

# Cảnh báo nếu không có TensorFlow
if not HAS_TENSORFLOW:
    st.warning("""
    ⚠️ **Cảnh báo: Không thể tải thư viện TensorFlow (chưa tương thích với Python 3.13+) trên máy chủ này.**
    - Ứng dụng Streamlit vẫn có thể hiển thị dữ liệu lịch sử, phân tích Sharpe và tính toán các chiến lược truyền thống (**Phân bổ đều** & **Phân bổ 80-20**).
    - Để sử dụng mô hình học sâu **LSTM-GRU**, vui lòng mở phần cài đặt nâng cao (**Advanced Settings**) trong trang quản trị Streamlit Cloud của bạn và đổi phiên bản Python thành **3.11** hoặc **3.12**, sau đó nhấn **Save**.
    """)

# --- 1. PHẦN TẢI DỮ LIỆU ---
pivot_df_clean = None
raw_data = None

if data_source == "Chọn ngành có sẵn":
    if len(selected_tickers) > 0:
        with st.spinner(f"Đang kết nối nguồn dữ liệu chứng khoán ngành {selected_industry}..."):
            raw_data, failed_tickers = download_stock_data(selected_tickers, train_start_date, test_end_date)
        
        if failed_tickers:
            st.warning(f"Không tải được dữ liệu cho một số mã: {failed_tickers}")
            
        if raw_data is not None and not raw_data.empty:
            pivot_df_clean = raw_data.pivot_table(
                index="time",
                columns="ticker",
                values="close",
                aggfunc="last"
            ).sort_index()
            pivot_df_clean.index = pd.to_datetime(pivot_df_clean.index)
            pivot_df_clean = pivot_df_clean.sort_index()
    else:
        st.error("Không tìm thấy mã cổ phiếu nào cho nhóm ngành này.")
else:
    if uploaded_file is not None:
        try:
            pivot_df_clean = pd.read_csv(uploaded_file, index_col=0, parse_dates=True).sort_index()
            st.success("Tải tệp dữ liệu giá cổ phiếu của bạn thành công!")
        except Exception as e:
            st.error(f"Lỗi khi đọc tệp CSV: {e}")

# --- NẾU DỮ LIỆU ĐÃ ĐƯỢC LOAD ---
if pivot_df_clean is not None:
    # Điền giá trị trống và tính toán daily returns
    price_filled = pivot_df_clean.ffill().bfill()
    returns_df = price_filled.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="any")
    
    # Tạo các Tabs
    tab_data, tab_sharpe, tab_train, tab_allocation, tab_performance = st.tabs([
        "📊 Dữ Liệu Lịch Sử",
        "⚖️ Phân Tích Sharpe",
        "🧠 Huấn Luyện LSTM-GRU",
        "📐 Phân Bổ Tỷ Trọng",
        "🏆 Đánh Giá Hiệu Quả"
    ])
    
    # --- TAB 1: DỮ LIỆU LỊCH SỬ ---
    with tab_data:
        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Tổng quan dữ liệu gốc</div>', unsafe_allow_html=True)
        col_summary1, col_summary2 = st.columns(2)
        with col_summary1:
            st.write(f"**Số mã cổ phiếu:** {pivot_df_clean.shape[1]}")
            st.write(f"**Số phiên giao dịch:** {pivot_df_clean.shape[0]}")
        with col_summary2:
            st.write(f"**Ngày bắt đầu:** {pivot_df_clean.index.min().strftime('%d/%m/%Y')}")
            st.write(f"**Ngày kết thúc:** {pivot_df_clean.index.max().strftime('%d/%m/%Y')}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Biểu đồ giá chuẩn hóa
        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Biểu đồ giá chuẩn hóa (Cơ sở 100)</div>', unsafe_allow_html=True)
        norm_prices = (price_filled / price_filled.iloc[0]) * 100
        
        fig_price = go.Figure()
        for col in norm_prices.columns[:15]: # Show max 15 mã để tránh rối mắt
            fig_price.add_trace(go.Scatter(x=norm_prices.index, y=norm_prices[col], name=col, mode='lines'))
            
        fig_price.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=10, b=20),
            height=450,
            xaxis=dict(showgrid=True, gridcolor='#232D42'),
            yaxis=dict(showgrid=True, gridcolor='#232D42')
        )
        st.plotly_chart(fig_price, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Bảng hiển thị mẫu dữ liệu
        col_tbl1, col_tbl2 = st.columns(2)
        with col_tbl1:
            st.write("**Mẫu bảng giá đóng cửa (Close Price):**")
            st.dataframe(price_filled.head(10))
        with col_tbl2:
            st.write("**Mẫu tỷ suất sinh lời hàng ngày (Daily Returns):**")
            st.dataframe(returns_df.head(10))
            
    # --- TAB 2: PHÂN TÍCH SHARPE ---
    with tab_sharpe:
        # Tính toán Sharpe Ratio của từng mã trên tập huấn luyện
        rf_daily = rf_annual / trading_days
        train_returns_all = returns_df.loc[train_start_date:train_end_date]
        
        if not train_returns_all.empty:
            mean_ret_all = train_returns_all.mean()
            std_ret_all = train_returns_all.std().replace(0, np.nan)
            sharpe_all = ((mean_ret_all - rf_daily) / std_ret_all).dropna().sort_values(ascending=False)
            
            top_symbols = sharpe_all.head(top_n).index.tolist()
            
            col_sh1, col_sh2 = st.columns([1, 2])
            with col_sh1:
                st.markdown('<div class="card-box" style="height: 100%;">', unsafe_allow_html=True)
                st.markdown(f'<div class="card-title">Top {top_n} cổ phiếu được chọn</div>', unsafe_allow_html=True)
                st.write("Các cổ phiếu này có tỷ số Sharpe cao nhất trong tập huấn luyện và sẽ được đưa vào huấn luyện mô hình tối ưu hóa danh mục.")
                
                sh_table = pd.DataFrame({
                    "Hệ số Sharpe": sharpe_all.head(top_n)
                })
                st.dataframe(sh_table.style.format("{:.4f}"))
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col_sh2:
                st.markdown('<div class="card-box" style="height: 100%;">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">Biểu đồ Sharpe Ratio trong nhóm ngành</div>', unsafe_allow_html=True)
                
                fig_sharpe = px.bar(
                    x=sharpe_all.head(top_n).index,
                    y=sharpe_all.head(top_n).values,
                    labels={'x': 'Mã cổ phiếu', 'y': 'Sharpe Ratio'},
                    color=sharpe_all.head(top_n).values,
                    color_continuous_scale=px.colors.sequential.Blues
                )
                fig_sharpe.update_layout(
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=20, r=20, t=10, b=20),
                    height=380,
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='#232D42'),
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_sharpe, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("Không có dữ liệu trong khoảng thời gian huấn luyện được chọn.")
            
    # Chuẩn bị trước các dữ liệu train/test cho top_symbols
    price_top10 = price_filled[top_symbols].copy()
    returns_top10 = returns_df[top_symbols].copy()
    
    train_prices = price_top10.loc[train_start_date:train_end_date].copy()
    test_prices = price_top10.loc[test_start_date:test_end_date].copy()
    
    train_prices = train_prices.sort_index().ffill().bfill()
    test_prices = test_prices.sort_index().ffill().bfill()
    
    train_returns = train_prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="any")
    test_returns = test_prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="any")

    # --- TAB 3: HUẤN LUYỆN LSTM-GRU ---
    with tab_train:
        if HAS_TENSORFLOW:
            # Chuẩn bị dữ liệu đặc trưng cho LSTM-GRU
            train_features = build_features(train_prices, train_returns)
            test_features = build_features(test_prices, test_returns)
            
            # Chuẩn hóa
            scaler = StandardScaler()
            train_features_scaled = pd.DataFrame(
                scaler.fit_transform(train_features),
                index=train_features.index,
                columns=train_features.columns
            )
            test_features_scaled = pd.DataFrame(
                scaler.transform(test_features),
                index=test_features.index,
                columns=test_features.columns
            )
            
            train_target_returns = train_returns.loc[train_features_scaled.index].copy()
            test_target_returns = test_returns.loc[test_features_scaled.index].copy()
            
            X_train, y_train_target, train_seq_dates = create_sequences_and_targets(
                train_features_scaled, train_target_returns, window_size, horizon=horizon
            )
            X_test, y_test_target, test_seq_dates = create_sequences_and_targets(
                test_features_scaled, test_target_returns, window_size, horizon=horizon
            )
            
            col_tr1, col_tr2 = st.columns([1, 2])
            
            with col_tr1:
                st.markdown('<div class="card-box">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">Cấu hình huấn luyện</div>', unsafe_allow_html=True)
                st.write(f"**Số lượng đặc trưng đầu vào:** {X_train.shape[2]}")
                st.write(f"**Số lượng mẫu Train (Sequences):** {X_train.shape[0]}")
                st.write(f"**Số lượng mẫu Test (Sequences):** {X_test.shape[0]}")
                st.write(f"**Số lượng hạt giống đang chọn:** {len(selected_seeds)}")
                
                # Nút bấm bắt đầu huấn luyện
                btn_train = st.button("▶️ Bắt Đầu Huấn Luyện (Chạy Tối Ưu Hóa)")
                st.markdown('</div>', unsafe_allow_html=True)
                
                if btn_train:
                    if not selected_seeds:
                        st.error("Không có hạt giống nào được chọn để huấn luyện.")
                    else:
                        results_runs = []
                        best_model = None
                        best_sharpe = -1e9
                        best_seed = None
                        best_history = None
                        best_portfolio_returns = None
                        best_pred_weights_test = None
                        
                        status_text = st.empty()
                        progress_bar = st.progress(0)
                        
                        # Chạy mô hình trên từng seed
                        for seed_idx, seed in enumerate(selected_seeds):
                            status_text.text(f"Đang chuẩn bị hạt giống {seed}...")
                            model, history = train_one_run(seed, X_train, y_train_target, status_text, progress_bar)
                            
                            pred_weights_test = model.predict(X_test, verbose=0)
                            
                            weights_test_df = pd.DataFrame(
                                pred_weights_test,
                                index=test_seq_dates,
                                columns=top_symbols
                            )
                            y_test_df = pd.DataFrame(
                                y_test_target,
                                index=test_seq_dates,
                                columns=top_symbols
                            )
                            
                            portfolio_returns = (weights_test_df * y_test_df).sum(axis=1)
                            run_er = portfolio_returns.mean() * trading_days
                            run_std = portfolio_returns.std() * np.sqrt(trading_days)
                            run_sharpe = (run_er - rf_annual) / (run_std + 1e-12)
                            
                            results_runs.append({
                                "seed": seed,
                                "Lợi nhuận TB năm": run_er,
                                "Độ lệch chuẩn năm": run_std,
                                "Sharpe": run_sharpe
                            })
                            
                            if run_sharpe > best_sharpe:
                                best_sharpe = run_sharpe
                                best_seed = seed
                                best_model = model
                                best_history = {
                                    "loss": history.history["loss"],
                                    "val_loss": history.history["val_loss"]
                                }
                                best_portfolio_returns = portfolio_returns.copy()
                                best_pred_weights_test = pred_weights_test.copy()
                                
                        status_text.text("Huấn luyện hoàn tất thành công!")
                        progress_bar.progress(1.0)
                        
                        # Lưu vào Session State
                        st.session_state["trained"] = True
                        st.session_state["results_runs_df"] = pd.DataFrame(results_runs)
                        st.session_state["best_seed"] = best_seed
                        st.session_state["best_sharpe"] = best_sharpe
                        st.session_state["best_history"] = best_history
                        st.session_state["best_pred_weights_test"] = best_pred_weights_test
                        st.session_state["portfolio_returns_lstm_dynamic"] = best_portfolio_returns
                        
            with col_tr2:
                st.markdown('<div class="card-box" style="height: 100%;">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">Đồ thị kết quả và quá trình huấn luyện</div>', unsafe_allow_html=True)
                
                if st.session_state.get("trained", False):
                    # 1. Hiển thị bảng kết quả các run
                    st.write(f"**Kết quả chạy trên các hạt giống (Seed):**")
                    st.dataframe(st.session_state["results_runs_df"].sort_values("Sharpe", ascending=False).style.format({
                        "Lợi nhuận TB năm": "{:.2%}",
                        "Độ lệch chuẩn năm": "{:.2%}",
                        "Sharpe": "{:.4f}"
                    }))
                    
                    # 2. Vẽ biểu đồ Loss của seed tốt nhất
                    best_hist = st.session_state["best_history"]
                    fig_loss = go.Figure()
                    fig_loss.add_trace(go.Scatter(y=best_hist["loss"], name="Train Loss", mode="lines", line=dict(color="#3B82F6")))
                    fig_loss.add_trace(go.Scatter(y=best_hist["val_loss"], name="Val Loss", mode="lines", line=dict(color="#EF4444")))
                    
                    fig_loss.update_layout(
                        title=f"Quá trình giảm hàm mất mát (Loss) - Best Seed = {st.session_state['best_seed']}",
                        template="plotly_dark",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=20, r=20, t=40, b=20),
                        height=300,
                        xaxis=dict(title="Epoch", showgrid=True, gridcolor='#232D42'),
                        yaxis=dict(title="Loss", showgrid=True, gridcolor='#232D42')
                    )
                    st.plotly_chart(fig_loss, use_container_width=True)
                else:
                    st.info("Hãy bấm nút huấn luyện ở cột bên trái để bắt đầu tạo mô hình tối ưu hóa danh mục đầu tư.")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("Mô hình học máy LSTM-GRU bị vô hiệu hóa vì không tìm thấy thư viện TensorFlow tương thích trên máy chủ này. Vui lòng làm theo hướng dẫn ở đầu trang để đổi phiên bản Python.")
            
    # --- TAB 4: PHÂN BỔ TỶ TRỌNG ---
    with tab_allocation:
        # Tính các tỷ trọng danh mục đối chứng (Phân bổ đều, Phân bổ 80-20)
        Allo_1 = pd.DataFrame({
            "Asset": top_symbols,
            "Weight": [1 / len(top_symbols)] * len(top_symbols)
        })
        
        Allo_2 = build_allocation_80_20(train_returns, rf_annual, trading_days)
        
        col_al1, col_al2 = st.columns([1, 2])
        
        with col_al1:
            st.markdown('<div class="card-box" style="height: 100%;">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Tỷ trọng các chiến lược</div>', unsafe_allow_html=True)
            
            # Ghép bảng tỷ trọng
            if st.session_state.get("trained", False):
                best_pred_weights = st.session_state["best_pred_weights_test"]
                weights_lstm_gru_avg = best_pred_weights.mean(axis=0)
                results_LSTM_GRU = pd.DataFrame({
                    "Asset": top_symbols,
                    "Weight": weights_lstm_gru_avg
                }).sort_values("Weight", ascending=False).reset_index(drop=True)
                
                tbl_weights = pd.merge(Allo_1, Allo_2, on="Asset", suffixes=("_Equal", "_80_20"))
                tbl_weights = pd.merge(tbl_weights, results_LSTM_GRU, on="Asset")
                tbl_weights.columns = ["Cổ phiếu", "Phân bổ đều", "Phân bổ 80-20", "LSTM-GRU (Mô hình)"]
                st.dataframe(tbl_weights.style.format({
                    "Phân bổ đều": "{:.2%}",
                    "Phân bổ 80-20": "{:.2%}",
                    "LSTM-GRU (Mô hình)": "{:.2%}"
                }))
            else:
                tbl_weights = pd.merge(Allo_1, Allo_2, on="Asset", suffixes=("_Equal", "_80_20"))
                tbl_weights.columns = ["Cổ phiếu", "Phân bổ đều", "Phân bổ 80-20"]
                st.dataframe(tbl_weights.style.format({
                    "Phân bổ đều": "{:.2%}",
                    "Phân bổ 80-20": "{:.2%}"
                }))
                st.info("⚠️ Gợi ý: Huấn luyện mô hình ở Tab 3 để so sánh thêm với LSTM-GRU.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_al2:
            st.markdown('<div class="card-box" style="height: 100%;">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Biểu đồ phân bổ tỷ trọng (So sánh)</div>', unsafe_allow_html=True)
            
            fig_alloc = go.Figure()
            if st.session_state.get("trained", False):
                fig_alloc.add_trace(go.Bar(x=tbl_weights["Cổ phiếu"], y=tbl_weights["LSTM-GRU (Mô hình)"]*100, name="LSTM-GRU (Mô hình)", marker_color='#3B82F6'))
            fig_alloc.add_trace(go.Bar(x=tbl_weights["Cổ phiếu"], y=tbl_weights["Phân bổ đều"]*100, name="Phân bổ đều", marker_color='#10B981'))
            fig_alloc.add_trace(go.Bar(x=tbl_weights["Cổ phiếu"], y=tbl_weights["Phân bổ 80-20"]*100, name="Phân bổ 80-20", marker_color='#F59E0B'))
            
            fig_alloc.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=10, b=20),
                height=380,
                xaxis=dict(showgrid=False),
                yaxis=dict(title="Tỷ trọng (%)", showgrid=True, gridcolor='#232D42'),
                barmode='group'
            )
            st.plotly_chart(fig_alloc, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Vẽ biểu đồ Treemap (Plotly) nếu đã train thành công
        if st.session_state.get("trained", False):
            st.markdown('<div class="card-box">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Biểu đồ khối phân bổ tỷ trọng đầu tư của LSTM-GRU</div>', unsafe_allow_html=True)
            fig_tree = px.treemap(
                results_LSTM_GRU,
                path=['Asset'],
                values='Weight',
                color='Weight',
                color_continuous_scale=px.colors.sequential.Blues,
                custom_data=['Weight']
            )
            fig_tree.update_traces(
                textinfo="label+percent entry",
                hovertemplate="<b>%{label}</b><br>Tỷ trọng: %{customdata[0]:.2%}"
            )
            fig_tree.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=10, b=10),
                height=400,
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_tree, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
    # --- TAB 5: ĐÁNH GIÁ HIỆU QUẢ ---
    with tab_performance:
        # Tính toán hiệu năng cho 2 chiến dịch đối chứng
        Er_1, std_1 = port_char(Allo_1, test_returns, annualize=True, freq=trading_days)
        Er_2, std_2 = port_char(Allo_2, test_returns, annualize=True, freq=trading_days)
        
        sharpe_1 = sharpe_port(Allo_1, test_returns, rf=rf_annual, freq=trading_days)
        sharpe_2 = sharpe_port(Allo_2, test_returns, rf=rf_annual, freq=trading_days)
        
        strategies = ["Phân bổ đều", "Phân bổ 80-20"]
        er_vals = [Er_1, Er_2]
        std_vals = [std_1, std_2]
        sharpe_vals = [sharpe_1, sharpe_2]
        
        # Nếu đã train, bổ sung thêm mô hình LSTM-GRU
        if st.session_state.get("trained", False):
            portfolio_returns_lstm_dynamic = st.session_state["portfolio_returns_lstm_dynamic"]
            Er_lstm, std_lstm = port_char_from_series(portfolio_returns_lstm_dynamic, annualize=True, freq=trading_days)
            sharpe_lstm = sharpe_from_series(portfolio_returns_lstm_dynamic, rf=rf_annual, freq=trading_days)
            
            strategies.insert(0, "LSTM-GRU (Dynamic)")
            er_vals.insert(0, Er_lstm)
            std_vals.insert(0, std_lstm)
            sharpe_vals.insert(0, sharpe_lstm)
            
        comparison_table = pd.DataFrame({
            "Chiến lược đầu tư": strategies,
            "Lợi nhuận trung bình": er_vals,
            "Độ lệch chuẩn": std_vals,
            "Hệ số Sharpe": sharpe_vals
        })
        
        # Vẽ KPI Cards
        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
        with col_kpi1:
            st.markdown(f"""
            <div class="card-box">
                <div class="card-title">Hệ số Sharpe Cao Nhất</div>
                <div class="card-value">{max(sharpe_vals):.4f}</div>
                <div class="card-subtitle">Mức bù rủi ro tối ưu</div>
            </div>
            """, unsafe_allow_html=True)
        with col_kpi2:
            st.markdown(f"""
            <div class="card-box">
                <div class="card-title">Lợi Nhuận Tối Đa</div>
                <div class="card-value">{max(er_vals):.2%}</div>
                <div class="card-subtitle">Chiến lược có hiệu quả cao nhất</div>
            </div>
            """, unsafe_allow_html=True)
        with col_kpi3:
            st.markdown(f"""
            <div class="card-box">
                <div class="card-title">Rủi ro Thấp Nhất</div>
                <div class="card-value">{min(std_vals):.2%}</div>
                <div class="card-subtitle">Chiến lược có biến động thấp nhất</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Hiển thị bảng so sánh
        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Bảng so sánh hiệu quả các chiến lược</div>', unsafe_allow_html=True)
        
        comparison_table_formatted = comparison_table.copy()
        comparison_table_formatted["Lợi nhuận trung bình"] = comparison_table_formatted["Lợi nhuận trung bình"].map(lambda x: f"{x:.2%}")
        comparison_table_formatted["Độ lệch chuẩn"] = comparison_table_formatted["Độ lệch chuẩn"].map(lambda x: f"{x:.2%}")
        comparison_table_formatted["Hệ số Sharpe"] = comparison_table_formatted["Hệ số Sharpe"].map(lambda x: f"{x:.4f}")
        st.dataframe(comparison_table_formatted)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Biểu đồ dual axis dạng cột & đường kết hợp
        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Biểu đồ so sánh Dual Axis (Lợi nhuận, Rủi ro & Sharpe)</div>', unsafe_allow_html=True)
        
        fig_dual = go.Figure()
        # Cột Lợi nhuận
        fig_dual.add_trace(go.Bar(
            x=comparison_table["Chiến lược đầu tư"],
            y=comparison_table["Lợi nhuận trung bình"]*100,
            name="Lợi nhuận trung bình (%)",
            marker_color='#3B82F6',
            yaxis='y1'
        ))
        # Cột Độ lệch chuẩn
        fig_dual.add_trace(go.Bar(
            x=comparison_table["Chiến lược đầu tư"],
            y=comparison_table["Độ lệch chuẩn"]*100,
            name="Độ lệch chuẩn (%)",
            marker_color='#10B981',
            yaxis='y1'
        ))
        # Đường Sharpe
        fig_dual.add_trace(go.Scatter(
            x=comparison_table["Chiến lược đầu tư"],
            y=comparison_table["Hệ số Sharpe"],
            name="Hệ số Sharpe (Phải)",
            mode='lines+markers',
            line=dict(color='#EF4444', width=3),
            marker=dict(size=10),
            yaxis='y2'
        ))
        
        fig_dual.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=10, b=20),
            height=450,
            xaxis=dict(showgrid=False),
            yaxis=dict(
                title="Giá trị (%)",
                showgrid=True,
                gridcolor='#232D42'
            ),
            yaxis2=dict(
                title="Hệ số Sharpe",
                overlaying='y',
                side='right',
                showgrid=False
            ),
            barmode='group',
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig_dual, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Kết luận nhanh
        best_idx = comparison_table["Hệ số Sharpe"].idxmax()
        best_strategy = comparison_table.loc[best_idx, "Chiến lược đầu tư"]
        st.success(f"🏆 **KẾT LUẬN NHANH:** Chiến lược đầu tư đem lại hiệu suất tối ưu và hệ số Sharpe cao nhất là **{best_strategy}**.")
        
else:
    st.info("Vui lòng tải tệp CSV lên hoặc đợi quá trình tải dữ liệu ngành hoàn tất để bắt đầu.")
