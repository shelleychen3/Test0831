import yfinance as yf
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
import mplfinance.original_flavor as mpf
from mplfinance.original_flavor import candlestick_ohlc
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.patches as mpatches
matplotlib.rc('font', family='Microsoft JhengHei')
matplotlib.rc('axes', unicode_minus=False)
#plt.rcParams["axes.unicode_minus"] = False
stock_id = '2330.TW'
end_date = datetime.today().date()
start = end_date - timedelta(days=180)
start_date = start - timedelta(days=60)
df = yf.download(stock_id, start=start_date, end=end_date)

if isinstance(df.columns, pd.MultiIndex):
    # 展平欄位名稱
    df.columns = df.columns.get_level_values(0)
# 計算SMA
df['SMA_5'] = df['Close'].rolling(window=5).mean()
df['SMA_10'] = df['Close'].rolling(window=10).mean()
df['SMA_20'] = df['Close'].rolling(window=20).mean()

# 計算布林帶
df['middle_band'] = df['SMA_20']
df['std_dev'] = df['Close'].rolling(window=20).std()
df['upper_band'] = df['middle_band'] + (df['std_dev'] * 2)
df['lower_band'] = df['middle_band'] - (df['std_dev'] * 2)

# 计算RSV
n = 9
low_min = df['Low'].rolling(window=n).min()
high_max = df['High'].rolling(window=n).max()
df['RSV'] = ((df['Close'] - low_min) / (high_max - low_min)) * 100
# # 計算KD線
df['K'] = df['RSV'].ewm(alpha=1/3, adjust=False).mean()
df['D'] = df['K'].ewm(alpha=1/3, adjust=False).mean()
#计算J值,df['3K-2D'] 有些是參考這反指標
#df['3K-2D'] = 3 * df['K'] - 2 * df['D']
df['J'] = 3 * df['D'] - 2 * df['K']
# 計算OBV
df['OBV'] = np.where(df['Close'] > df['Close'].shift(1), df['Volume'], -df['Volume'])
df['OBV'] = df['OBV'].cumsum()
# 計算 MACD
fast_period = 12
slow_period = 26
signal_period = 9
# 直接用ewm計算因有預熱60天資料不需被始化
df['EMA12'] = df['Close'].ewm(span=fast_period, adjust=False).mean()
df['EMA26'] = df['Close'].ewm(span=slow_period, adjust=False).mean()
df['DIF'] = df['EMA12'] - df['EMA26']
df['MACD'] = df['DIF'].ewm(span=signal_period, adjust=False).mean()
df['MACD Histogram'] = df['DIF'] - df['MACD']
#RSI公式
def calculate_yahoo_rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    
    # Yahoo 的靈魂：alpha = 1 / period 且 adjust=False
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# 算出 RSI5 與 RSI10 兩組指標
df['RSI5'] = calculate_yahoo_rsi(df['Close'], period=5)
df['RSI10'] = calculate_yahoo_rsi(df['Close'], period=10)
# 計算兩組乖離率
df['BIAS10'] = ((df['Close'] - df['SMA_10']) / df['SMA_10']) * 100
df['BIAS20'] = ((df['Close'] - df['SMA_20']) / df['SMA_20']) * 100

# 計算柱狀圖數據 (B10 - B20)
df['B10-B20'] = df['BIAS10'] - df['BIAS20']
#強哥範例算完再切因df之後不會再用到,不用先設不同變數df_all
df=df.loc[start:,:].copy()
df.index = df.index.map(lambda x: x.strftime('%y-%m-%d'))
# 繪圖部份
x_ticks_pos = range(0, len(df.index), 15)
x_ticks_labels = df.index[::15]

fig = plt.figure(figsize=(12,10), layout='constrained')
ax1 = fig.add_subplot(8,1,(1,3))
ax1.set_xticks(x_ticks_pos)
ax1.set_xticklabels(x_ticks_labels)
mpf.candlestick2_ochl(ax1, df['Open'], df['Close'], df['High'], df['Low'], width=0.8, colorup='r', colordown='g', alpha=1)
ax1.plot(df['SMA_5'],label='5日均線',alpha=0.9,color='cyan',lw=0.5)
ax1.plot(df['SMA_10'],label='10日均線',alpha=0.9,color='purple',lw=0.5)
ax1.plot(df['SMA_20'],label='20日均線',alpha=0.9,color='orange',lw=0.5)
ax1.plot(df['upper_band'], label='upperband',alpha=0.9,color='g',ls=':')
#ax1.plot(df['middle_band'], label='middleband',alpha=0.9)
ax1.plot(df['lower_band'], label='lowerband',alpha=0.9,color='g',ls=':')
ax1.legend(loc=0)
ax1.set_title("2026歡慶端午2330股市開AI紅盤作實做")
ax2 = fig.add_subplot(8,1,4)
ax2.set_xticks(x_ticks_pos)
ax2.set_xticklabels([])

conditions = [
    df['Close'] > df['Close'].shift(1),  # 漲 -> 紅
    df['Close'] < df['Close'].shift(1)   # 跌 -> 綠
]
choices = ['r', 'g']
colors = np.select(conditions, choices, default='gray')
ax2.plot(df['OBV'],color='purple',linestyle='--',label='OBV')
ax2.legend(loc=1)
ax2_1 = ax2.twinx()
ax2_1.bar(df.index,height=df['Volume'], color=colors, width=0.8, alpha=0.8)
red_patch = mpatches.Patch(color='red', label='紅色漲')
green_patch = mpatches.Patch(color='green', label='綠色跌')
gray_patch = mpatches.Patch(color='gray', label='灰持平')
ax2_1.legend(handles=[red_patch, green_patch,gray_patch],loc=2,title="交易量")
ax3 = fig.add_subplot(8,1,5)
ax3.plot(df['K'], label='K line',color='cyan',lw=0.5)
ax3.plot(df['D'], label='D line',color='purple',lw=0.5)
ax3.plot(df['J'], label='J line',linestyle='--',color='orange')
ax3.set_xticks(x_ticks_pos)
ax3.set_xticklabels(x_ticks_labels)
ax3.legend(loc=0)
ax4 = fig.add_subplot(8,1,6)
ax4.plot(df['DIF'], label='DIF9',color='purple')
ax4.plot(df['MACD'], label='MACD',color='skyblue')
macd_colors = np.where(df['MACD Histogram'] >= 0, 'r', 'g')
ax4.bar(df.index,height=df['MACD Histogram'] ,color=macd_colors,alpha=0.8)
ax4.axhline(0, color='gray', linestyle='--', linewidth=1.2)
ax4.set_xticks(x_ticks_pos)
ax4.set_xticklabels([])
ax4.set_ylim(-100, 100)
macd_red_patch = mpatches.Patch(color='red', label='MACD多頭')
macd_green_patch = mpatches.Patch(color='green', label='MACD空頭')
handles, labels = ax4.get_legend_handles_labels()
handles.extend([macd_red_patch, macd_green_patch])
ax4.legend(handles=handles, loc=2, fontsize=8, framealpha=0.5)
ax5 = fig.add_subplot(8,1,7)
ax5.plot(df['RSI5'], label='RSI5',color='cyan',lw=0.5)
ax5.plot(df['RSI10'], label='RSI10',color='purple',lw=0.5)
ax5.set_xticks(x_ticks_pos)
ax5.set_xticklabels([])
ax5.set_ylim(0, 100)
ax5.axhline(70, color='red', linestyle='--', linewidth=0.8, alpha=0.5) # 超買線
ax5.axhline(30, color='green', linestyle='--', linewidth=0.8, alpha=0.5) # 超賣線
ax5.legend(loc=2)
ax6 = fig.add_subplot(8,1,8)
ax6.plot(df['BIAS10'], label='BIAS10',color='cyan',lw=0.5)
ax6.plot(df['BIAS20'], label='BIAS20',color='purple',lw=0.5)
bias_colors = np.where(df['B10-B20'] >= 0, 'r', 'g')
ax6.bar(df.index,height=df['B10-B20'] ,color=bias_colors,alpha=0.8)
ax6.axhline(0, color='gray', linestyle='--', linewidth=1.2)
max_bias = max(df['B10-B20'].max(), 15)
min_bias = min(df['B10-B20'].min(), -15)
ax6.set_ylim(min_bias * 1.1, max_bias * 1.1)
bias_red_patch = mpatches.Patch(color='red', label='BIAS正強')
bias_green_patch = mpatches.Patch(color='green', label='BIAS負弱')
handles, labels = ax6.get_legend_handles_labels()
handles.extend([bias_red_patch, bias_green_patch])
ax6.set_xticks(x_ticks_pos, labels=x_ticks_labels)
ax6.legend(handles=handles, loc=1, fontsize=8, framealpha=0.5)
plt.show()
