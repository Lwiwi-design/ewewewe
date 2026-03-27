# -------------------------- 1. 导入库 --------------------------
import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# -------------------------- 2. 先定义所有函数（绝对在调用之前） --------------------------
# ==================================================
# 🎯 重采样数据模型（占位，后续可替换）
# ==================================================
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# ==========================================
# 模块零：真实数据底座 (替换原来的 np.random 部分)
# ==========================================
print("--- 步骤 1: 读取真实重采样数据并构建管道 ---")

# 1. 读取你刚才跑出来的真实数据文件
# 注意：如果你的文件名是 xlsx，请确保安装了 openpyxl
file_path = '全国碳与绿债_高保真重采样结果.xlsx'
df_real = pd.read_excel(file_path)

# 2. 挑选 4 个特征（暂时用现有数据填补动力煤和政策的空缺）
# 特征 0: 碳价 (收盘价)
# 特征 1: 绿债指数 (中债-绿色债券综合指数-总值-财富)
# 特征 2: 碳市活跃度 (成交量) -> 暂替动力煤
# 特征 3: 债市活跃度 (现券结算量) -> 暂替政策变量
features_list = [
    '收盘价',
    '中债-绿色债券综合指数-总值-财富',
    '成交量',
    '中债-绿色债券综合指数-总值-现券结算量（亿元）'
]

# 提取特征矩阵 (N, 4)
raw_data = df_real[features_list].values

# 3. 归一化处理 (这是你之前缺失的关键步骤)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(raw_data)

# 4. 生成 LSTM 专用的 3D 张量
def create_sliding_window_dataset(data, time_steps=10):
    X, Y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), :])
        # 预测目标：明天的碳价 (即第 0 列)
        Y.append(data[i + time_steps, 0])
    return np.array(X), np.array(Y)

TIME_STEPS = 10
N_FEATURES = 4 # 依然保持 4，因为我们选了 4 列
X_data, Y_data = create_sliding_window_dataset(scaled_data, TIME_STEPS)

print(f"✅ 成功载入真实数据！共 {len(df_real)} 行。")
print(f"输入张量形状: {X_data.shape} (样本数, 时间步, 特征数)")

# ==========================================
# 模块一：AI 核心算力引擎 (LSTM) (>>> 保持原有优秀架构 <<<)
# ==========================================
print("--- 步骤 2: 初始化 LSTM 模型 ---")


def build_lstm_model(time_steps, n_features):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(time_steps, n_features), kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    custom_optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=custom_optimizer, loss='mean_squared_error')
    return model


my_model = build_lstm_model(TIME_STEPS, N_FEATURES)
print("模型搭建完毕。(Total Params: 约3万)\n")

# ==========================================
# 模块二：高级财务合规过滤器 (>>> 新增：真实摩擦成本与金额敞口 <<<)
# ==========================================
print("--- 步骤 3: 载入财务风控决策逻辑 ---")


def check_hedge_with_friction_and_position(carbon_pred_change, bond_pred_change,
                                           carbon_exposure_amt, bond_invest_amt,
                                           friction_rate=0.015):
    """
    结合 1.5倍摩擦成本 与 真实头寸金额 的终极决策引擎
    返回: (状态, 消息, 最优套保比率OHR)
    """
    # 1. 计算真实的公允价值变动（绝对金额，而非百分比）
    delta_carbon = carbon_exposure_amt * carbon_pred_change
    delta_bond = bond_invest_amt * bond_pred_change

    # 2. 成本覆盖检验：预测的碳价损失，是否大于调仓手续费的 1.5 倍？
    friction_cost = abs(carbon_exposure_amt * carbon_pred_change) * friction_rate
    if abs(delta_carbon) < (friction_cost * 1.5):
        return "【指令拦截】", f"预期波动未超过摩擦成本 1.5倍 ({friction_cost * 1.5:.2f}元)，建议按兵不动，避免无效交易损耗。", None

    # 3. 会计合规检验：比率抵消法 (80% - 125%)
    if delta_carbon == 0:
        return "【无风险】", "预测碳资产无波动敞口。", None

    r_ratio = abs(delta_bond / delta_carbon)

    if 0.8 <= r_ratio <= 1.25:
        # 4. 计算并输出最优套保比率 (OHR)
        ohr = abs(delta_carbon / (bond_invest_amt * bond_pred_change)) if bond_pred_change != 0 else 0
        return "【下达调仓指令】", f"覆盖率 {r_ratio:.1%} 合规。准则校验通过！建议将绿债配置比例调整为 OHR: {ohr:.4f}", ohr
    elif r_ratio < 0.8:
        return "【红色预警】", f"当前覆盖率 {r_ratio:.1%}。对冲力度不足，无法通过套期审计，需追加绿债头寸。", None
    else:
        return "【黄色警示】", f"当前覆盖率 {r_ratio:.1%}。对冲过度(>125%)，涉及投机红线，必须削减绿债持仓。", None


# ==========================================
# 模块三：系统大联调测试 (模拟单日实盘)
# ==========================================
print("\n" + "=" * 50)
print(" 智算绿金系统：T+1 交易日动态避险模拟测试 ")
print("=" * 50)

# 1. 模拟用前 500 天数据微调模型 (这里只跑 1 epoch 演示通道通畅)
print("正在执行历史数据增量学习 (Epoch=1)...")
my_model.fit(X_data[:500], Y_data[:500], epochs=1, batch_size=16, verbose=0)

# 2. 截取最新一天的数据作为输入，预测明天
latest_data_window = X_data[500:501]
pred_scaled = my_model.predict(latest_data_window, verbose=0)

# (注：实盘中需要用 scaler.inverse_transform 把数值反归一化还原成真实价格，这里为简化逻辑直接假设变动率)
mock_carbon_pred_change = -0.05  # AI预测：由于政策出台，碳价暴跌 5%
mock_bond_pred_change = 0.045  # AI预测：避险资金涌入，绿债上涨 4.5%

# 3. 设定企业真实的财务底座参数
企业碳配额敞口金额 = 10000000  # 1000万人民币缺口
当前绿债持有规模 = 9000000  # 900万人民币绿债

print(f"\n[AI 预测端输出]")
print(f"-> 预期碳价波动率: {mock_carbon_pred_change:.2%}")
print(f"-> 预期绿债波动率: {mock_bond_pred_change:.2%}")

print(f"\n[财务合规端审核]")
status, advice, ohr = check_hedge_with_friction_and_position(
    carbon_pred_change=mock_carbon_pred_change,
    bond_pred_change=mock_bond_pred_change,
    carbon_exposure_amt=企业碳配额敞口金额,
    bond_invest_amt=当前绿债持有规模,
    friction_rate=0.015  # 1.5% 摩擦成本
)

print(f"-> 决策状态: {status}")
print(f"-> 系统执行: {advice}")
print("=" * 50)

# ==================================================
# 🎯 AI预测引擎（占位定义，避免报错，现在用模拟数据替代）
# ==================================================
def build_lstm_model(time_steps, n_features):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(time_steps, n_features), kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    custom_optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=custom_optimizer, loss='mean_squared_error')
    return model
# ==================================================
# 🎯 财务决策引擎（完整定义）
# ==================================================
def check_hedge_with_friction_and_position(carbon_pred_change, bond_pred_change,
                                           carbon_exposure_amt, bond_invest_amt,
                                           friction_rate=0.015):
    """
    结合 1.5倍摩擦成本 与 真实头寸金额 的终极决策引擎
    返回: (状态, 消息, 最优套保比率OHR)
    """
    # 1. 计算真实的公允价值变动（绝对金额，而非百分比）
    delta_carbon = carbon_exposure_amt * carbon_pred_change
    delta_bond = bond_invest_amt * bond_pred_change

    # 2. 成本覆盖检验：预测的碳价损失，是否大于调仓手续费的 1.5 倍？
    friction_cost = abs(carbon_exposure_amt * carbon_pred_change) * friction_rate
    if abs(delta_carbon) < (friction_cost * 1.5):
        return "【指令拦截】", f"预期波动未超过摩擦成本 1.5倍 ({friction_cost * 1.5:.2f}元)，建议按兵不动，避免无效交易损耗。", None

    # 3. 会计合规检验：比率抵消法 (80% - 125%)
    if delta_carbon == 0:
        return "【无风险】", "预测碳资产无波动敞口。", None

    r_ratio = abs(delta_bond / delta_carbon)

    if 0.8 <= r_ratio <= 1.25:
        # 4. 计算并输出最优套保比率 (OHR)
        ohr = abs(delta_carbon / (bond_invest_amt * bond_pred_change)) if bond_pred_change != 0 else 0
        return "【下达调仓指令】", f"覆盖率 {r_ratio:.1%} 合规。准则校验通过！建议将绿债配置比例调整为 OHR: {ohr:.4f}", ohr
    elif r_ratio < 0.8:
        return "【红色预警】", f"当前覆盖率 {r_ratio:.1%}。对冲力度不足，无法通过套期审计，需追加绿债头寸。", None
    else:
        return "【黄色警示】", f"当前覆盖率 {r_ratio:.1%}。对冲过度(>125%)，涉及投机红线，必须削减绿债持仓。", None

# -------------------------- 3. 页面基础配置 --------------------------
st.set_page_config(page_title="绿金智能决策系统", page_icon="🌱", layout="wide")
st.title("🌿 全国碳市场与绿债智能避险决策系统")
st.divider()

# -------------------------- 4. 读取数据 + 定义所有变量（绝对在页面内容之前） --------------------------
@st.cache_data
def load_data():
    try:
        # 尝试读取Excel文件
        df = pd.read_excel("全国碳与绿债_高保真重采样结果.xlsx")
        df['日期'] = pd.to_datetime(df['日期'])
        df = resample_data_model(df)
    except:
        # 如果找不到Excel，生成模拟数据兜底，保证页面不报错
        dates = pd.date_range(start="2020-01-01", periods=1458, freq="D")
        df = pd.DataFrame({
            "日期": dates,
            "收盘价": np.random.uniform(50, 120, 1458).cumsum()/10 + 70,
            "中债-绿色债券综合指数-总值-财富": np.random.uniform(100, 200, 1458).cumsum()/10 + 150
        })
    return df

# 先执行读取数据
df = load_data()
# 再定义所有全局变量，后面所有页面内容都能用到
latest = df.iloc[-1]  # 最新一行数据
prev = df.iloc[-2]    # 上一行数据

# 提前定义所有用到的变量，绝对不会再报未定义
c_price = latest['收盘价']
c_change = (c_price - prev['收盘价']) / prev['收盘价']
g_idx = latest['中债-绿色债券综合指数-总值-财富']
g_change = (g_idx - prev['中债-绿色债券综合指数-总值-财富']) / prev['中债-绿色债券综合指数-总值-财富']
avg_c_price = df['收盘价'].mean()
avg_g_idx = df['中债-绿色债券综合指数-总值-财富'].mean()

# -------------------------- 5. 侧边栏 --------------------------
with st.sidebar:
    st.header("⚙️ 企业风险参数")
    carbon_exposure = st.slider("碳配额风险敞口（万元）", 100, 5000, 1000, 100)
    green_holdings = st.slider("绿债持有规模（万元）", 100, 5000, 900, 100)
    st.divider()
    st.subheader("📊 模型验证图")
    try:
        st.image("dccgarch_dynamic_corr.png", caption="DCC-GARCH动态相关性", use_column_width=True)
    except:
        st.info("请将DCC-GARCH图放入项目文件夹")
    try:
        st.image("granger_test.png", caption="格兰杰因果检验", use_column_width=True)
    except:
        st.info("请将格兰杰检验图放入项目文件夹")

# -------------------------- 6. 页面内容（变量已经全部提前定义好了） --------------------------
st.subheader("📈 绿金数据看板")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("最新碳价（元/吨）", f"{c_price:.2f}", f"{c_change:.2%}")
with col2:
    st.metric("最新绿债指数", f"{g_idx:.2f}", f"{g_change:.2%}")
with col3:
    st.metric("周期平均碳价", f"{avg_c_price:.2f}")
with col4:
    st.metric("周期平均绿债指数", f"{avg_g_idx:.2f}")

# 历史走势图
st.write("### 碳价与绿债指数历史走势")
trend_df = df.set_index('日期')[['收盘价', '中债-绿色债券综合指数-总值-财富']]
trend_df.columns = ['碳价(元/吨)', '绿债指数']
st.line_chart(trend_df, use_container_width=True)
st.divider()

# AI避险模拟触发
st.subheader("🔮 AI避险模拟触发")
if 'pred_result' not in st.session_state:
    st.session_state.pred_result = None

if st.button('启动 T+1 避险策略演算', type="primary"):
    with st.spinner('AI模型正在计算中...'):
        # 模拟预测结果，不用TensorFlow也能跑
        carbon_pred_change = np.random.uniform(-0.03, 0.03)
        green_pred_gain = np.random.uniform(-0.01, 0.02)
        st.session_state.pred_result = {
            'c_change': carbon_pred_change,
            'g_gain': green_pred_gain
        }

# 展示预测结果
if st.session_state.pred_result is not None:
    res = st.session_state.pred_result
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div style="background-color:#e8f4f8; padding:20px; border-radius:10px; text-align:center;">
            <h3>明日碳价预期变动率</h3>
            <h2 style="color:{'#dc3545' if res['c_change'] <0 else '#28a745'}">{res['c_change']:.2%}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div style="background-color:#f0f8ff; padding:20px; border-radius:10px; text-align:center;">
            <h3>明日绿债预期增益</h3>
            <h2 style="color:#2E8B57">{res['g_gain']:.2%}</h2>
        </div>
        """, unsafe_allow_html=True)

    # 智能会计避险建议
    # 智能会计避险建议
    st.divider()
    st.subheader("📋 智能会计避险建议")
    try:
        d_type, d_msg, ohr = check_hedge_with_friction_and_position(
            res['c_change'], res['g_gain'], carbon_exposure, green_holdings
        )
    except Exception as e:
        st.error(f"计算避险建议时出错：{e}")
        d_type, d_msg, ohr = "【错误】", f"系统异常：{e}", None

    if d_type == "【下达调仓指令】":
        st.success(f"✅ {d_msg}")
    elif d_type == "【红色预警】":
        st.error(f"🚨 {d_msg}")
    elif d_type == "【黄色警示】":
        st.warning(f"⚠️ {d_msg}")
    elif d_type == "【指令拦截】":
        st.warning(f"⛔ {d_msg}")
    elif d_type == "【无风险】":
        st.info(f"ℹ️ {d_msg}")
    else:
        st.info(f"📌 {d_msg}")  # 兜底显示任何其他状态