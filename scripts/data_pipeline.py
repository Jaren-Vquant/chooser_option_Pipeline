"""
实时数据拉取模块(用于测试数据管道的实时数据处理能力)
数据源: Yahoo Finance (yfinance) + FRED (pandas_datareader)
"""

"""
原始数据获取模块(起始日是2018-01-01，结束日是最新日期，定期执行获取数据)
获取：JPM股价，VIX指数，国债利率
"""
def fetch_raw_data(FRED_API_KEY: str = "add45cfc6b591adda898f43319ed4619"):
    import yfinance as yf
    import pandas as pd
    from datetime import datetime
    from pathlib import Path

    #输出目录(确保存在)
    OUTPUT_DIR = Path(__file__).resolve().parent.parent / "scripts" / "raw_data"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    #FRED API Key (替换成你自己的)
    API_KEY = FRED_API_KEY

    #日期范围: 固定起始日期, 结束日期自动设为今天
    START_DATE = "2018-01-01"
    END_DATE = datetime.now().strftime("%Y-%m-%d")#自动获取最新日期

    #使用FRED获取国债利率数据
    from fredapi import Fred  
    fred = Fred(api_key=API_KEY)
    """
    对照表:
    DGS3MO = 3个月期国债收益率
    DGS1   = 1年期国债收益率
    DGS10  = 10年期国债收益率
    """
    series_ids = {
        "DGS3MO": "3M_Treasury_Yield",
        "DGS1": "1Y_Treasury_Yield",
        "DGS10": "10Y_Treasury_Yield"
    }
    all_series = {}
    for series_id, label in series_ids.items():
        data = fred.get_series(series_id, observation_start=START_DATE, observation_end=END_DATE)
        all_series[label] = data
        print(f"{series_id}抓取完成，总共{len(data)}个观测值")
    df = pd.DataFrame(all_series)
    df.index.name = "Date"
    df = df.reset_index()
    output_path = OUTPUT_DIR / "国债利率.csv"
    df.to_csv(output_path, index = False)

    #获取JPM股价
    TICKER_1 = "JPM"
    df = yf.download(TICKER_1, start=START_DATE, end=END_DATE, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        df.columns.name = None
    df = df.reset_index()
    output_path = OUTPUT_DIR / "JPM股价.csv"
    df.to_csv(output_path, index = False)

    #获取VIX数据
    TICKER_2 = "^VIX"
    df = yf.download(TICKER_2, start=START_DATE, end=END_DATE, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        df.columns.name = None
    df = df.reset_index()
    output_path = OUTPUT_DIR / "VIX指数.csv"
    df.to_csv(output_path,index = False)


"""
数据清洗和交易日历对齐
输入(读取 fetch_raw_data() 输出的三个 CSV 文件):
    JPM股价.csv
    VIX指数.csv
    国债利率.csv
输出:
    merged_clean.csv(三份数据清洗对齐后合并的结果)
"""
def clean_and_align():
    import pandas as pd
    import numpy as np
    from pathlib import Path

    #路径配置
    RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "scripts" / "raw_data"
    NEW_DATA_DIR = Path(__file__).resolve().parent.parent / "scripts" / "new_data"
    NEW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    JPM_FILE = RAW_DATA_DIR / "JPM股价.csv"
    VIX_FILE = RAW_DATA_DIR / "VIX指数.csv"
    RATE_FILE =  RAW_DATA_DIR / "国债利率.csv"

    #读取数据
    jpm = pd.read_csv(JPM_FILE, parse_dates=["Date"])
    jpm = jpm.rename(columns={
            "Close": "JPM_Close", "High": "JPM_High", "Low": "JPM_Low",
            "Open": "JPM_Open", "Volume": "JPM_Volume"})

    vix = pd.read_csv(VIX_FILE, parse_dates=["Date"])
    vix = vix.loc[:,["Date", "Close"]].rename(columns={"Close": "VIX_Close"})

    rate = pd.read_csv(RATE_FILE, parse_dates=["Date"])

    print(f"  JPM: {len(jpm)} 条")
    print(f"  VIX: {len(vix)} 条")
    print(f"  利率: {len(rate)} 条")

    #交易日历对齐(以JPM的交易日作为基准日历)
    df = pd.merge(jpm, vix, on="Date", how="left")
    df = pd.merge(df, rate, on="Date", how="left")
    print(f"\n合并后行数(以JPM交易日为基准): {len(df)}")
    print("合并后缺失值统计:")
    print(df.isnull().sum())

    #异常值处理(用IQR方法识别异常值,超出[Q1 - k*IQR, Q3 + k*IQR]范围的值替换为NaN(后续会用插值法填补,而不是直接删除整行,避免破坏时间序列连续性))
    def remove_outliers_iqr(series: pd.Series, k: float = 3.0) -> pd.Series:
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1#代表中间部分数据的波动范围
            lower = q1 - k * iqr
            upper = q3 + k * iqr
            return series.where((series > lower) & (series < upper), other = np.nan)
    price_cols = ["JPM_Close", "JPM_High", "JPM_Low", "JPM_Open"]#仅对价格类列做IQR检查
    for col in price_cols:
        returns = df[col].pct_change()#计算日收益率(涨跌幅),用收益率来进行异常值筛选更符合逻辑
        outlier_mask = remove_outliers_iqr(returns).isnull()
        outlier_mask.iloc[0] = False  #第一天没有收益率数据，不算异常值
        before_na = df[col].isnull().sum()
        df.loc[outlier_mask, col] = np.nan  # 把异常波动日对应的原始价格标记为缺失
        after_na = df[col].isnull().sum()
        flagged = after_na - before_na
        print(f"{col}: 识别出 {flagged} 个异常值,已标记为缺失待插值")


    #缺失值插值填补(时间序列数据用线性插值,比直接填均值更合理,能保留趋势连续性)
    numeric_cols = df.columns.drop("Date")#Date 是日期类型，不是用来做插值计算的对象，所以要单独排除掉，只保留那些真正需要插值的数值列（价格、VIX、利率等）
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
    print("\n插值后缺失值统计:")
    print(df.isnull().sum())

    #保存清洗对齐后的结果
    path =  NEW_DATA_DIR / "merged_clean.csv"
    df.to_csv(path, index=False)
    print(f"\n已保存清洗对齐后的数据: {path}")
    print(f"最终行数: {len(df)}, 时间范围: {df['Date'].min().date()} ~ {df['Date'].max().date()}")


"""
特征工程制作
输入:
    merged_clean.csv (clean_and_align.py 的输出)
输出:
    features.csv (含11个特征的最终结构化数据集)
"""
def feature_engineering():
    import pandas as pd
    import numpy as np
    from pathlib import Path

    #路径配置
    INPUT_FILE = Path(__file__).resolve().parent.parent / "scripts" / "new_data" / "merged_clean.csv"
    OUTPUT_FILE = Path(__file__).resolve().parent.parent / "scripts" / "new_data" / "features.csv"

    #定义日频数据年化波动率(标准是乘以根号252天)
    ANNUALIZATION_FACTOR = np.sqrt(252)

    #读取数据
    df = pd.read_csv(INPUT_FILE, parse_dates=["Date"])

    #JPM股价衍生特征
    #1. 日收益率 (simple return)
    df["JPM_Return"] = df["JPM_Close"].pct_change()
    #2. 对数收益率 (log return) —— 金融建模里比简单收益率更常用,具有可加性
    df["JPM_LogReturn"] = np.log(df["JPM_Close"] / df["JPM_Close"].shift(1))
    #3. 20日滚动波动率(年化) —— 短期历史波动率,可作为BSM模型sigma参数的候选输入
    df["Vol_20D"] = df["JPM_LogReturn"].rolling(window=20).std() * ANNUALIZATION_FACTOR
    #4. 60日滚动波动率(年化) —— 中期波动水平,用于对比短期/中期波动率差异
    df["Vol_60D"] = df["JPM_LogReturn"].rolling(window=60).std() * ANNUALIZATION_FACTOR
    #5. 成交量相对变化(20日均值偏离度) —— 反映异常放量/缩量
    df["Volume_Deviation"] = (df["JPM_Volume"] - df["JPM_Volume"].rolling(20).mean()) / df["JPM_Volume"].rolling(20).mean()
    #6. 20日价格动量 —— 过去20个交易日的累计涨跌幅（(当日收盘价 − 20 个交易日之前的收盘价) ÷ 20 个交易日之前的收盘价）
    df["Momentum_20D"] = df["JPM_Close"].pct_change(periods=20)
    
    #VIX衍生特征
    #7. VIX水平(直接保留,已在merged_clean.csv中,这里显式重命名是为了突出其特征身份)
    df["VIX_Level"] = df["VIX_Close"]
    #8. VIX日变化率 —— 恐慌情绪的边际变化
    df["VIX_Change"] = df["VIX_Close"].pct_change()
    
    #跨数据源特征
    #9. VIX与JPM收益率的20日滚动相关性
    df["VIX_JPM_Corr_20D"] = df["JPM_Return"].rolling(window=20).corr(df["VIX_Change"])
    #10. 利率期限利差(10年期 - 3个月期) —— 经典收益率曲线斜率指标
    df["Rate_Spread_10Y_3M"] = df["10Y_Treasury_Yield"] - df["3M_Treasury_Yield"]
    #11. 10年期利率5日动量(环比变化)
    df["Rate_10Y_Momentum_5D"] = df["10Y_Treasury_Yield"].diff(periods=5)
    
    #保存完整数据集(保留原始列 + 新特征列)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n已保存最终特征数据集: {OUTPUT_FILE}")
    print(f"总行数: {len(df)}, 总列数: {len(df.columns)}")

     
if __name__ == "__main__":
    fetch_raw_data()
    clean_and_align()
    feature_engineering()


   
