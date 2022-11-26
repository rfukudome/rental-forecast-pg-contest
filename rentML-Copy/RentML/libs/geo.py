#%%
# /*** ライブラリ読み込み ***/
#
import numpy as np
import pandas as pd
import os
import os.path as path
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plot
import japanize_matplotlib
from sklearn.metrics import mean_squared_error
from libs.tune import tune_params, load_params
import datetime

PATH_GEO_CSV = path.join('input', 'geolonia.csv') # 住所+緯度経度データ


def get_geo_location(df):
    df_rent = df.copy()
    df_geo = pd.read_csv(PATH_GEO_CSV).query('8 <= 都道府県コード <= 14')
    # 小字単位の粒度なので大字単位にまとめる
    df_geo = df_geo.groupby(['都道府県名','市区町村名', '大字町丁目名', '大字町丁目名ローマ字']).mean(['緯度','経度']).reset_index()

    # 丁目の数値部分を抽出（半角で取るためローマ字行から）
    df_geo['丁目2'] = df_geo['大字町丁目名ローマ字'].str.extract(' ([0-9]+)$', expand=False)
    # 丁目部分を除いて大字部分だけを抽出
    df_geo['大字町2'] = df_geo['大字町丁目名'].replace('[一二三四五六七八九十]+丁目$', '', regex=True)\
    # 表記揺れを統一
    for name in ['市区町村名', '大字町2']:
        df_geo[name] = df_geo[name].replace("ヶ", "ケ", regex=True).fillna('')

    # マージ実行
    df_rent_geo = pd.merge(df_rent, df_geo, left_on=['都道府県名', '市区町村名', '大字町', '丁目'], right_on=['都道府県名', '市区町村名', '大字町2', '丁目2'], how='left')

    return df_rent_geo[['緯度', '経度']]

