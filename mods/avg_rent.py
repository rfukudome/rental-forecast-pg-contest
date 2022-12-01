import pandas as pd
import os.path as path
from datetime import datetime as dt

PATH_AVG_RENT_CSV = path.join('data','avg_rent.csv')

def get_avg_rent(df):
    df_rent = df.copy()    
    df_avg = pd.read_csv(PATH_AVG_RENT_CSV).drop({'1部屋前年同月比','2部屋前年同月比','3部屋前年同月比','総平均前年同月比','総平均賃料','東京100%'},axis=1)
    marge_df_avg = pd.merge(df_rent, df_avg, left_on=['都道府県名'], right_on=['都道府県名'], how='inner')
    df_avg_rent = marge_df_avg.apply(make_df_avg_rent,axis=1)
    df_avg_rent.drop({'1部屋','2部屋','3部屋'},axis=1)
    return df_avg_rent

def make_df_avg_rent(df_avg_rent):
    if df_avg_rent['部屋数'] == 1:
        df_avg_rent['平均賃貸金額'] = df_avg_rent['1部屋']
    elif df_avg_rent['部屋数'] == 2:
        df_avg_rent['平均賃貸金額'] = df_avg_rent['2部屋']
    else:
        df_avg_rent['平均賃貸金額'] = df_avg_rent['3部屋']   
    return df_avg_rent