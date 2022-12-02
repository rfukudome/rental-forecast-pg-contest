import pandas as pd 
import os.path as path
import lightgbm as lgb
from datetime import datetime as dt
import numpy as np
from glob import glob
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from mods.pre_data_clean import data_cleansing
from mods.csv_to_pickle import csv_to_pickle


PATH_INPUT_RENT = path.join('data','test-input.csv')
PATH_MODEL_DIR = path.join('model','1202-1117-8671')
PATH_OUTPUT_PREDICT = path.join('output','rent_predict_'+str(dt.now().strftime('%m%d-%H%M'))+'.csv')


print(dt.now(),': 家賃予測を開始します')
#予測データをcsvからpickleへ変換
input_df = pd.read_csv(PATH_INPUT_RENT)
print('現在読み込まれているデータ件数は',len(input_df),'件です')
# xs, ys = data_cleansing(input_df)
xs = data_cleansing(input_df)
# print(xs)
# モデルをロード
model = lgb.CVBooster()
for file in sorted(glob(path.join(PATH_MODEL_DIR, 'model-*.txt'))):
    model.boosters.append(lgb.Booster(model_file=file))

# 予測実行
predict = model.predict(xs)
# ２次元配列を1次元に変換
# pre_y = predict[0]
#  スコアを出力します。(平均平方二乗誤差)
# mse = mean_squared_error(ys, np.mean(predict, axis=0))
# rmse = np.sqrt(mse)
# print(f'rmse: {rmse}, mse: {mse}')

#データ出力
serial_num = pd.RangeIndex(start=0, stop=len(predict[0]), step=1)
predict_data = pd.DataFrame(columns=['ID','家賃 + 管理費・共益費'])
predict_data['ID'] = serial_num
predict_data['家賃 + 管理費・共益費'] = [int(i) for i in predict[0]]
predict_data.to_csv(PATH_OUTPUT_PREDICT,index=False)

print(dt.now(),': 家賃予測を完了しました。')
print('結果は、outputフォルダへ保存済み。')

