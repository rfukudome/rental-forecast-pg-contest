import pandas as pd 
import os.path as path
import lightgbm as lgb
import matplotlib.pyplot as plot
from datetime import datetime as dt
import numpy as np
import os
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from mods.data_clean import data_cleansing


# トレーニング開始
PATH_MODEL_DIR = './model'
INPUT_DIR = 'data'
INPUT_FILE_NAME = 'rent'
# 学習データ読み込み（DataFrame形式）
# csvからpickle変換
# input_df = csv_to_picle(INPUT_DIR,INPUT_FILE_NAME)
PATH_INPUT_RENT = path.join('input',INPUT_FILE_NAME+'.pickle')
input_df = pd.read_pickle(PATH_INPUT_RENT)
# input_df = input_df.head(1000)
# train 
print('現在読み込まれているデータ件数は',len(input_df),'件です')
xs, ys = data_cleansing(input_df)
#説明変数データCSV出力
xs.head(1000).to_csv('./data/tarin_x.csv')

# 学習データと検証データに行を分離（先頭7000を検証データとして使用）
valid_size = 7000 if len(xs) > 70000 else int(len(xs)*0.1)
valid_x = xs[:valid_size]
valid_y = ys[:valid_size]
train_x = xs[valid_size:]
train_y =  ys[valid_size:]

# LightGBMモデル作成'%m%d-%H:%M'
print(dt.now(),': LightGBMにてモデルを作成します')
params = {
    'objective': 'regression',
    'metric': 'mse',
    }
cv_result = lgb.cv(
    params,
    lgb.Dataset(train_x, label=train_y),
    num_boost_round=7000,
    # verbose_eval=1000,
    return_cvbooster=True,
    )
model = cv_result['cvbooster']
print(dt.now(),': モデルの作成が完了しました')
# 学習曲線を出力します。
print(dt.now(),': 学習曲線を出力・保存します')
PATH_SAVE_TRAIN_PLOT = './output/train-plot-'+str(dt.now().strftime('%m%d-%H%M'))+'.jpg'
plot.plot(cv_result['l2-mean'])
plot.savefig(PATH_SAVE_TRAIN_PLOT)

# 重要な特徴量を出力します。
result_pdc = pd.DataFrame({
    'feature': model.boosters[0].feature_name(),
    'importance': np.mean(model.feature_importance(), axis=0)
    })\
    .sort_values('importance', ascending=False).head(30)

print(dt.now(),': 重要な特徴量を出力します(上位30件)\n', result_pdc)

print(dt.now(),': 検証データから予測を実施します')
# モデルから予測を実行します
predict = model.predict(valid_x)

# 決定係数（R2値の判断目安
# 0.6以下モデルとして意味ない
# 0.8以上モデルとして完成度が高い
# 0.9以上過学習の可能性あり
pre_y = predict[0]+predict[1]+predict[2]+predict[3]+predict[4]
pre_y_int = list(map(lambda x: int(x),pre_y))
print(dt.now(),': 評価指数 R2スコアの結果')
print(r2_score(valid_y, pre_y_int))
# print(dt.now(),': 正答率のスコア結果')
# print(accuracy_score(valid_y, pre_y_int))


#  スコアを出力します。(平均平方二乗誤差)
mse = mean_squared_error(valid_y, np.mean(predict, axis=0))
rmse = np.sqrt(mse)
print(f'rmse: {rmse}, mse: {mse}')

# 交差検証(CV)で作成された5つのモデルを別名保存（日付とスコアを付与したフォルダ名を作成）
print(dt.now(),': モデルを保存します')
date_str = dt.now().strftime('%m%d-%H%M')
model_dir = path.join(PATH_MODEL_DIR, f'{date_str}-{str(int(rmse))}')
os.makedirs(model_dir, exist_ok=True)
for i, booster in enumerate(model.boosters):
    booster.save_model(path.join(model_dir, f'model-{i}.txt'))

print(dt.now(),': トレーニング終了しました')


