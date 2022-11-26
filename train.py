
import pickle
import pandas as pd 
import os.path as path
import lightgbm as lgb
import matplotlib.pyplot as plot
import datetime.datetime as datetime
from logging import getLogger
from mods.data_clean import data_cleansing
from mods.csv_to_pickle import csv_to_picle

# トレーニング開始
logger = getLogger(__name__) 
PATH_MODEL_DIR = './model'
INPUT_DIR = 'data'
INPUT_FILE_NAME = 'rent'
# 学習データ読み込み（DataFrame形式）
# csvからpickle変換
INPUT_DF = csv_to_picle(INPUT_DIR,INPUT_FILE_NAME)
# train
print('現在読み込まれているデータ件数は',len(INPUT_DF),'件です')
x_train, y_train, x_vaild, y_vaild = data_cleansing(INPUT_DF)
# LightGBMモデル作成
logger.info('LightGBMにてモデルを作成します')
params = {'objective': 'regression'}
cv_result = lgb.cv(
    params,
    lgb.Dataset(train_x, label=train_y),
    return_cvbooster=True,
    )
model = cv_result['cvbooster']
logger.info('モデルの作成が完了しました')
# 学習曲線を出力します。
logger.info('学習曲線を出力・保存します')
PATH_SAVE_TRAIN_PLOT = path.join('output','train-plot'+datetime.now+'png')
plot.plot(cv_result['l2-mean'])
plot.show()
plot.savefig(PATH_SAVE_TRAIN_PLOT)

# 重要な特徴量を出力します。
logger.info('重要な特徴量を出力します')
result_pdc = pd.DataFrame({
    'feature': model.boosters[0].feature_name(),
    'importance': np.mean(model.feature_importance(), axis=0)
    })\
    .sort_values('importance', ascending=False).head(20)

print(result_pdc)

logger.info('検証データから予測を実施します')
# モデルから予測を実行します
predict = model.predict(valid_x)

#  スコアを出力します。(平均平方二乗誤差)
mse = mean_squared_error(valid_y, np.mean(predict, axis=0))
rmse = np.sqrt(mse)
print(f'rmse: {rmse}, mse: {mse}')

# 交差検証(CV)で作成された5つのモデルを別名保存（日付とスコアを付与したフォルダ名を作成）
logger.info('モデルを保存します')
date_str = datetime.now().strftime('%m%d-%H%M')
model_dir = path.join(PATH_MODEL_DIR, f'{date_str}-{str(int(rmse))}')
os.makedirs(model_dir, exist_ok=True)
for i, booster in enumerate(model.boosters):
    booster.save_model(path.join(model_dir, f'model-{i}.txt'))

logger.info('トレーニング終了しました')


