import numpy as np
import pandas as pd
import os
import os.path as path
import datetime
import lightgbm as lgb
import matplotlib.pyplot as plot
from sklearn.metrics import mean_squared_error
from glob import glob

PATH_INPUT_FILE = path.join('.\\RentML', 'input', 'rent_geo.pickle')
PATH_MODEL_DIR =  path.join('.\\RentML', 'model', 'prod')

def main():
    print(datetime.datetime.now())

    # 検証用データを取り出し
    df = pd.read_pickle(PATH_INPUT_FILE)
    valid_xs, valid_ys = valid_extract(df,7000 )
    print(valid_xs.info())

    # モデルをロード
    model = lgb.CVBooster()
    for file in sorted(glob(path.join(PATH_MODEL_DIR, 'model-*.txt'))):
        model.boosters.append(lgb.Booster(model_file=file))

    # モデルから予測を実行します
    predict = model.predict(valid_xs)

    #  スコアを出力します。(平均平方二乗誤差)
    mse = mean_squared_error(valid_ys, np.mean(predict, axis=0))
    rmse = np.sqrt(mse)
    print(f'rmse: {rmse}, mse: {mse}')

    print(datetime.datetime.now())



def valid_extract(df, length=0):
    # 教師データとなる行を切り出し
    ys = df['家賃 + 管理費・共益費'].astype(int).copy()

    # 学習につかう列を切り出し
    xs = df[USE_COLUMNS].copy()

    # すべての列をfloatに変換（変換エラーの行はfactorize）
    for name, column in xs.iteritems():
        try:
            xs[name] = column.fillna(0).astype(float)
        except ValueError:
            print(name + ' factorize')
            xs[name] = pd.factorize(column)[0]
            xs[name].astype(float)

    if length <= 0 :
        return xs, yx
    else:
        return xs[:length], ys[:length]

USE_COLUMNS = [\
    '緯度',
    '経度',
    # '建物名',
    '沿線0',
    '沿線1',
    '沿線2',
    '駅0',
    '駅1',
    '駅2',
    '徒歩0',
    '徒歩1',
    '徒歩2',
    '都道府県名',
    '市区町村名',
    '大字町',
    '丁目',
    '築年数',
    '専有面積',
    '向き',
    '階',
    '階建',
    '建物種別',
    '構造',
    '総戸数',
    '台所',
    '駐車場',
    '納戸',
    '部屋数',
    '敷金',
    '礼金',
    'エアコン',
    # '室内洗濯置',
    # 'バストイレ別',
    'バルコニー',
    # 'フローリング',
    # 'シューズボックス',
    # 'クロゼット',
    # 'TVインターホン',
    '温水洗浄便座',
    '都市ガス',
    # '洗面所独立',
    'システムキッチン',
    # '敷地内ごみ置き場',
    '3駅以上利用可',
    #'2駅利用可',
    # '2沿線利用可',
    # '即入居可',
    '角住戸',
    # '礼金不要',
    # '光ファイバー',
    # '駅徒歩10分以内',
    # 'オートロック',
    # '浴室乾燥機',
    # '宅配ボックス',
    # '最上階',
    '陽当り良好',
    # '閑静な住宅地',
    '敷金・礼金不要',
    # 'プロパンガス',
    # 'エレベーター',
    # '南向き',
    # '3沿線以上利用可',
    # '駅まで平坦',
    # 'ロフト',
    'ペット',
    # 'ウォークインクロゼット',
    # '始発駅',
    # 'エアコン全室',
    # 'エアコン2台',
    # 'LDK12畳以上',
    # 'デザイナーズ',
    # '南面リビング',
    # '駐車場1台無',
    # '高層階',
    # '未入居',
    'リノベーション',
    # 'エアコン3台',
]

if __name__ == "__main__":
    main()