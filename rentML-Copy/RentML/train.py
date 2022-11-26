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
from libs.geo import get_geo_location

import datetime

PATH_RENT_FILE = path.join('input', 'rent.pickle') # 学習させる家賃情報
PATH_GEO_CSV = path.join('input', 'geolonia.csv') # 住所+緯度経度データ
PATH_TRAIN_PARAMS = path.join('model', 'params.pickle.trial') # ハイパーパラメーター
PATH_MODEL_DIR = './model'

# 賃貸情報をPICKLEからロード(CSV遅いので変換した)
df = pd.read_pickle(PATH_RENT_FILE)
# 実行結果をアーカイブ
rmse_history=[]

#%%
#   /*** データを正規化 ***/
#
print(datetime.datetime.now())
df_rent = df.copy() # 大元をいじらないよう複製

# 各利用駅の情報を要素毎にパース
df_stations = df_rent['駅徒歩'].str.split('\n', expand=True)
for name, station_info in df_stations.iteritems():
    i = str(name)
    df_rent['沿線'+ i] = station_info.str.split('/', expand=True)[0].fillna('')
    df_rent['駅'+ i]   = station_info.str.split('/| ', expand=True)[1].fillna('')
    df_rent['徒歩'+ i] = station_info.str.extract('歩([0-9]+)分', expand=False).fillna(50).astype(float)
    # df_rent['沿線'+ i].where(df_rent['徒歩'+i] <= 20, None, inplace=True) #徒歩20分以上はないものとする
    # df_rent['駅'+ i].where(df_rent['徒歩'+i] <= 20, None, inplace=True)   #削らないほうがスコア良かった
    # df_rent['徒歩'+ i].where(df_rent['徒歩'+i] <= 20, None, inplace=True)

# 単位がついてるだけの列は数字部分だけ抽出
for name in ['専有面積', '築年数', '総戸数', '階']:
    df_rent[name] = df_rent[name].str.extract('([0-9]+)', expand=False).fillna(0).astype(float)

# 階建は数字複数あるので個別に抽出
df_rent['階建'] = df_rent['階建'].str.extract('([0-9]{1,2})階建', expand=False).fillna(0).astype(float)
# 部屋番除いたスペース手前を抽出
df_rent['建物名'] = df_rent['名称'].str.split(' ', expand=True)[0]


# 部屋数は数字、LDKはSを除いてラベル化
df_rent['間取り'].replace('ワンルーム', '1', inplace=True)
df_rent['台所']   = df_rent['間取り'].str.extract('([LDK]+)', expand=False)\
    .map({'K':1, 'LK':2, 'DK':3, 'LDK':4,}).fillna(0)
df_rent['納戸'] = df_rent['間取り'].apply(lambda s: 1 if 'S' in s else 0).astype(int)
df_rent['部屋数'] = df_rent['間取り'].str.extract('([0-9]+)', expand=False).fillna(0).astype(int)
#df_rent['部屋数'] += df_rent['納戸']
#df_rent['部屋数'] += df_rent['間取り'].apply(lambda ldk: 1 if 'L' in ldk else 0) #Lは1部屋として加算

# 部屋条件から目立ったものをピックアップ
topics = ['エアコン', '室内洗濯置', 'バストイレ別', 'バルコニー', 'フローリング', 'シューズボックス',
'クロゼット', 'TVインターホン', '温水洗浄便座', '都市ガス', '洗面所独立', 'システムキッチン',
'敷地内ごみ置き場', '3駅以上利用可', '2駅利用可', '2沿線利用可', '即入居可', '角住戸', '礼金不要',
'光ファイバー', '駅徒歩10分以内', 'オートロック', '浴室乾燥機', '宅配ボックス',
'最上階', '陽当り良好', '閑静な住宅地', '敷金・礼金不要', 'プロパンガス', 'エレベーター',
'南向き', '3沿線以上利用可', '駅まで平坦', 'ロフト', 'ペット', 'ウォークインクロゼット',
'始発駅', 'エアコン全室','エアコン2台','LDK12畳以上','デザイナーズ','南面リビング','駐車場1台無',
'高層階','未入居', 'リノベーション', 'エアコン3台',]
for topic in topics:
    df_rent[topic] = df_rent['部屋の特徴・設備'].apply(lambda s: 1 if topic in s else 0)

# 駐車場はあるかないかだけ判定（敷地xxxxx円と無料だけは区別）
df_rent['駐車場'] = df_rent['駐車場'].replace('[^敷地無料]+','', regex=True)\
    .map({'':0, '敷地':1, '無料':2})

# 指向性を持たせられるものは数値指定で変換
df_rent['向き'] = df_rent['向き'].map({'-':0, '北':0, '北東':1, '東':2, '南東':3, '南':4, '南西':5, '西':6, '北西':7}).fillna(0)
df_rent['建物種別'] = df_rent['建物種別'].map({ 'その他':0, 'テラス・タウンハウス':1, 'アパート':2,'マンション':3, '一戸建て':4,}).fillna(0)
df_rent['構造'] = df_rent['構造'].map({ 'その他':0, '木造':0, '軽量鉄骨':0, '鉄骨':2, '気泡コン':2, 'ブロック':3, '鉄筋コン':3, '鉄骨鉄筋':3, 'プレコン':3, '鉄骨プレ':3}).fillna(0)


# 住所を分解
df_rent['都道府県名'] = df_rent['所在地'].str.extract('(...??[都道府県])', expand=False)
df_rent['市区町村名'] = df_rent['所在地'].str.extract('...??[都道府県]((?:東村山市|武蔵村山市|佐波郡玉村町)|.+?郡.+?[町村]|.+?市.+?区|.+?[市区町村])', expand=False)
df_rent['大字町丁目'] = df_rent['所在地'].str.extract('...??[都道府県](?:(?:東村山市|武蔵村山市|佐波郡玉村町)|.+?郡.+?[町村]|.+?市.+?区|.+?[市区町村])(.+)', expand=False)
# 末尾の丁目を抽出し半角数字に変換（geoデータとフォーマットと合わせるため）
df_rent['丁目']   = df_rent['大字町丁目'].str.extract('([０-９]{1,2})$') \
    .apply(lambda s: s.str.translate(str.maketrans('０１２３４５６７８９', '0123456789')))
# 末尾の数字を消し込んで大字部分を抽出
df_rent['大字町'] = df_rent['大字町丁目'].replace('[０１２３４５６７８９]{1,2}$', '', regex=True)
# 表記揺れを統一
for name in ['市区町村名', '大字町']: df_rent[name] = df_rent[name].replace("ヶ", "ケ", regex=True)\
    .fillna('') #naあるとquery落ちて調査時に面倒


print(datetime.datetime.now())

#%%
#  /*** geoデータから緯度経度を付与***/
#
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

# ! 緯度経度とれなかったものは一旦除外 (住所パース処理がまだ甘い)
row_count = len(df_rent_geo)
df_rent_geo = df_rent_geo.dropna(subset=['緯度', '経度'])
print(f'{row_count-len(df_rent_geo)} rows drop')

# predict.pyで使う用にエクスポート
pd.to_pickle(df_rent_geo, path.join('./input', 'rent_geo.pickle'))


# %%
# /*** 学習データと検証データを切りだし ***/
#
print(datetime.datetime.now())
df_train = df_rent_geo.copy()

# 教師データとなる行を切り出し
ys = df_train['家賃 + 管理費・共益費'].astype(int).copy()
# 学習につかう列を切り出し
xs = df_train[[
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
]].copy()

# すべての列をfloatに変換（変換エラーの行はfactorize）
for name, column in xs.iteritems():
    try:
        xs[name] = column.fillna(0).astype(float)
    except ValueError:
        print(name + ' factorize')
        xs[name] = pd.factorize(column)[0]
        xs[name].astype(float)


# 学習データと検証データに行を分離（先頭7000を検証データとして使用）
valid_size = 7000 if len(xs) > 70000 else int(len(xs)*0.1)
valid_xs = xs[:valid_size]
valid_ys = ys[:valid_size]
train_xs = xs[valid_size:]
train_ys =  ys[valid_size:]

print(train_xs.info())

print(datetime.datetime.now())


#%%
#   /** 学習を実行し結果を検証 **/
#
params = load_params(PATH_TRAIN_PARAMS) or {'objective': 'regression'}
cv_result = lgb.cv(
    params,
    lgb.Dataset(train_xs, label=train_ys),
    return_cvbooster=True,
    )
model = cv_result['cvbooster']

# 学習曲線を出力します。
plot.plot(cv_result['l2-mean'])
plot.show()

# 重要な特徴量を出力します。
result_pdc = pd.DataFrame({
    'feature': model.boosters[0].feature_name(),
    'importance': np.mean(model.feature_importance(), axis=0)
    })\
    .sort_values('importance', ascending=False).head(20)

print(result_pdc)

# モデルから予測を実行します
predict = model.predict(valid_xs)

#  スコアを出力します。(平均平方二乗誤差)
mse = mean_squared_error(valid_ys, np.mean(predict, axis=0))
rmse = np.sqrt(mse)
print(f'rmse: {rmse}, mse: {mse}')

# 過去3回分のrmseを表示し今回の値を保存
print('history:')
for history in rmse_history[:3]:
    print(history)
rmse_history.insert(0, rmse)

# 交差検証(CV)で作成された5つのモデルを別名保存（日付とスコアを付与したフォルダ名を作成）
date_str = datetime.datetime.now().strftime('%m%d-%H%M')
model_dir = path.join(PATH_MODEL_DIR, f'{date_str}-{str(int(rmse))}')
os.makedirs(model_dir, exist_ok=True)
for i, booster in enumerate(model.boosters):
    booster.save_model(path.join(model_dir, f'model-{i}.txt'))


#%%
#  /*** ハイパーパラメーターを導出 ***/
#
print(datetime.datetime.now())
#tune_params(xs, ys, PATH_TRAIN_PARAMS)
print(datetime.datetime.now())

# %%
df_result = df_train[:valid_size].copy()
df_result["predict"] = np.mean(predict, axis=0)
df_result["answer"] = valid_ys
df_result["gap"] = (df_result["answer"]-df_result["predict"])\
    .apply(lambda value: abs(value))

df_result[['市区町村名', 'gap']]\
    .groupby('市区町村名').mean()\
    .sort_values('gap', ascending=False)

df_result[['建物種別', 'gap']]\
    .groupby('建物種別').mean()\
    .sort_values('gap', ascending=False)

# %%
test = df_result.query("市区町村名=='猿島郡五霞町'").sort_values('gap', ascending=False)


# %%
df_corr = valid_xs.copy()
df_corr["predict"] = np.mean(predict, axis=0)
df_corr["answer"] = valid_ys
df_corr["gap"] = (df_corr["answer"]-df_corr["predict"])\
    .apply(lambda value: abs(value))
df_corr["gap_per"] = (df_corr["gap"]/df_corr["answer"])

#%%
sns.heatmap(df_corr.corr(),
    cmap='bwr',
    vmin=-0.5,
    vmax=0.5,
    annot=True,
    fmt=".2f",
    annot_kws={"fontsize":6},
    )


# %%
c = pd.cut(df_corr['建物種別'], [0,1,2,3,4])
df_corr.groupby(c).mean()['gap_per']


# %%
c = pd.cut(df_corr['専有面積'], 10)
df_corr.groupby(c).mean()['gap_per']






# %%
