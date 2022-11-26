import pandas as pd
from logging import getLogger

def  data_cleansing(df):
    logger = getLogger(__name__) 
    logger.info('データクレンジング開始')
    #不要なデータを削除
    df = df.dropna() #NON値の行を削除
    df = df.drop_duplicates() #重複行を削除
    #不要要素を削除
    df_rent = df.copy()
    df_stations = df_rent['駅徒歩'].str.split('\n', expand=True)
    for name, station_info in df_stations.iteritems():
        i = str(name)
        df_rent['沿線'+ i] = station_info.str.split('/', expand=True)[0].fillna('')
        df_rent['駅'+ i]   = station_info.str.split('/| ', expand=True)[1].fillna('')
        df_rent['徒歩'+ i] = station_info.str.extract('歩([0-9]+)分', expand=False).fillna(50).astype(float)

    # 単位がついてるだけの列は数字部分だけ抽出
    for name in ['専有面積', '築年数', '総戸数', '階']:
        df_rent[name] = df_rent[name].str.extract('([0-9]+)', expand=False).fillna(0).astype(float)

    # 部屋数は数字、LDKはSを除いてラベル化
    df_rent['間取り'].replace('ワンルーム', '1', inplace=True)
    df_rent['台所']   = df_rent['間取り'].str.extract('([LDK]+)', expand=False)\
        .map({'K':1, 'LK':2, 'DK':3, 'LDK':4,}).fillna(0)
    df_rent['納戸'] = df_rent['間取り'].apply(lambda s: 1 if 'S' in s else 0).astype(int)
    df_rent['部屋数'] = df_rent['間取り'].str.extract('([0-9]+)', expand=False).fillna(0).astype(int)


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
    
    df_train = df_rent.copy()
    # 教師データとなる行を切り出し
    ys = df_train['家賃 + 管理費・共益費'].astype(int).copy()
    # 学習につかう列を切り出し
    xs = df_train[[
        # '緯度',
        # '経度',
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
    valid_x = xs[:valid_size]
    valid_y = ys[:valid_size]
    train_x = xs[valid_size:]
    train_y =  ys[valid_size:]

    print(train_xs.info())

    logger.info('データクレンジング完了')

    return train_x, train_y, valid_x, valid_y




