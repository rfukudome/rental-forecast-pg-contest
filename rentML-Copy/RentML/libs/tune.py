import optuna.integration.lightgbm as lgb
import numpy as np
import pandas as pd
import os.path as path
import pickle


def tune_params(df_train, df_valid, path):
    # データセットを取得します。
    xs = df_train
    ys = df_valid

    # LightGBMのパラメーターを作成します。
    params = {
        'objective': 'regression',
        'metric': 'mse',
    }

    # ハイパー・パラメーター・チューニングをします。
    tuner = lgb.LightGBMTunerCV(params, lgb.Dataset(xs, label=ys), return_cvbooster=True, optuna_seed=0)
    cv_result = tuner.run()
    model = tuner.get_best_booster()

    # 重要な特徴量を出力します。
    print(pd.DataFrame({'feature': model.boosters[0].feature_name(), 'importance': np.mean(model.feature_importance(), axis=0)}).sort_values('importance', ascending=False).head(n=20))

    # LightGBMのパラメーターを保存します。
    with open(path, mode='wb') as f:
        pickle.dump(tuner.best_params, f)

def load_params(path):
    try:
        with open(path, mode='rb') as f:
            return pickle.load(f)
    except:
        print('file not found: ', path)
        return None
