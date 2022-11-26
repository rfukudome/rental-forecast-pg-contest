import pickle
import pandas as pd 
from data_clean import data_cleansing
from sklearn.metrics import mean_squared_error


def predict(test_data,data):
    print('##予測開始')
    #モデル読み込み
    model_name = './model/model.pkl'
    load_model = pickle.load(open(model_name,'rb'))
    #予測
    predict = load_model.predict(test_data)

    #データ出力
    serial_num = pd.RangeIndex(start=0, stop=len(predict), step=1)
    predict_data = pd.DataFrame(columns=['ID','家賃 + 管理費・共益費'])
    predict_data['ID'] = serial_num
    predict_data['家賃 + 管理費・共益費'] = list(map(int,predict))
    predict_data.to_csv('./data/predict.csv',index=False)

    #作成したモデルを評価
    print('MSE Train : %.3f' % (mean_squared_error(data['家賃 + 管理費・共益費'], predict)))
    ('##予測完了')

#評価データ
data = pd.read_csv('./data/rent.csv')
test_data = data_cleansing(data)
predict(test_data,data)




