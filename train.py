
import pickle
import pandas as pd 
from sklearn.linear_model import LinearRegression#線形回帰
from data_clean import data_cleansing


def train(x_data,y_data):
    print('##学習開始')
    x = x_data.values.copy()
    y = y_data.values.copy()
    #予測
    model = LinearRegression()
    model.fit(x, y)
    #モデルを保存
    print('##モデル保存')
    model_name = './model/model.pkl'
    pickle.dump(model,open(model_name,'wb'))
    
# CSV読み込み（DataFrame形式）
data = pd.read_csv('./data/rent7.csv',encoding='utf8')
print('データ件数',len(data))
x_data = data_cleansing(data)
y_data = pd.DataFrame(data['家賃 + 管理費・共益費'],columns=['家賃 + 管理費・共益費'])
#モデル作成
train(x_data,y_data)
#データ出力
# x_data.to_csv('./output_data.csv')



