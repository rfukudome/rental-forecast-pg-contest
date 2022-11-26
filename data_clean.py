import pandas as pd
import re

def walk(item):
    # 徒歩
    walk_pattern = '歩[0-9]+'
    walk_list = re.findall(walk_pattern, item)
    # Castのエラーチェックをしていないのでチェックを実装する必要がある
    walk_list = [int(x.replace("歩", "")) for x in walk_list]
    return walk_list

def bus(item):
    # バス
    bus_pattern = 'バス[1-9]+'
    bus_list = re.findall(bus_pattern, item)
    # Castのエラーチェックをしていないのでチェックを実装する必要がある
    bus_list = [int(x.replace("バス", "")) for x in bus_list]
    return bus_list

def car(item):
    # 車
    car_pattern = '車[1-9]+'
    car_list = re.findall(car_pattern, item)
    # Castのエラーチェックをしていないのでチェックを実装する必要がある
    car_list = [int(x.replace("車", "")) for x in car_list]
    return car_list

def  data_cleansing(data):
    print('##データクレンジング実行')
    #初期化
    new_data = pd.DataFrame(columns=['歩1','歩2','歩3','バス1','バス2','バス3','車1','車2','車3','敷金','礼金','間取り','S','L','D','K','ワンルーム','専有面積','築年数','階','北','南','西','東','マンション','アパート','一戸建て','テラス・タウンハウス','その他'])
    #不要なデータを削除
    data = data.dropna() #NON値の行を削除
    data = data.drop_duplicates() #重複行を削除
    #不要要素を削除
    #data = data.drop('ID','名称','敷金','礼金','所在地','駅徒歩','間取り','専有面積','築年数','階','向き','建物種別','部屋の特徴・設備','間取り詳細','構造','階建','築年月','損保','駐車場','入居','取引態様','条件','総戸数','情報更新日','次回更新日','備考',axis=1)
    data = data.drop(['ID','名称','所在地','部屋の特徴・設備','間取り詳細','構造','階建','築年月','損保','駐車場','入居','取引態様','条件','総戸数','情報更新日','次回更新日','備考'],axis=1)

    #駅徒歩 所在地 駅徒歩   間取り     専有面積   築年数   階  向き   建物種別
    for index in range(len(data)):
        #初期化
        new_data.loc[index] = 0 
        #徒歩
        walk_list = walk(data['駅徒歩'][index])
        if len(walk_list) == 1:
            new_data.loc[index,"歩1"] = walk_list[0]
        elif len(walk_list) == 2:
            new_data.loc[index,"歩1"] = walk_list[0]
            new_data.loc[index,"歩2"] = walk_list[1]            
        elif len(walk_list) == 3:
            new_data.loc[index,"歩1"] = walk_list[0]
            new_data.loc[index,"歩2"] = walk_list[1]
            new_data.loc[index,"歩3"] = walk_list[2]
        
        #バス
        bus_list = bus(data['駅徒歩'][index])
        if len(bus_list) == 1:
            new_data.loc[index,"バス1"] = bus_list[0]
        elif len(bus_list) == 2:
            new_data.loc[index,"バス1"] = bus_list[0]
            new_data.loc[index,"バス2"] = bus_list[1]            
        elif len(bus_list) == 3:
            new_data.loc[index,"バス1"] = bus_list[0]
            new_data.loc[index,"バス2"] = bus_list[1]
            new_data.loc[index,"バス3"] = bus_list[2]
        #車        
        car_list = car(data['駅徒歩'][index])
        if len(car_list) == 1:
            new_data.loc[index,"車1"] = car_list[0]
        elif len(car_list) == 2:
            new_data.loc[index,"車1"] = car_list[0]
            new_data.loc[index,"車2"] = car_list[1]            
        elif len(car_list) == 3:
            new_data.loc[index,"車1"] = car_list[0]
            new_data.loc[index,"車2"] = car_list[1]
            new_data.loc[index,"車3"] = car_list[2]

        #間取り
        if 'S' in data['間取り'][index]:      
            new_data.loc[index,'S'] = 1
        if 'L' in data['間取り'][index] :    
            new_data.loc[index,'L'] = 1 
        if 'D' in data['間取り'][index] :
            new_data.loc[index,'D'] = 1
        if 'K' in data['間取り'][index] :
            new_data.loc[index,'K'] = 1    
        if 'ワンルーム' in data['間取り'][index] :
            new_data.loc[index,'ワンルーム'] = 1
        #間取りの数
        if  'ワンルーム' in  data['間取り'][index] :
            new_data.loc[index,'間取り']= 1
        else:
            num_data = data.loc[index,'間取り']
            num = re.sub("\D", "", num_data)
            new_data.loc[index,'間取り'] = int(num)
        #向き
        if '北' in data['向き'][index] :          
            new_data.loc[index,'北'] = 1
        if '南' in data['向き'][index] :
            new_data.loc[index,'南'] = 1 
        if '西' in data['向き'][index] :
            new_data.loc[index,'西'] = 1
        if '東' in data['向き'][index] :
            new_data.loc[index,'東'] = 1
        #建物種別
        if data['建物種別'][index] == 'アパート':          
            new_data.loc[index,'アパート'] = 1
        elif data['建物種別'][index] == 'マンション':
            new_data.loc[index,'マンション'] = 1 
        elif data['建物種別'][index] == '一戸建て':
            new_data.loc[index,'一戸建て'] = 1
        elif data['建物種別'][index] == 'テラス・タウンハウス':
            new_data.loc[index,'テラス・タウンハウス'] = 1
        elif data['建物種別'][index] == 'その他':
            new_data.loc[index,'その他'] = 1
            
    #専有面積
    data['専有面積'] = data['専有面積'].str.replace('m2','')
    new_data['専有面積'] = data['専有面積']
    #築年数
    data['築年数'] = data['築年数'].str.replace('築','')
    data['築年数'] = data['築年数'].str.replace('新','0')
    data['築年数'] = data['築年数'].str.replace('年','')
    data['築年数'] = data['築年数'].str.replace('以上','')
    new_data['築年数'] = data['築年数']
    #階
    data['階'] = data['階'].str.replace('B','-')
    data['階'] = data['階'].str.replace('M','')
    data['階'] = data['階'].str.replace('階','')
    data['階'] = data['階'].str.replace('-','0')
    new_data['階'] = data['階']
    #敷金・礼金
    new_data['敷金'] = data['敷金']
    new_data['礼金'] = data['礼金']

    print('##データクレンジング完了')

    return new_data




