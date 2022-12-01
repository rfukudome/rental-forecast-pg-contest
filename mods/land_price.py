import pandas as pd
import os.path as path
import mods.csv_to_pickle as csv_to_pk
from datetime import datetime as dt

# PATH_LAND_PRICE_CSV = path.join('data','land_price_2022.csv')
PATH_LAND_PRICE_PICKLE = path.join('input','land_price_2022.pickle')

def get_land_price(df):
    df_rent = df.copy()
    # df_land_price = pd.read_csv(PATH_LAND_PRICE_CSV)
    # pk_land_price = csv_to_pk.csv_to_pickle('data','land_price_2022')
    # pd.to_pickle(pk_land_price,PATH_LAND_PRICE_PICKLE)
    df_land_price = pd.read_pickle(PATH_LAND_PRICE_PICKLE)
    df_land_price['地価'] = df_land_price['postedLandPrice']
    df_land_price['都道府県名'] = df_land_price['location'].str.extract('(...??[都道府県])', expand=False)
    df_land_price['市区町村名'] = df_land_price['location'].str.extract('...??[都道府県]((?:東村山市|武蔵村山市|佐波郡玉村町)|.+?郡.+?[町村]|.+?市.+?区|.+?[市区町村])', expand=False)
    df_land_price['市区町村名'] = df_land_price['市区町村名'].apply(lambda s:s.replace('　',''))
    df_land_price['大字町丁目名2'] = df_land_price['location'].str.extract('...??[都道府県](?:(?:東村山市|武蔵村山市|佐波郡玉村町)|.+?郡.+?[町村]|.+?市.+?区|.+?[市区町村])(.+)', expand=False)
    df_land_price['丁目3'] = df_land_price['大字町丁目名2'].str.extract('([０-９]{1,3}丁目)').replace('丁目','', regex=True).fillna('０')
    df_land_price['丁目3'] = df_land_price['丁目3'].str.extract('([０-９]{1,2})$') \
        .apply(lambda s: s.str.translate(str.maketrans('０１２３４５６７８９', '0123456789')))

    df_land_price['大字町3'] = df_land_price['大字町丁目名2'].str.extract('^([^０-９]{1,8})', expand=False)

    for name in ['市区町村名', '大字町3']: df_land_price[name] = df_land_price[name].replace("ヶ", "ケ", regex=True)\
        .fillna('') #naあるとquery落ちて調査時に面倒
    
    df_land_price = drop_needless_colums(df_land_price)
    df_rent = pd.merge(df_rent, df_land_price, left_on=['都道府県名', '市区町村名','大字町', '丁目'], right_on=['都道府県名', '市区町村名', '大字町3', '丁目3'], how='left')
    
    return df_rent

def drop_needless_colums(df_land_price):
    drop_columns = [
        "postedLandPrice","volatilityOverthePreviousyear","cityName","location","address","acreage",\
        "currentUse","usageDescription","usageClassification","buildingStructure","waterFacility","gasFacility",\
        "sewageFacility","configuration","frontageRatio","depthRatio","numberOfFloors","numberOfBasementFloors",\
        "frontalRoad","directionOfFrontalRoad","widthOfFrontalRoad","stationSquareOfFrontalRoad","pavementOfFrontalRoad",\
        "sideRoad","directionOfSideRoad","proximityWithTransportationFacility","surroundingPresentUsage","nameOfNearestStation",\
        "distanceFromStation","useDistrict","fireArea","urbanPlanningArea","altitudeDistrict","forestLaw","parksLaw",\
        "buildingCoverage","floorAreaRatio","extraFloorAreaRatio","commonSurveyedPosition","selectedYear","postedLandPriceOfS58",\
        "postedLandPriceOfS59","postedLandPriceOfS60","postedLandPriceOfS61","postedLandPriceOfS62","postedLandPriceOfS63",\
        "postedLandPriceOfH01","postedLandPriceOfH02","postedLandPriceOfH03","postedLandPriceOfH04","postedLandPriceOfH05",\
        "postedLandPriceOfH06","postedLandPriceOfH07","postedLandPriceOfH08","postedLandPriceOfH09","postedLandPriceOfH10",\
        "postedLandPriceOfH11","postedLandPriceOfH12","postedLandPriceOfH13","postedLandPriceOfH14","postedLandPriceOfH15",\
        "postedLandPriceOfH16","postedLandPriceOfH17","postedLandPriceOfH18","postedLandPriceOfH19","postedLandPriceOfH20",\
        "postedLandPriceOfH21","postedLandPriceOfH22","postedLandPriceOfH23","postedLandPriceOfH24","postedLandPriceOfH25",\
        "postedLandPriceOfH26","postedLandPriceOfH27","postedLandPriceOfH28","postedLandPriceOfH29","postedLandPriceOfH30",\
        "postedLandPriceOfR01","postedLandPriceOfR02","postedLandPriceOfR03","postedLandPriceOfR04","attributeChangeOfS59",\
        "attributeChangeOfS60","attributeChangeOfS61","attributeChangeOfS62","attributeChangeOfS63","attributeChangeOfH01",\
        "attributeChangeOfH02","attributeChangeOfH03","attributeChangeOfH04","attributeChangeOfH05","attributeChangeOfH06",\
        "attributeChangeOfH07","attributeChangeOfH08","attributeChangeOfH09","attributeChangeOfH10","attributeChangeOfH11",\
        "attributeChangeOfH12","attributeChangeOfH13","attributeChangeOfH14","attributeChangeOfH15","attributeChangeOfH16",\
        "attributeChangeOfH17","attributeChangeOfH18","attributeChangeOfH19","attributeChangeOfH20","attributeChangeOfH21",\
        "attributeChangeOfH22","attributeChangeOfH23","attributeChangeOfH24","attributeChangeOfH25","attributeChangeOfH26",\
        "attributeChangeOfH27","attributeChangeOfH28","attributeChangeOfH29","attributeChangeOfH30","attributeChangeOfR01",\
        "attributeChangeOfR02","attributeChangeOfR03","attributeChangeOfR04"\
        ]
    df_land_price.drop(drop_columns, axis=1, inplace=True)

    return df_land_price