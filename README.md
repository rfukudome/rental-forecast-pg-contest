# 賃貸家賃予測プログラム

## 実行環境
　Pythonのバージョンは3.9.13。
　venvにて仮想環境を作成しています。
　下記コマンドにて仮想環境を使用できます。
  ※Linuxの場合
    `.venv/Scripts/activate`
  ※Windwosの場合
    ```
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
    .\.venv\Scripts\Activate.ps1
    ```
　requirements.txtに必要なパッケージ群の情報を記載済み。必要に応じて使用してください。

## プログラム概要
 rental-forecast-pg-contest/
　data/　賃貸予測に必要なCSVファイルを配置
　input/　学習速度を上げるためにcsvからpickleにて変換したデータを配置
　model/　学習した生成されたモデルが保存されるフォルダ
　mods/　学習に必要なデータを整理してくれるモジュール群
　　.avg_rent.py （都道府県の部屋数ごとの平均家賃データを追加するプログラム）
　　.csv_to_pickle.py　（csvをpickleに変換するプログラム）
　　.data_clean.py　（データクレンジングをするプログラム）
　　.geo.py　（都道府県の緯度経度データを追加するプログラム）
　　.land_price.py　（都道府県の地価データを追加するプログラム）
　output/　学習グラフ・家賃予測した結果を保存するフォルダ
　train.py　予測モデルを作成するプログラム
　predict.py　モデルを用いて予測を実施するプログラム
　
## 使用手順（学習）
　1. dataフォルダに下記CSVデータを投入してください。※本来想定されるデータ以外を投入すると予測できないエラーが発生します。
　　・rent.csv（学習用家賃データ）
　　・geolonia.csv（緯度経度データ）
　　・avg_rent.csv（都道府県の部屋ごとの平均家賃）
　　・land_price_2022.csv（都道府県ごとの地価データ）
　2. train.pyを実行。
　3. model/にモデルが出力されます。


## 使用手順（予測）
　1. dataフォルダに予測したいCSVデータを投入してください。※ファイル名は、test-input.csvに固定
　2. predict.pyを実行。
　3. output/pre-rent-price.csvに結果が出力されるため確認してください。
