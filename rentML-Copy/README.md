# SUUMO 家賃予測 AI 検証

- プロコン課題を活用した機械学習勉強会のサンプルです
- LightGBM で家賃予測のモデルを作ります
- ダッシュで作ったので推敲もリファクタリングもしてませんがご了承ください
- 無いよりマシくらいの切り張り備忘録とでも思ってください

## Requirement

- miniconda をダウンロードしてインストール
  - CLI はスタートメニューの Anaconda3 から
- 有償規約に抵触しないよう default チャネルから conda-forge に変更

```
> conda config --remove channels defaults
> conda config --append channels conda-forge
```

- (任意)仮想環境を作成し、仮想環境へ切り替え
  - ライブラリのインストールでデフォルト環境を汚したくない人向け

```
> conda create -n Namae python=3.10
> conda activate Namae
```

- ライブラリをインストール

```
> conda install numpy lightgbm pandas matplotlib seaborn optuna funcy jupyterlab
```

- ブラウザでやるなら jupyter を起動

```
> jupyter lab
```

## Feature

- train.py

  - jupyter での実行を想定したモデル生成コード
    - 上からコメント%##の上部「セルを実行」か Shift+Enter でコードブロックを実行します
  - 学習の実行毎に model フォルダに生成モデルを保存します
    - 適宜削除してください
    - 良いモデルができたら model/prod にコピーしておきます
  - ファイル分けると変数表示とかうまくいかなくなるのでこの中で思考錯誤します

- predict.py

  - model/prod のデータを使って検証を行います
  - predict を実行しスコアを出すだけで、プロコン提出向けの加工とかはしてません
  - rentML のルートから「python -m RentML」で呼び出します
  - ハイパーパラメーターは「model/params.pickle」を使います
  - train.py から部分コピペしただけです

- libs/tune.py

  - パイパーパラメータの自動調整をする optuna を実行します
  - あと保存、読み出し用のメソッドとか

- libs/geo.py
  - 住所から緯度経度を呼び出す処理を外部に出してみた
  - 汎用化が面倒なので途中で挫折、現状 train.py の中で完結してる

## License

- 社外秘（社内利用は自由）
- SUUMO のデータは取り扱い注意。禁持出
