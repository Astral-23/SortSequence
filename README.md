# 概要
input: 整数列  

output: inputをsortした整数列  

を目的としたニューラルネットワークの実装

<br><br>

# ファイル構成

## data
各種データ置き場

### SequenceMaker.py
input/outputを作成するためのSSDクラスの実装

## model
モデル置き場

### {各種モデル}.py
モデルのクラスの実装


## src

### main.py
各種モデルの訓練・計測。

### util.py
main.pyで用いる関数の実装

### params.py
各種定数。エポックサイズや数列のサイズ

<br><br>

# 使うライブラリ
Pytorch
