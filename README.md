# keras_audio_cnn

```
sudo apt-get install -y python3-sklearn  

python3 -m virtualenv env-keras -p python3
source env-keras/bin/activate
pip freeze > requirements.txtls
deactivate

```

Install sklearn
```
(env) python -m pip install sklearn
```

## localをremoteで強制上書きし、remoteと一致させる方法
以下のコマンドを実行すると、localをremoteで強制上書きし、remoteと一致できる。
```
$ git fetch origin
$ git reset --hard origin/master
```

## 【機械学習】モデル評価・指標についてのまとめと実行( w/Titanicデータセット)
http://qiita.com/kenmatsu4/items/0a862a42ceb178ba7155
