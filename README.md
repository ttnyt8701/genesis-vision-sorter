# genesis-vision-sorter

Genesis 上の Franka Panda が、真上のカメラ画像から赤・緑・青のブロックの位置と向きを CNN で推定し、同じ色のトレイへ仕分けます。黄色のブロックは対象外で、触らずに残します。

## 構成

```
sorter/
  scene.py        シーン（Franka、ブロック、トレイ、カメラ）
  controller.py   IK によるピック&プレース
  model.py        CubeNet（ブロック検出用 CNN）
  perception.py   画像 → ロボット座標への変換
  runner.py       撮影 → 認識 → 1 個運ぶ → 撮り直し のループ
  baseline.py     比較用の色しきい値ルール
  overlay.py      動画の画面合成
scripts/
  gen_data.py         学習データの生成
  train.py            学習
  eval_perception.py  CNN と色しきい値の比較
  eval_sorting.py     仕分けの成功率
  make_demo.py        デモ動画の作成
models/cubenet.pt     学習済みの重み
results/              評価結果
```

## 動かし方

CPU だけで動きます（GPU なしの Ubuntu 24.04 で確認）。画面のない環境では EGL を使うので `PYOPENGL_PLATFORM=egl` を付けます。

```sh
pip install -r requirements.txt
export PYOPENGL_PLATFORM=egl

python scripts/gen_data.py --n 4000 --out data/train.npz --seed 0
python scripts/gen_data.py --n 500 --out data/val.npz --seed 1
python scripts/train.py --epochs 18
python scripts/eval_perception.py
python scripts/eval_sorting.py --episodes 20
python scripts/make_demo.py --out video/demo.mp4
```

学習済みの重みを同梱しているので、評価とデモだけならデータ生成と学習は飛ばせます（`eval_perception.py` は検証データが必要）。手元の CPU 4 コアでは、データ生成が約 7 分、学習が約 10 分でした。

## しくみ

1. アームをカメラに写らない位置へ逃がし、真上カメラ（160×160）で撮影する
2. CubeNet が 1/4 解像度のマップで、色ごとの中心ヒートマップ・中心のずれ・向き（cos 4θ, sin 4θ）を出す
3. ヒートマップのピークを取り出し、視線とブロック上面の高さの平面との交点からロボット座標を求める
4. ロボットに近いブロックから 1 個だけ運び、待機姿勢に戻ってもう一度撮影する

学習データはシミュレータで作っています。ブロックの数・位置・向き・黄色の妨害物をランダムに変え、カメラの取り付け位置も少しずらしながら 4000 枚撮影し、正解は投影計算で自動で付けました。学習時は反転・90° 回転・明るさ・ノイズで水増ししています。

## 結果

検証画像 500 枚（約 1500 個）での検出の再現率です。中心が 4 画素以内で色も合っていれば正解としています。

| 条件 | CNN | 色しきい値 |
| --- | --- | --- |
| 標準 | 100% | 100% |
| 暗い (×0.6) | 100% | 0% |
| 明るい (×1.3+20) | 100% | 49% |
| ノイズ (σ=15) | 100% | 100% |
| 色かぶり（青み） | 100% | 100% |

標準の照明では色しきい値でも十分ですが、明るさが変わるとしきい値を決め直す必要があります。CNN は学習時の明るさの水増しだけで、どの条件でも検出できました。

CNN の推定だけで仕分けを最後まで行った結果、ランダム配置 20 回（ブロック計 86 個）で 86 個すべてを正しいトレイに入れられました。余計な把持はありませんでした。ただし 1 回だけ、近くの黄色ブロックにアームが当たって 1cm 以上動かしています。

## 動画

YouTube: https://youtu.be/WZXg5mbChSY
