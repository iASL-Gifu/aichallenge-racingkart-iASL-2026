# Center射影の計算再利用（2026-09-08）

複数台を考慮するレーン提案と対象選択の分離を維持し、Controller の
`_center_frenet` を固定経路用の prepared projector へ接続した。

- 全354区間の始点・ベクトル・長さ・累積距離を起動時に準備。
- 全区間をNumPyで比較。局所探索やWPの間引きはせず、同距離は元と同じ最初の区間を選ぶ。
- 座標が完全一致する結果だけを最大2048件のLRUで再利用。丸めなし。
- 経路データはコピーして所有。ControllerのCenter射影用座標・累積距離もtuple化。
  現在この形状は起動時に一度構築され、レーン・境界更新では変化しない。
  将来経路を動的交換する場合はprojectorも再構築すること。
- 他車の通過可否、MPC解、衝突確認結果、レーン判断自体はキャッシュしない。
- 既存ControlTimingのcontextに `center_projection_hits_total` / `center_projection_misses_total`
  を追加。起動後の累積値なので、隣接ログの差分で再利用率を確認できる。

## 単体計測

`OPENBLAS_NUM_THREADS=1 python aichallenge_submit/multi_purpose_mpc_ros/scripts/benchmark_center_projection.py`

354点の実経路周辺354座標、各3反復の中央値（1射影当たり）:

|方式|時間|
|---|---:|
|従来のPython走査|74.39 µs|
|事前計算・全区間ベクトル比較、キャッシュなし|10.32 µs|
|同一座標の再利用|0.94 µs|

比較座標の最大絶対誤差は0。計測値は `center_projection_benchmark.json`。
制御ループ全体やAWSIMでの改善率を示す数値ではない。
次回走行は同じ台数・条件でControlTimingのlane_decision、safety_fallback、
全周期時間と射影cacheの累積差分を比較する。

## 検証

実経路・WP230〜280周辺、頂点、同距離の交差、退化区間、非有限入力、
完全一致キー、LRU上限、コピー所有、対象変更時の再利用を8テストで確認。
全体594 passed / 1 failed。既存の失敗は `test_outer_course_margin_remains_point_six_metres`:
現設定0.7mに対してテストが0.6mを期待している。今回マージン変更なし。
AWSIM走行は未確認。
