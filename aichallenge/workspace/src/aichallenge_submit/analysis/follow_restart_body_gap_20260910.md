# 再発進と車間制御の共通化

対象は13:18:08走行ログの発進遅れ。13:18:29.161にd3へのFollowRestartが成立しても、13:18:30.509には前車速度1.61m/sに対して再び0m/sが要求されていた。再発進・通常ACCの車両位置間距離と、後段の車体間距離による制限が別の速度則を使っていた。

## 変更

- 移動中の同じ追従対象について、既存の車体包絡から得るCenter弧長方向の車体間距離、Center接線方向の前車速度、制動予測を共有する。
- 通常追従、再発進、DynamicGapHazard、同じ対象のEmergencyBrake、低速車追越し準備の制限で共通の観測・速度目標を使用する。別車両の停止要求を上書きしない。
- 反応時間中の自車最大加速と、その後の制動距離を含む速度上限を計算する。前車の将来加速は見込まず、自車と前車が同じ設定減速度で制動するモデルを使う。使用減速度はmoving_emergency_available_decelerationと実際のa_minの小さい側。
- 希望車間3m未満でも、前車が動いていれば制動上限内で追従する。必要制動車間を下回ると、比例制御だけに頼らずa_minを要求する。最終指令にも上限を適用する。
- 前車停止、車体重複、不明な車体形状、古い観測、逆方向・大きな横方向移動には共通の移動車追従を適用しない。既存の停止・衝突処理を維持する。再発進タイマーは対象IDに紐づける。

## 設定

follow_restartの旧min_gap/start_min_gapは車両位置間距離だったため、名前を変更した。

| 設定 | 値 | 意味 |
|---|---:|---|
| minimum_body_gap | 1.0m | 車体間の最低余裕。反応・制動に必要な距離を別途加算 |
| min_body_gap | 3.0m | 通常再発進の車体間距離条件 |
| start_min_body_gap | 1.0m | 初期再発進の車体間距離条件。これだけでは発進せず、共通速度上限も必要 |
| observation_max_age | 0.5s | 共通追従・再発進に使う観測の最大経過時間 |

`initial_start_boost.enabled`はfalseのまま。壁マージン・車線選択設定は今回変更していない。

## 検証

次の5ファイルで271件成功。Python構文、YAML読み込み、git diff --checkも成功。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/follow-mpl \
PYTHONPATH=aichallenge_submit/multi_purpose_mpc_ros:/opt/ros/humble/lib/python3.10/site-packages \
python3 -m pytest -q \
  aichallenge_submit/multi_purpose_mpc_ros/test/test_following.py \
  aichallenge_submit/multi_purpose_mpc_ros/test/test_final_emergency_limit.py \
  aichallenge_submit/multi_purpose_mpc_ros/test/test_stopped_vehicle_safety.py \
  aichallenge_submit/multi_purpose_mpc_ros/test/test_hybrid_integration.py \
  aichallenge_submit/multi_purpose_mpc_ros/test/test_collision_body_pose.py
```

新規テストは記録された発進時の車間・前車速度、実ACC/再発進分岐とEmergency分岐の整合、対象切替、停止優先、鮮度・交差方向、最終速度・加速度制限を確認する。反応遅れ0/0.2/0.4秒で前車が発進後に2.5m/s²で制動する簡易モデルでも、発進と1m以上の余裕を確認した。これはログ全体のリプレイやAWSIMの再現ではない。

AWSIM実走行・ROSビルドは未実施。既存のMPC失敗や他車の割込みによる停止まで解消したとは判断していない。実際の制動性能・遅延・前車減速度がモデルの範囲に入ることも実走行で確認する必要がある。

## 次回走行での確認

再ビルド・再起動して設定と追加モジュールを反映する。`FollowRestart`はbody_gapを表示する。`BodyGapFollow`は対象ID、body_gap、required_gap、lead_speed、safe_limit、target、command、restartを0.5秒間隔で表示する。同じ対象のEmergencyBrakeは`mode=shared_body_gap_follow`となる。targetが正でcommandだけ0の場合は、別車両・MPC回復など残る制限と時刻を照合する。

作業前から複数ファイルに変更があったため、その変更を維持したまま今回の変更を追加した。
