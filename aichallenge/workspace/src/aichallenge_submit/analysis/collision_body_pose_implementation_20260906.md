# 衝突用車体姿勢の共通化（2026-09-06）

## 実装

`collision_geometry.py`へBodyPose・BodyGeometryと共通交差計算を追加した。map上の車体中心、yaw、位置観測時刻、yaw観測時刻、取得元、軸方向／前後方向の有効性、位置有効性、原点種別、中心補正量、不確実幅を保持する。

現在SAT、committed lane hold、直線MPC予測照合、Center経路予測、停止車群の非重複判定を共通化。車体寸法の既定値はvehicle_info.param.yamlに基づく全長2.064m・全幅1.45m。車輪間隔1.087mを車体全長としては使わない。現在車体の交差と、追加余裕を含めた交差を同じ世界座標の形状で評価し、Center arc_gap/lateral_gapは経路上の診断値として別記する。

停止相手の向きをCenter接線へ置き換える経路を除去。V2Xの有効な移動から得た軸方向を速度ゼロ補間とは別に保持する。位置飛び、逆行時刻、1秒超の観測不連続、フレーム変更、過大速度、ID消失、clearで無効化する。snapshotは方向・原点情報・測定姿勢も一緒にコピーする。

速度方向だけでは車両の前後を判別できないため、`direction_valid=False`を保持する。向きが実測済みなら、後退で速度が反転しても車体方向を180度反転させない。Center予測も符号付きの進行速度を使い、実／保持yawを基準に経路の曲がり分だけ回転する。停止中は回転させない。

初回から停止していて方向不明なら、架空の確定yawは与えない。全方向を包む外接円と不確実性で可能衝突を評価する。位置やframeが無効・古い場合はNone（判定不能）であり、衝突なしの証明や前進例外には使わない。

## 実車yawの入力

既存V2XVehiclePosition.msgの通信形式は変更していない。AWSIM送信元のC#ソースはこのワークスペースに存在せず、送信側の実装・動作確認はできていない。

任意入力として各車両の次トピックを購読する。
- `/v2x/d1/body_pose`〜`/v2x/d4/body_pose`
- 型: `geometry_msgs/msg/PoseStamped`
- header.frame_id: `map`
- position: **車体中心**の世界位置
- orientation: 実際の車体方向の正規化Quaternion
- stamp: V2Xと同じシミュレーション時計による観測時刻

V2X位置との観測時刻差0.2秒以内、現在から既定0.5秒以内の測定を優先する。後輪軸位置をそのままこのトピックへ送ってはいけない。他車の情報が必要な自車ROS_DOMAIN_ID側にもトピックを配信する必要がある。自車の測定中心が入力された場合は、現在GNSS/Odom位置との対応を求めて現在姿勢とMPC予測の両方へ同じ変換を適用する。

## 原点設定と未確認事項

`config/config.yaml`の`collision_geometry`:
- `length: 2.064`, `width: 1.45`
- `rear_axle_to_center: 0.522`
- `ego_position_origin: unconfirmed`
- `v2x_position_origin: unconfirmed`
- `max_observation_age: 0.5`

原点は`center`、`rear_axle`、`unconfirmed`を想定。GNSSとV2Xで独立設定する。AWSIMの配信原点はメッセージ定義だけでは確定しないため、一律+0.522m補正は行わない。

`rear_axle`と確認され、前後方向も有効な場合だけ、yaw方向へ0.522m移動して中心化する。軸方向のみ、または原点未確認では、未確認の前後中心差0.522mを不確実幅として残す。V2Xの位置標準偏差があればその2倍も加える。**原点未確認の初期設定では従来より保守的になる場合がある**。推奨原点値を推測でcenterにして停止を解消する変更はしていない。

自車はGNSS位置とOdom yawのframe・鮮度・観測時刻差も検査する。map以外の入力を黙ってmapとして使わず、実測topicか配信側でのTF変換が必要。

## 診断

`/mpc/collision_bodies`（MarkerArray）を追加。既存autoware.rvizのV2XグループへCollisionBodies表示を追加した。
- 既知yaw: SATで用いる矩形。原点・位置不確実性があれば拡張した形状。
- 不明yaw: 外接円。
- 各車のID、yaw取得元、原点、中心補正、不確実幅、位置観測経過をラベル表示。
- `[CollisionBodyPair]`で位置・yaw・両観測時刻・frame・有効性を出力。
- 従来EmergencyBrakeの重なりログにも`overlap_kind=body/possible`と`yaw_known`を追加。可能衝突を実測矩形の交差と区別する。

実際の車体とこの表示を比較すれば、向き・原点のどちらが食い違っているか確認できる。初期停止車の実yawを位置メッセージだけから復元することはできない。

## 検証

パッケージ回帰テスト494件成功。停止中の方向保持、snapshot独立性、各種履歴無効化、方向不明の保守的判定、後退方向と中心補正、実yawの優先、鮮度/frame拒否、現在／2種類の予測判定の一致、RViz形状の一致、自車中心変換の一度だけの適用を検証。

実行: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPYCACHEPREFIX=/tmp/group_cache python -m pytest -q -p no:cacheprovider aichallenge_submit/multi_purpose_mpc_ros/test`

ROSビルド、実トピック受信、AWSIM実走行は未実施。誤停止が解消したことの実走行確認は未完了。既存の比較用`unsafe_static_fallback_on_narrow`は今回の対象外として変更していない。
