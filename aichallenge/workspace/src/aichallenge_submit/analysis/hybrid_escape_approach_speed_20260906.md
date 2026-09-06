# 空間Hybridの微速脱出と接近速度余裕の適合移植

現在のSubmitにOldのhybrid_lateral_escape_creep_allowedとrolling_precommit_speed_marginを移植した。

## 微速脱出

設定はstopped_vehicle_overtake.hybrid_escape_creep_speed=0.6m/s、hybrid_escape_min_lateral_gap=0.10m。

同じ対象・同じ左右レーンのHybridが未完了かつpauseされておらず、現在の適用レーンとも一致する場合に評価する。速度推定が有効で既存の微速対象速度閾値以下、車体非重複、横の車体余裕0.10m以上、対象側の物理通過可、道路幅有効、front/side競合なしを必要とする。前回修正した関連他車の動的安全判定も通すので、近い後方車の速度不明や危険を無視しない。FollowOnlyでは許可しない。Oldと同様、厳格Shadow証明が残っていること自体はこの脱出条件に含めない。

現在のcontrollerに合わせ、対象IDが一致するACCと緊急制動の個別上限にだけ最低0.6m/sを適用し、その後で全体の速度上限とminを取る。全体の速度をmax(0.6)で持ち上げない。既存の厳格Shadow微速についても、全体上限を上書きしていたmaxを同じ車両の緊急上限への適用に変更した。他車が先に処理されても後から処理されても、その制限は残る。

この条件で脱出可能な対象については、停止対象として後退回復を新たに起動しない。他の対象の回復要求は残る。

参照速度だけを変更すると前周期のゼロ速度MPC出力が残り得るため、最終参照上限が確定した後、最大0.6m/sの前進要求も適用する。MPCの正確解または安全確認済みPure Pursuit操舵が必要で、MPC安全回復・Prepass全幅回復・ParallelAbort・後退要求・停止要求等では適用しない。後段の指令制限・無効化・ギア回復処理も従来の順序で動く。

ログ: EmergencyBrakeHybridEscapeCreep / emergency mode=hybrid_lateral_escape_creep。

## 接近速度

config.yamlとcontrollerのデフォルトをOldに合わせた。

| 状態 | 対象速度への余裕 |
|---|---:|
| 確定前、35m以上 | +3.0m/s |
| 確定前、22.5m | +1.9m/s |
| 確定前、10m以下 | +0.8m/s |
| 外側レーン適用後、速度合わせ中 | +2.5m/s |

確定前は基本0.8m/sに最大2.2m/sを加算し、35mから10mへ線形に追加分を減らす。これは速度合わせが必要なときの上限であり、車両最大速度や他の安全上限を超える加速指示ではない。ラッチ距離・Shadow条件・地理的レーン規制は今回変更していない。

## 検証

添付ログ6100行を確認。EmergencyBrakeの0m/s要求は26件あり、対象速度が有効な移動車へのものも含まれる。すべてが今回の脱出条件を満たすとは限らず、今回の変更で全停止が解消するとの判断はしていない。

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPYCACHEPREFIX=/tmp/escape_pycache python3 -m pytest -q -p no:cacheprovider aichallenge_submit/multi_purpose_mpc_ros/test`

404 passed（28件追加）、git diff --check通過。

- 厳格Shadow証明なしのアクティブHybridでも安全条件を満たせば0.6m/sを許可。
- pause/完了/対象変更/適用レーン変更/速度不明/重複/余裕不足/幅不足/後方unknown/front競合/FollowOnlyでは拒否。
- 実controllerの個別緊急上限処理で、他車の0または0.2m/s上限が処理順序にかかわらず残る。
- 実controllerの微速指令処理で、全体上限0・0.2・5m/sに対し、停止出力からそれぞれ0・0.2・0.6m/sを要求する。
- 実controllerの速度余裕選択とconfig.yamlの値、距離ごとの減衰を検証。
- 前段のunknown修正・Shadow・Hybrid・レーン所有権・距離ゲートの回帰も通過。

ROSビルド・AWSIM再走行は未実施。
