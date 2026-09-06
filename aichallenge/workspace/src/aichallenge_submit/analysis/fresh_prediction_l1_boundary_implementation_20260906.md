# 残存差分1・2・4の適合実装

対象: 現在のaichallenge_submit。最新外側予測によるCenter停止解除、L1復帰の実距離割込み、境界初期化をまとめて導入した。

## 1. 最新外側予測による対象別の停止解除

Oldの_fresh_outer_prediction_releases_center_stopを抽出し、現在の操舵fallbackと安全回復に合わせた。

現在車体が非重複で横余裕が既存warning閾値以上、対象速度推定が有効、外側レーンの正確なMPC解、fallback・時間超過・安全回復なしを必要とする。その上で、予測点が有限かつ0.5m以上前進し、直線速度予測とCenter沿い予測の両方が安全な場合に、その車のCenterハザード保持だけを解除する。

予測contextは制御周期の入口で破棄し、MPC solve直後に記録する。MPC本体・予測オブジェクト・車線・経路オブジェクト・V2X snapshotが一致しなければ使わない。Pure Pursuit等が操舵を引き継いでいる間も解除しない。他車のハザードや停止制限は残す。

ログ: FreshOuterEmergencyRelease。

## 2. L1復帰中の実距離による再捕捉

Waypointによる前方検出がfalseでも、有効な速度情報で停止/低速と分類された対象が正のCenter実距離かつラッチ距離未満なら、既存のL1割込み候補評価へ入れる。

地理的車線規制、FollowOnly、起動抑制、ParallelAbort、MPC安全回復、Prepass等の既存選択優先順位は維持した。候補の通過幅・交通評価も従来どおり。

入口だけ変えると後段のラッチwriterがWaypoint判定で再び対象を落とすため、ラッチ側の検出条件も有効な低速対象の実距離に対応した。候補そのものは距離・厳格Shadow条件を通るまでNone/L1で保留する既存経路を使う。速度不明を停止車として割り込ませない。

## 4. 境界の生成時確定

ReferencePath生成時に、Waypointへ対応付け済みのCSV境界から外周マージンを一度だけ適用し、静的境界セルと初期動的境界を同期する。CSVとWaypointの点数が違う場合は既存の補間済み値を使う。

現在版のOUTER_COURSE_MARGIN=0.6mを維持した。Oldの0.3mへの変更は行っていない。狭い結果を最低道路幅2.2mまで広げる処理は廃止し、狭窄・反転を既存の制約崩壊判定へ渡す。

marker callbackは幅・動的障害物境界・制約snapshotを変更しない。controllerの地図準備通知・高さ取得・subscription終了は維持する。これによりmarker受信前のShadowも、受信後の本番と同じ静的境界を使う。

## 検証

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPYCACHEPREFIX=/tmp/rem124_cache python3 -m pytest -q -p no:cacheprovider aichallenge_submit/multi_purpose_mpc_ros/test`

437 passed（33件追加）。git diff --check通過。

- 最新予測の安全解除、および停止予測・古いcontext・異なる経路/車線/snapshot・不正確解・速度不明・衝突・操舵fallback等での拒否。
- 実controllerの解除分岐を複数車ループで実行し、安全が証明された車だけハザードが消え、他車は制動評価へ進むことを確認。
- 実controllerのL1割込み条件で、Waypoint検出falseの低速対象を受け入れ、速度不明・後方・距離外・MPC安全回復を拒否。後段の実ラッチ検出条件にも到達することを確認。
- 狭い/反転する幅を広げず、マージンを重複適用しない。markerで動的境界を消さない。
- 実際のRace/Center CSV・地図からMPCを構築し、marker前に全WaypointがCSV−外周余裕と一致し、marker後も本番状態が変わらないことを確認。
- 前段のShadow実OSQP試算、Hybrid境界同期、速度優先順位、unknown修正等も回帰通過。

ROSビルド・AWSIM再走行は未実施。特に狭い場所は最低幅補正をなくしたため制約崩壊が明示される可能性があり、停止が全て減ると保証する変更ではない。追加guardの段階復帰（前回候補3）は今回の対象外。
