# 停止車群の通過と状態管理の整合（2026-09-06）

対象: 現在の aichallenge_submit。既存のShadow検証・最終レーン決定・車両別速度制限へ統合した。Oldの状態管理全体を置き換える移植ではない。

## 1. 停止車へのParallelAbort

幅、車両位置・速度、通過余裕、他車の進入・前後競合、レーン禁止条件を評価し、同じ外側レーンで避けられる停止車群を毎回算出する。対象がこの群に含まれる場合、並走時間だけを理由に譲り待ちへ移らず、既存の通過動作を維持する。ParallelSafetyとEmergencyBrakeの速度制限は引き続き適用する。

既にAbortに入っている場合も、対象が停止していて同じ候補経路に前方の停止車が残るなら、Abortと並走タイマーを解除して通常のShadow取得へ戻す。これは前進許可ではない。新しい候補への進路確定は通常の検証を通す。回復動作中、速度不明、車体重複、幅・通過余裕不足、禁止レーンでは解除しない。

Abortが継続している間は通常の外側レーン選択でShadowを準備しない。直後のAbort処理に毎周期破棄されるProbeHold反復を防ぐ。

## 2. Hybrid参照完了と実車到達

HybridTransitionにreference_completedを追加。移行距離の98%到達は参照完了とし、実車と目的レーン中心の横偏差が既存設定0.45m以内になって初めてcompletedとする。

横偏差が残る間は同じ空間開始点・距離・境界ブレンドを維持する。期限経過や参照の飽和だけで、Hybridの継続判定・低速脱出の資格を消さない。物理的な追い越し完了や別対象・別レーンへの通常切替は、既存の終了・破棄処理に従う。

## 3. d2・d3の共通通過判断

同じ空きレーンで避けられる停止車群を、Abort判定、Hybrid低速脱出、連続追い越しの空間状態引継ぎで共用する。群の評価では本番の交通判定履歴を書き換えない。

各車の横余裕、車体非重複、速度推定、他車競合を個別に満たす場合だけ、その車由来のACC・EmergencyBrake上限に既存の最大0.6m/s脱出を適用する。d2に許可があってもd3の危険を無視しない。全車の上限の最小値を残すため、別車の停止や回復処理が優先される。

対象変更は既存ConsecutiveOvertakeHandoffの厳格Shadow成功分岐に統合した。次対象が共通停止車群に含まれ、同じレーンの新しいShadowが成功した場合だけ、Hybridの空間開始点・進捗・期限を引き継ぐ。旧対象の検証、通過余裕喪失タイマー、交通履歴、追従キャッシュ、動的速度制限、後方完了タイマーを破棄し、既存の分岐が新対象IDと新しい検証を確定する。別の予測判定による対象切替経路は追加していない。

## 検証と限界

回帰テスト: 456件成功。実コントローラのメソッド・制御分岐を実行し、参照完了時の横偏差0.82mでの継続、Strict Shadow引継ぎのID整合と開始点・期限保持、無検証・別レーン・移動車への引継ぎ拒否、幅不足・速度不明・回復中のAbort継続、他車速度上限の処理順独立性を検証した。

実行: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPYCACHEPREFIX=/tmp/group_cache python -m pytest -q -p no:cacheprovider aichallenge_submit/multi_purpose_mpc_ros/test`

ROSビルドとAWSIM実走行は未実施。減速なしの通過や周期計算時間は実走行では未確認。今回の変更は安全条件を満たした動作の不要な解除を防ぐものであり、横余裕不足などで必要な減速まで解除するものではない。

確認ログ:
- `StationaryParallelContinue`: 停止車群のため並走時間によるAbortを回避。
- `StationaryAbortReacquire`: 既存Abortを解除してShadow取得へ戻る。
- `HybridLaneArrivalWait`: 参照は完了したが実車横移動が未完了。
- `HybridOvertakeCompleted`: 参照と実車到達の両方が完了。
- `StationaryGroupHandoff`: 次対象のShadow成功に基づく空間状態引継ぎ。
- `EmergencyBrakeHybridEscapeCreep`の`permitted_ids`: 車両別の低速脱出許可。
