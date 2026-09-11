# 詳細処理時間ログ

既存の `[ControlTiming]` に `detail_functions` を追加。既定5秒間隔で集計し、
関数呼び出しごとのログ出力や外部プロセス起動はしない。
`mpc.runtime_diagnostics_enabled: false` で詳細計測も無効になる。

各項目はその報告期間内の:

- calls: 呼び出し回数（キャッシュヒット・早期returnも含む）
- wall_total_ms / cpu_total_ms: 経過時間 / 制御スレッドCPU時間の合計
- wall_max_ms / cpu_max_ms: 1呼び出しの最大時間
- exceptions: 例外終了回数。元の例外はそのまま再送出

controller.*: 通過幅、相対位置、順位提案、committed laneの交通確認、車体重なり、
MPC/Center予測衝突、停止車通過の連続安全確認、Pure Pursuit経路確認、Hybrid参照、車体表示。

live_mpc / probe_mpc._init_problem: 問題構築全体（再試行ごとに計数）。
prepare.reference / linearize / boundaries / matrix / constraint_vectors / cost_vectors /
optimizer_setup_update: 問題構築の7段階。完了した問題構築について記録。
update_prediction: 予測位置への変換。求解時間は従来の live_mpc_solve_wall_ms 等で確認。

**入れ子の時間は内包時間**。例えば _init_problem は prepare.* の時間を含む。
controllerの親関数も内部で計測した子関数を含むため、全項目を単純合計しない。
平均1回時間は total/calls、周期当たり時間は total/counts.cycles から計算できる。
経過時間とCPU時間の差はscheduler/GIL/I/O/lock等を含み、GPU負荷を直接表さない。

計測は制御スレッドが活動中の呼び出しに限定。別スレッドのcallbackは混入させない。
関数名は固定集合で、集計データは報告時にクリアする。
走行条件・速度・境界・解の受理条件は変更していない。
次回は通常どおり起動し、ControlTimingのdetail_functionsを含むログを取得する。

## 通過幅評価の内訳

`detail_functions` の `vehicle_passage.*` を追加:

- `cache_key`: 境界全体等を含むキー生成
- `cache_lookup`: 辞書検索・ヒット時の戻り値コピー（キーのhash/比較を含む）
- `target_clearance`: 他車予測点と境界の通過余裕計算
- `horizon_widths`: 自車ホライズンの車線幅収集
- `width_decision`: L0/L2の幅判定
- `diagnostic_format`: 失敗理由・診断文字列の生成
- `diagnostic_log`: logger.info呼び出し。throttleによる抑制時も計数
- `cache_store`: 距離計算・結果保存（キーのhashを含む）

`counts.vehicle_passage_cache_hits` / `vehicle_passage_recomputes` は同じ報告期間内の回数。
width_decision/diagnostic_*は再計算1回につき原則2回（L0/L2）。
対象なし・観測なしは内訳計測前にreturnし、親の呼び出し回数だけに含まれる。
親の`controller._vehicle_passage`はこの内訳を含む。計測処理自体の時間もあるため、
内訳合計と親の時間は完全には一致しない。判定条件・ログのthrottle間隔は変更なし。

## 診断ログ削減

`mpc.passage_diagnostics_enabled: false`を既定とする。True時は
`passage_diagnostics_interval_sec: 5.0`（最小1秒）の事前ゲートを通った
対象車のみ、左右の詳細を出力。抑制中も幅判定・結果保存は行う。
Off/抑制中はdiagnostic_format/diagnostic_logの項目が現れない。
TrafficLaneProposalも1秒の事前ゲートを適用。旧MPCの累積計測printは削除し、
ControlTimingへ集約。安全警告・状態変更・復旧ログは維持。
