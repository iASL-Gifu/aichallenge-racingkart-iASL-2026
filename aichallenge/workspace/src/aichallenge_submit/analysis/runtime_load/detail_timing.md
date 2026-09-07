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
