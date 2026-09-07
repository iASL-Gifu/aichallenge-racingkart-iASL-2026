# 負荷・MPC失敗の切り分けログ

2026-09-08の005741実行ログでは、間引きされたMPC計測66件の合計時間は中央値9.9ms、95パーセンタイル23.225ms、最大41.9ms。構築時間は中央値8.0ms、最大33.8ms。設定周期25msを超える処理がある。一方で`primal infeasible`による復旧もあり、GPU使用率やCPU負荷だけから全ての失敗原因を説明することはできない。

事前生成L0参照は今回もインストール先のPythonファイルが見つからず無効。これは以前の実装の配置検証不備。計測追加と同時に経路の挙動を変更しないため、この修正では有効化しない。

## 自動で出るログ

再ビルド・再起動後、各車両の通常ログに出る。

- `[RuntimeIdentity]`：実際に読み込んだcontroller/MPC/計測モジュールと設定ファイルのパス・SHA256、PID、CPU数、BLAS/OMP関連のスレッド設定。コードと設定が同一かを比較する。車両位置や乱数状態の同一性を保証するものではない。
- `[ControlTiming]`：壁時計5秒ごとの集計。全制御周期、失敗、途中return、試算を含む。各metricは件数・中央値・95パーセンタイル・最大値。サンプルは項目ごとに最大1024件、通常40Hz×5秒なら200件程度。超過時は直近1024件の分布となり、countsは期間全体の件数。

| 項目 | 読み方 |
| --- | --- |
| `cycle_wall_ms` | 制御呼び出し開始から次の開始までの実時間。目標は25ms。ログ出力時間も次周期へ含まれる |
| `rate_wait_wall_ms` | ROS Rate.sleepで待った実時間。通常の周期調整待ちを計算遅延と区別する |
| `work_wall_ms` | 周期待ちを除く制御全体の実時間 |
| `work_thread_cpu_ms` | 制御スレッド自身がCPUを使った時間。BLAS等の別スレッド分は含まない |
| `non_thread_wall_ms` | work_wall − thread_cpu。スケジューラ待ち、GIL、I/O、ロック等を含み、CPU競合だけの時間ではない |
| `live_mpc_*` / `probe_mpc_*` | 本番と試算を分けたMPC全体時間・CPU時間・呼び出し数・結果・予測再利用数。共有MPCでも経路コピー上の試算はprobeとして扱う |
| `*_solve_wall_ms` / `*_solve_thread_cpu_ms` | OSQP.solve一回ごとの時間。再試行を含む |
| `*_iterations` / `*_solver:<status>` | ソルバー反復回数と結果。infeasible・不正確解も含む |
| `*_non_solver_wall_ms` | MPC全体からsolve時間を除いた時間。構築・境界更新・予測処理等 |
| `non_mpc_work_wall_ms` | 制御全体からRate待ちと全MPC呼び出しを除いた時間 |
| `command_gap_wall_ms` | 通常のコマンドpublish間隔。復旧等が直接publishする経路は対象外 |
| `command_stamp_age_sim_ms` | 通常コマンドのstampからpublish完了付近までのsim時間。制御処理中に指令時刻が古くなる量を見る |
| `process_cpu_pct` | ROS制御プロセス全スレッドのCPU時間／壁時間。1コア100%換算で100%を超える場合がある。GPU使用率ではない |
| `sim_wall_ratio` | 制御開始点間のsim時間／壁時間。瞬時の/clock更新間隔の影響があるため分布で見る。逆戻りは別カウント |
| `work_over_period` | 周期待ちを除く制御処理が25msを超えた周期数 |
| `fallback_owned_cycles` | 周期終了時にFallbackが操舵を所有していた周期数。時間割合ではない |
| `context` | 出力時点のWP・レーン・使用経路・L0事前参照の有効状態。期間内全ての周期が同じ状態という意味ではない |

`work_wall`と`thread_cpu`がともに増え、反復数も増えるならソルバーの計算量増加を疑う。実時間だけが増える場合はホストの競合・待ちを合わせて見る。`probe_mpc_calls`が増えていれば試算反復の負荷、`non_mpc_work`が増えていれば境界・追い越し判断等、MPC外の処理を絞り込む。短時間でもinfeasibleになる場合、まず制約・姿勢・境界の既存ログと照合する。

`mpc.runtime_diagnostics_enabled: false`で周期計測を無効化できる。出力間隔は`runtime_diagnostics_interval_sec`（既定5秒、最小1秒）。操舵・速度・境界・復旧の条件は変更していない。

## GPU/ホストCPUの記録

**AWSIMを動かしているホストで**、リポジトリルートから別ターミナルで実行する。コンテナ内ではホスト側のAWSIMやCPUプロセスが見えない場合がある。

```bash
python3 aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros/scripts/record_host_load.py \
  --output /tmp/awsim-load.jsonl --duration 600 --interval 5
```

約5秒間隔（GPU問い合わせ時間が加わる）でUTC時刻、CPU上位15プロセスのPID・名前・CPU%・RSS、loadavg、CPU/メモリpressure、GPU使用率・メモリ・クロック・温度・電力、nvidia-smi pmonのプロセス情報をJSONLへ追記する。プロセスの引数や環境変数は保存しない。GPU機種・ドライバによってpmonのグラフィックスプロセス情報が得られない場合は、その出力自体を保存する。

GPU問い合わせは制御プロセスから実行しない。GPU取得不能時もエラーを記録してCPU記録を継続する。こちらの環境のsmoke testではNVIDIAドライバにアクセスできず、取得失敗を正しく記録した。実ホストのGPU取得成功までは確認していない。

次回は各車のControlTimingを含むログと、このJSONLを同じ走行について保存する。実行途中の再ビルドや経路生成・テストを避けた基準走行と比較すると、GPU/CPU負荷と制御遅延の時刻を照合できる。

## 検証

既存スイート578件通過、作業前からある外周マージン期待値不一致が1件失敗（現在0.7m、期待0.6m）。追加の計測テスト7件通過。周期待ちの除外、集計リセット、時計逆戻り、失敗ソルバー記録、返り値と例外の保持、制御の途中return/例外の計測、ファイルハッシュを確認した。AWSIM実走行での追加負荷は未測定。
