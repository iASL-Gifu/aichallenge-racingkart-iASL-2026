**Shadow分離・Hybrid移植の実装記録（2026-09-06）**

現在のsubmitへ調査候補1・2を実装。oldの速度調停・RecoverySession・並走side lock・追い越し規制緩和は移植していない。

**変更内容**

- `mpc_controller.py` の `_probe_corridor()` で経路本体とWaypointをcopy、境界セルをdeepcopyする。追い越しShadowとRace復帰probeの両方に適用。例外時もモデルの参照経路とcurrent_waypointを本番経路へ戻す。読み取りに使うマップは共有する。
- Shadow実行時はHybridの参照・人工境界weightsを解除し、従来の厳格な候補レーン検証を使う。
- `overtake_session.py` に対象・選択側・コミット・Shadow証拠・Hybrid進捗を集約。既存controllerの対応するprivate変数を置換し、二重の状態を持たせない。
- 現在のレーン要求選択と安全側の上書き順序を残し、最終適用を `_apply_lane_decision()` へ集約。同じ対象・同じ側の一時全幅化では開始点・進捗・期限・クールダウン時刻を保持し、実行可能性の証拠は破棄する。
- L1復帰、安全回復、地理的制限、初期L0保持、ParallelAbortがHybridより優先する。古い外側レーン監視がL1所有中に割り込まない条件も追加。初回の非追い越しL0保持は従来通り即時適用する。
- 距離に対する五次曲線で横方向参照を作り、N+1点の参照と対応するN点の人工境界を同期。移行距離は8〜20m、2m/s以下の開始では8m。前回の正確な同一対象・同一レーンの予測だけを、ずれ0.75m以内なら30%混合する。
- WP間移動は前後の符号を付けて積算する。後退や周回境界の逆行を「一周分進んだ」と扱わない。時間経過だけでは距離進捗を増やさない。8秒の設定は初期遷移監視窓であり、距離未達の参照をタイムアウトだけで外側中心へ飛ばさない。
- 対象変更・喪失・追い越し完了では旧証拠・再開資格を破棄し、対象固有の速度制限を解除する。実際の通過不能・他車危険による解除ではHybrid開始点も破棄する。
- MPCの目的参照が適用済みレーン目標より優先されるのは、Hybrid参照と人工境界weightsが同時にある場合だけ。通常のL1・初期発進・オフセットの優先順位は従来通り。

壁余白0.6m、既存の外側レーン目標位置、物理壁・障害物層、追い越し開始距離、速度、操舵設定、禁止区間の値は変更していない。config.yamlへ追加した11項目はHybrid専用。`hybrid_overtake_enabled: false` では従来の時間ベース参照へ戻せる。

**検証**

パッケージの全テスト **291 passed**（新規73件）。状態・呼び出し箇所のテストは実controllerメソッド/条件をASTから取り出して実行し、ROSの周辺をstub化している。別途、設定済みCenter/Race経路と実MPC/OSQPを使用して次を検証した。

- 両probe呼び出しの成功・例外で本番Waypoint・境界・レーン指定を変更しない。
- ShadowにHybridの緩やかな人工境界を持ち込まない。
- Hybridの目的参照が実際のOSQP目的関数に入り、weightsを外すと従来のハードレーン目標へ戻る。
- 人工境界の収縮で物理境界が変化しない。
- 停止、周回境界、後退、全幅化からの再開、対象/左右変更、完了、初回L0保持、L1優先、クールダウンを検証。

前回調査で失敗した既存2テストの期待値も修正した。壁余白は実装通り0.6m、0.5秒で1.1m移動する他車速度は2.2m/s。走行側の値は変更していない。構文チェック・`git diff --check` も成功。

**コピー時間**

Python 3.13.13、実設定のCenter 354WP / Race 345WP、各条件10回warmup後500回。mapロード・MPC solve・ROSは計測対象外。測定はpytest終了後に単独実行した。

| 経路・境界キャッシュ | 中央値 | p95 | 最大 |
|---|---:|---:|---:|
| Center・初期 | 0.396ms | 0.408ms | 0.449ms |
| Center・solve後 | 0.389ms | 0.403ms | 0.439ms |
| Race・初期 | 0.384ms | 0.397ms | 0.645ms |
| Race・solve後 | 0.381ms | 0.395ms | 0.458ms |

詳細とcontrollerのソースhashは `shadow_probe_benchmark_20260906.json`。40Hz全周期の負荷保証ではない。

再実行は `aichallenge_submit/multi_purpose_mpc_ros` から：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPYCACHEPREFIX=/tmp/submit_hybrid_pycache python3 -m pytest -q -p no:cacheprovider test
MPLCONFIGDIR=/tmp/submit_hybrid_mpl PYTHONPYCACHEPREFIX=/tmp/submit_hybrid_pycache python3 -m test.benchmark_probe_corridor
```

ROSノード全体の起動・colconビルド・AWSIM走行は未実施。実車指令は送っていない。次の走行ではHybridの開始/中断/再開/完了ログ、低速車に対する実際の横移動、L1/Race復帰、周期時間を確認する。
