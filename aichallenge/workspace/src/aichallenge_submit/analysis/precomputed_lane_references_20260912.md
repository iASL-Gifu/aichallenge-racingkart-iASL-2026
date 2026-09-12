2026-09-13更新: L2の現行参照は `env/centerline/l2_reference_wp190_340.csv`。35 km/h・18度のまま再生成し、MPCのWP190〜340全151地点で正確解を確認した。L0は既存参照を維持。以下は従来参照の生成記録。

事前参照はOldと同じWP区間を使い、現在のSubmitの車線形状・設計条件で検証する。

| 用途 | CSV | WP区間 |
|---|---|---|
| 拡張L0 | `multi_purpose_mpc_ros/env/centerline/l0_reference_wp220_340.csv` | 220〜340 |
| 拡張L2 | `multi_purpose_mpc_ros/env/centerline/l2_reference_wp235_340.csv` | 235〜340 |
| 互換用L0 | `multi_purpose_mpc_ros/env/centerline/l0_reference_wp220_290.csv` | 220〜290 |

各CSVと同名のJSONに設計条件、CSVチェックサム、元ソースのチェックサム、補間後の検証値を保存する。
操舵上限18°、設計速度35km/h、操舵速度上限2.0/1.68 rad/sを使用する。
車体幅は1.6m、前後範囲は原点の違いを包含する[-1.032,+1.554]m。

生成器はOldのL0/L2対応・スプライン補間検証を取り込んだもの。
最適化で補間後の操舵角・操舵速度を制約し、出力前に補間点の車体とCSV壁・占有地図の距離を検査する。
端部3点は現在の車線中点に一致させ、方位・曲率の重みは端部10WPで接続する。
単純なファイルコピーやハッシュの書き換えだけで旧参照を有効化しない。

インストール先のshareディレクトリからCSV・JSON・地図を読み込み、Pythonソースのチェックサムは実際に実行中の`reference_path.py`で検証する。
読み込みが成功すると、MPCの車線横位置・方位・入力曲率の目標へ反映する。Hybridでは混合した横位置列から方位・曲率を求める。
拡張参照の方位ゲインはOldと同じ1.0、互換用L0は0.25。
外側車線の進入可否とShadow検証は従来の判定を通す。MPCの座標系・物理境界・車線境界は変更しない。

パッケージディレクトリ `aichallenge_submit/multi_purpose_mpc_ros` で再生成する。

```bash
OPENBLAS_NUM_THREADS=1 python3 scripts/generate_l0_reference.py --lane 0 --start-wp 220 --end-wp 340
OPENBLAS_NUM_THREADS=1 python3 scripts/generate_l0_reference.py --lane 2 --start-wp 235 --end-wp 340
OPENBLAS_NUM_THREADS=1 python3 scripts/generate_l0_reference.py --lane 0 --start-wp 220 --end-wp 290
```

同じ引数に`--check`を付けると、既存CSVの補間後の幾何検証だけを行う。
既存形状を現在の条件で再検証・再出力する場合は`--revalidate-from /path/to/profile.csv`を使う。
WP区間・端部・車線内位置・操舵・車体掃引のいずれかが不適合なら出力しない。
既存形状を初期値として再最適化する場合は`--initial-reference /path/to/profile.csv`を使う。
これは現在の車線内に初期値を収め、同じ制約付き最適化と出力前検証を行う。
たとえばL2の再最適化は次のように実行できる。

```bash
OPENBLAS_NUM_THREADS=1 python3 scripts/generate_l0_reference.py --lane 2 --start-wp 235 --end-wp 340 --initial-reference env/centerline/l2_reference_wp235_340.csv
```

起動時の確認ログ:

```text
[PrecomputedLane] loaded L0: env/centerline/l0_reference_wp220_340.csv
[PrecomputedLane] loaded L2: env/centerline/l2_reference_wp235_340.csv
```

静的な参照の幾何検証は、走行中のMPC収束や他車を含めた無接触走行を保証するものではない。
ROS再ビルド・ノード再起動・AWSIM走行は別途必要。


今回の生成・検証結果:

| 参照 | 最大操舵角 | 最大操舵速度(rad/s) | CSV壁の最小離隔 | グリッドの最小距離 |
|---|---:|---:|---:|---:|
| l0_reference_wp220_340 | 17.525° | 1.128952 | 0.631m | 0.600m |
| l2_reference_wp235_340 | 17.640° | 1.130952 | 0.369m | 0.412m |
| l0_reference_wp220_290 | 16.959° | 1.130952 | 0.659m | 0.781m |

拡張L0はOldの横位置列が現在の条件でも適合したため、再検証後に現在のメタデータで再出力した。互換L0は現在の車線中点を初期値に再生成した。L2はOldの横位置列を現在の端部に合わせて初期化し、同じ目的関数・補間制約で再最適化した。いずれも現在の車線形状と18°上限で検証し、署名の一致を確認した。

検証は既存関連407件、互換L0関連37件、拡張参照関連15件の計459件が成功。
拡張参照の起動テストは、最初にテスト側の抽出対象メソッドを誤っていたため失敗し、実際の初期化メソッド`_initialize`へ修正して再実行した。
インストール先を模したshare/envのみの配置から両車線を読み込むこと、古いソース署名・不正なCSVを拒否すること、参照を有効化してもMPCの制約行列・上下限を変えないことを確認した。
