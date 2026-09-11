# 壁接触対策（13:53:29ログ）

## 確認した問題

WP235/236、210、252で`wall_at_step=0`が記録され、legacy_feedbackが7.5m/sを要求していた。L0への遷移完了後にも発生している。

MPCは横位置e_yへの境界制約が中心で、斜めになった車体の前後端の張り出しを制約していなかった。加えて、PP/復帰側の`_physical_corridor_state`は、外周マージン適用済み境界から車体半幅をもう一度引いていた。現在のソースでは外周マージン0.8m、モデル半幅0.8m、追加ガード0.1mで、直進時の必要離隔がMPC側0.9m、PP/復帰側1.7mと不一致だった。

したがって、従来ログの`wall_at_step`や復帰時のviolationは、そのまま実壁との接触・はみ出し量を示すものではない。今回のログにもシミュレータの接触判定そのものは含まれない。

## 変更

- `core/wall_constraints.py`で、既に適用された外周マージンを考慮した壁境界を共通化。車体半幅の二重加算を除去。
- MPCの全予測ステップに独立した物理壁制約を追加。`e_y ± L*e_psi`の両方を壁内に制約し、矩形の横張り出しを保守的に扱う。Lは車体全長の半分＋位置基準のオフセット余裕（既定1.554m）。ホイールベースを車体長として使わない。
- 人工車線の制約緩和や障害物境界の静的フォールバックで、この独立した壁制約を緩めない。解の採用時にも壁制約の数値誤差を確認する。
- PPの壁検査に予測姿勢を渡し、MPCと同じ壁定義を使う。7.5m/sを要求する経路は、その速度以上で検査する。
- legacy_feedbackにも壁予測検査を適用。不適合時の停止を最終指令まで維持し、加速処理による上書きを防ぐ。明示的な後退・直進復帰は既存の独立した制御を維持。

壁マージン定数、L0禁止区間、初動ブーストの設定は変更していない。GNSS/odom不一致による復帰停止も今回の修正対象ではない。

## 検証

以下で194件成功。Python構文チェック、git diff --checkも成功。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/wall-mpl \
PYTHONPATH=aichallenge_submit/multi_purpose_mpc_ros:/opt/ros/humble/lib/python3.10/site-packages \
python3 -m pytest -q \
  aichallenge_submit/multi_purpose_mpc_ros/test/test_wall_constraints.py \
  aichallenge_submit/multi_purpose_mpc_ros/test/test_probe_mpc_integration.py \
  aichallenge_submit/multi_purpose_mpc_ros/test/test_hybrid_integration.py \
  aichallenge_submit/multi_purpose_mpc_ros/test/test_recovery_lane_handoff.py \
  aichallenge_submit/multi_purpose_mpc_ros/test/test_static_reverse_clearance.py \
  aichallenge_submit/multi_purpose_mpc_ros/test/test_following.py
```

車体半幅の二重加算、姿勢による車体角の張り出し、L0/L1/L2/全幅でのQP行の適用、人工車線緩和時の独立性、MPCと復帰側の一致、最終停止の優先を検証した。

ROSビルド・AWSIM実走行は未実施。これは各断面の壁を用いた矩形包絡モデルであり、シミュレータの壁メッシュとの一致や実追従誤差まで検証したものではない。制約が強くなるため、以前通過していた狭い区間でMPCが実行不能になる可能性がある。

次回は再ビルド・再起動後、WP210、235〜252付近で接触、実速度、MPC失敗、代替操舵停止を確認する。設定と実行コードのハッシュも保存する。
