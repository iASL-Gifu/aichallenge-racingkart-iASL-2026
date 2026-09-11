# 車体表示・連続安全確認・境界計算の軽量化

設定（config/config.yaml、mpcセクション）:

```yaml
collision_body_visualization_enabled: false
collision_body_visualization_rate_hz: 5.0
```

既定Off。On時は5〜10Hzに制限し、シミュレーション時刻の巻戻り時は表示を再開。
Offでは車体マーカーと付随CollisionBodyPairログを生成しない。
現在・予測・移行経路の衝突確認は従来どおり実行する。設定は起動時のconfig読み込みで反映。
既にRVizに残ったマーカーはDisplayのReset等で消す必要がある場合がある。

幾何共有:
- SAT投影寸法をyaw・不確実性・車体寸法・投影方向の完全一致キーで最大4096件保持。
- 連続区間の補間自車形状を区間両端のBodyPose・分割数・角度・paddingで最大128区間保持。
- 相手車の位置・速度・観測経過による予測と重なり判定は毎回行う。
- 分割数、512上限、クリアランス、方向不明・無効観測時の扱いは維持。

境界共有:
- 境界点までの符号付き距離（最大4096件）と法線方向（最大1024件）を完全一致キーで共有。
- 人工レーン境界はupdate_path_constraintsの1回の呼び出し内で共有。
- 障害物探索・物理境界との交差・retry緩和量適用・境界セル更新を省略しない。
- 結果の上限/下限や物理マージンは変更していない。

確認:
- 変更前のreference_path.pyと40条件（WP30/230/250/260/275、全幅/L0/L1/L2、Hybridあり/なし）を比較。
  final/hard/lane境界配列が完全一致。
- 変更前のswept_path_clearと150条件の判定が一致。
- 境界単体中央値: 1.4505ms → 1.2373ms（ローカル、制御周期全体の改善率ではない）。
- 全体608 passed / 1 failed。既存失敗はouter_course_margin実設定0.7mに対するテスト期待値0.6m。
- 事前生成L0参照の境界ソースハッシュを更新。CSV経路や設計パラメータは変更なし。
- AWSIM走行は未確認。次回ControlTimingの表示・連続安全確認・prepare.boundariesで実負荷を確認。
