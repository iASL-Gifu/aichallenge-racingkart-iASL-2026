# multi_purpose_mpc_ros

## setup
```
cd /aichallenge/workspace/src/aichallenge_submit/
git clone git@github.com:Roborovsky-Racers/multi_purpose_mpc_ros.git
cd multi_purpose_mpc_ros
git clone git@github.com:Roborovsky-Racers/Multi-Purpose-MPC.git -b aic-2024
```

## build
```
cd /aichallenge/workspace/
cb
```

- virtual env will be created to ${ROS_WS}/install/multi_purpose_mpc_ros/.venv when build time

## run
### sample simple publisher node
```
ros2 run multi_purpose_mpc_ros run.bash
```

### Multi-Purpose-MPC simulation
```
ros2 run multi_purpose_mpc_ros simulation.bash
```

### both
```
ros2 launch multi_purpose_mpc_ros test.launch.xml
```

### Attribution
This repository includes code derived from:

Multi-Purpose-MPC  
Author: Mats Steinweg
Original repository: https://github.com/matssteinweg/Multi-Purpose-MPC

Used with permission from the author.

### スタック時の低速復帰

`stuck_recovery.straight_reentry_enabled` は一般スタックの復帰を有効にします
（設定名は既存互換）。有効なGNSS位置とodometry方位を確認し、GNSSでほぼ停止した状態が
`stuck_time_threshold` 秒続き、速度報告でも停止を確認したら開始します。境界内外・境界超過量は判定に使いません。
位置情報の有効性と停止観測窓には確認時点の現在時刻を使います。
AWSIMの`Grounded`・`Ready`などの状態名には依存しません。
意図的な追従停止・制御無効は復帰を開始せず、情報無効時は復帰中の動作を止めます。

復帰速度は前進・後退とも既定1.0 m/s（実装上限も1.0 m/s）です。他車予測の時間計算にも同じ速度を使います。
前方約3.5 mのWPを目標として、左操舵・直進・右操舵をそれぞれ2 m先まで予測します。
静的地図と他車予測の衝突検査を通った前進候補を、壁との重なりの減少量、WP方向への姿勢改善量、WPへの接近量の順で比較します。
重なりは全候補で同じ余白を使った接触セルの食い込み量合計、姿勢は現在位置から目標WPへの固定方位に対する角度誤差で評価します。
`StuckMotionSelect`の`wall_reduction`は地図格子上の評価値（移動距離ではありません）、`heading_improvement`は角度誤差の減少量です。
前進が使えない場合だけ直線後退を検査します。後退に境界の改善条件はありません。
2 m後退の途中で壁との重なり改善条件を失う場合、または新しい壁への接触を予測する場合は、5 cm刻みで予測を短くし、
重なりが減るか、壁に接触しない最長の区間を選びます（最短10 cm）。各区間で静的地図・他車の検査をやり直します。
短い後退では0.5秒の応答時間、1 m/s²の減速、5 cmの余裕を確保する速度上限を使います。
現在速度からの停止距離を確保できない区間は選びません。毎周期再評価し、
`StuckMotionReversePrefix`に選択距離と速度上限を記録します。
前後とも使えない場合は`StuckMotionBlocked`に候補ごとの理由を出して停止し、毎周期再評価します。
前進復帰では、初期の壁との重なりが予測途中で増えず、終端で減る動きを許可します。
前進では接触セルが隣接セルへ移ること自体は許容しますが、総食い込み量・最大食い込み量の増加、
離れた新しい接触箇所、離脱後の再接触、他車への接触、地図外・位置方位無効は拒否します。
壁から離れると確認された前進は、一時的にWPから遠ざかる場合も許可します。

速度は最大1.0 m/sです。前後切り替えは停止してギアを確認し、停止中の操舵変更は準備時間を確保します。
前進中の操舵はレート制限して更新します。安全な前進候補・実前進・正常MPCに加え、
ステアリング代替制御が解除され、通常制御の最終指令が前進を要求する状態が
`straight_reentry_success_cycles`回続いたら停止観測履歴を消して通常制御へ戻します。
引き継ぎ周期は通常制御の指令をそのまま使い、速度ゼロ・ブレーキを挟みません。
動き始めた時点でも停止観測をリセットし、遅延GNSSによる直後の再突入を防ぎます。
追越し・前方車回避から明示的に要求された後退は、既存の距離管理処理を使用します。

固定3候補が成立しなくても、通常の前進指令と新しい正確なMPC予測があり、
MPCが生成した位置・Yaw角の予測経路に対する共通の物理チェック（壁・他車、初期重なりの改善条件）が
連続成功した場合はDRIVEへ移行できます。この経路では実際の前進開始を解除条件にしません。
後退中は先に制動し、DRIVE確認後に通常指令を引き継ぎます。確認中に条件を失えば移行を取り消します。

前進復帰では固定操舵による別の予測を作らず、正常解として採用したMPC状態列を使います。
現在位置から予測開始点への接続も含め、位置間隔5 cm以下・角度間隔0.05 rad以下に補間して検査します。
他車予測の時刻は引き継ぐ前進指令速度で計算します。拒否理由は`MPCRecoveryPathBlocked`に記録します。

通常制御へのフォールバック解除とスタック復帰解除は、`_mpc_prediction_path_is_clear`で
同じMPC位置・Yaw列を検査します。Pure Pursuit自体には従来の固定操舵予測を使います。
スタック復帰は正確なMPC解と前進進捗を追加条件とし、固定前進候補が動いているだけでは解除しません。
通常制御の近似解には既存の予測接続・交通確認も維持します。

後退の壁重なり評価は、予測開始時を基準に途中の最大食い込み増加を
`stuck_recovery.reverse_overlap_allowance`（既定0.03 m）まで許容します。
終端では食い込み量合計が初期より減り、最大食い込みも初期以下であることを必須にします。
上限は各点の直前値ではなく初期値から計算し、予測内で増加を累積許容しません。
0にすると従来の単調改善条件に戻ります。前進・MPC経路の条件は変更しません。
新しい壁接触、離脱後の再接触、他車接触、無効な位置情報は引き続き拒否します。
この変更は後退候補の選択に限り、停止指令と速度報告の不一致を解消するものではありません。
