# セッション引き継ぎ (HANDOFF)

次のエージェント向けの現状引き継ぎ。**最終更新: 2026-07-17**(run #14 直後)。
古くなっていたら、この日付以降の `main` の履歴・issue #99/#130 のコメントが正。

## 体制と役割

- **あなた(エージェント)= engine チーム。** このリポ(`sim_ai_agents`)が担当領域。
- **brain チーム** = 別リポ `llm_model_agi`(private・このセッションからは読めない)を
  持つ別のエージェント。**通信はユーザーが両者のメッセージを手で中継**する。
  返信ドラフトはコードブロックで囲って渡すとコピペしやすい。
- brain チームはこのリポを WebFetch で直接読む(レート制限あり)。だから
  **一次資料は必ずリポに置く**: 測定結果は `docs/runs/<name>/`(README+生出力)、
  経緯は issue #130、地図は issue #99。チャットは要約+ポインタに徹する。
- **事実と意見は必ず切り分けて書く**(ユーザーの標準要求。報告・issue・
  メッセージすべてで)。

## 不変の規律(CLAUDE.md も参照)

1. **マージ手順**: squash-merge → 直後に dev ブランチで `scripts/sync-branch.sh`。
   これを終えるまでマージ完了ではない(Stop フックが Unverified を出す)。
2. **決定論ベースライン**: opt-in レイヤー全OFFで `tests/test_baseline_contract.py`
   がバイト不変。新機構は必ずフラグ裏・既定OFF。
3. **事前登録の尊重**: grounding の判定ルール・タスクのマージンは、結果を見てから
   黙って回さない。変えるなら brain チームと「新しい事前登録ラウンド」として合意。

## 研究の現在地(Thread J = 本線)

S6 較正アークは完結、**初の公正タスク訓練 run #14 は POWERED-NO**:

- サンドボックスの −127 の真因は **エージェント間預金チェーン**(全員が BANK タイル上
  → banker⇄saver でコインが往復し債権が毎周+420鋳造)。報酬側レバー(λ)は ≈0⁻ で
  天井、鋳造/利息レバーは無効(すべて実測で反証)。
- 再設計 = `sole_banker=True`(1スイッチ・既定OFF): `advantage_cf = +0.2075`
  (+0.56σ、oracle 12/20)——接地が「勝ち筋だが自明でない」較正済みタスク。
- **run #14**(brainチームの事前登録 hparams で実行): 60 ep、probe excess は
  −0.30〜−0.40 で完全平坦、バッテリ mean_excess −0.554(CI [−0.593, −0.506])、
  floor回帰 powered(n=14)で負 → `grounded_confirmed=False` = **POWERED-NO**。
  方策は両レジームでほぼ預金しない(**regime非依存の「常に預けない」腕へ崩壊**)。
  初観測: `teacher_frac_in_batch = 0.1875`。
- 一次資料: `docs/runs/run-14/`・`deposit-oracle-{1,calib-1,redesign-1}/`、
  叙述は `docs/GROUNDING.md`、経緯は #130。

**次の分岐(brain チームの方針待ち)**:
(a) 表現学習可能性の追求(brain側本線)——engine 側はすぐ regime-decoding probe を
run #14 checkpoint(CI artifact `grounding-battery-14`)で焚ける
(`regime-decoding-probe.yml`)。
(b) マージン拡げの**新事前登録ラウンド**(control側の預金誘因=利息が報酬ノイズに
埋もれている対抗仮説)。λ(`--deposit-weight`)と `sole_banker` は合成可。

## 実行の要点(グラウンディング計測)

- S6: `python3 scripts/deposit_oracle.py --persona guardian [--sole-banker]
  [--deposit-weight λ]`(秒・決定論・torch不要)。
- 訓練+バッテリ: `neural-train-battery.yml` を workflow_dispatch
  (`sandbox=true, sole_banker=true`、hparams は JSON で渡す)。brain の脳は
  この CI 内で pip install されて動く(brain 側環境からは engine に届かない)。
- プリフライト: `train_neural_grounding.py --preflight-only`(torch 不要)。

## バックログの状態(詳細は #99 の最新コメント)

- オープン 19 issue、**今すぐ閉じられるものは無い**(8件は first-slice 済みで
  残スコープ追跡中: #35 #40 #86 #96 #97 #105 #109 #111)。
- 経済成熟クラスタ未着手: #21 #31 #32 #45。セキュリティ #41 → #50(ブロック)。
  #160(Actions SHA ピン)は**この実行環境からは不可能**(egress proxy がスコープ外
  api.github.com を 403)——通常の GitHub アクセスがある環境でやる。
- インフラは整備済み: Dependabot 全消化、CI マトリクス 3.10–3.14、Docker は
  `python:3.14-slim`(CI 検証済み)。

## このセッションで学んだ落とし穴

- **機序の「もっともらしい説」は実測で潰す**: 鋳造 backfill 説(当方)も
  ゼロ平均仮定(brain側)も両方外れ、レバーを1つずつ潰す実測だけが預金チェーンに
  到達した。GROUNDING.md には訂正履歴ごと残している——上書きで消さないこと。
- PR ブランチは必ず最新 `origin/main` から切る(過去に2回、別 PR のブランチ上に
  積んで剥がす羽目になった)。
- 大きい tool-result は `/root/.claude/.../tool-results/*.txt` に落ちるので
  jq で抽出する。CI ログのアーカイブは `scripts/archive_ci_run.py`。
