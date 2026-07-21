# セッション引き継ぎ (HANDOFF)

> **系の全体像（3リポジトリ＝1つの系・器官マップ・ロードマップ）は [`docs/SYSTEM.md`](SYSTEM.md) が真実の単一ソース。**
> この HANDOFF は sim(世界)側の運用引き継ぎ、SYSTEM.md は3リポ横断の背骨。

次のエージェント向けの現状引き継ぎ。**最終更新: 2026-07-21**(run #20/#25(A1)完了・
訓練不安定の根本修正済み・run #26(v1b)訓練中)。
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

接地はまだ未確認。だが run #14→#19 で**ブロッカーが1つずつ潰され、原因が測定として特定**
された。順に:

- **タスク較正(#15)**: `control-margin-1` が「control側の預金牽引は弱い」説を反証
  (+4.35σ・巨大)。小さいのは cf側の**随伴性マージン**(≡`advantage_cf`)だけ。
  `demurrage_per_day` を計器全体に配管し、事前登録ルール([+0.5,+1.0]σの最小rate)で
  **0.25/day** に較正(`contingency-calib-1`、+0.53σ)。run #15 は POWERED-NO。
- **計測バグ(#16)**: 初のS4計器付き run で `probe_teacher_n=0` が一発で露見——
  ドライバは `agents[0]`=**sole banker**(構造上預金できない)の脳を採点していた。
  **#14/#15 は公正タスクテストとして無効**。ドライバ修正=計測脳は sandbox saver
  `agents[1]`(計器の慣行と一致)。
- **初の有効判定(#17)**: POWERED-NO・`n_conclusive` 初の 20/20。S4を機序として定量化——
  バッチ内リターンは ±10〜140 で暴れ、随伴性の報酬差は ±0.60/ep。depositの正規化後
  advantage は ≈0(−0.0116)、生クレジットは逆順にすら見える=**採点マージン上の
  信号対雑音比**が唯一のブロッカー(表現0.81–0.98・動機+0.53σ・探索密は全て健全)。
- **分散低減(#18-19)**: brain の `adv_baseline="episode"`(エピソード区間の平均を
  正規化前に引く)で信号が復活。200ep(#19)で**depositの使用advantageが初めてregime分離**
  (cf −0.12 / control ≈0、中盤 −0.23/+0.05)。だが行動は平坦のまま。
- **今の綱引き(#19が示した次の壁)**: `probe_teacher_n 435 ≈ probe_self_n 443`——
  学習の半分が**regime盲目の親(HeuristicBrain)へのBC**で、両regime密に預金を実演する
  ので、疎で正しいPG勾配を打ち負かす。これは issue #10 R2(盲目teacherのアンカー問題)
  そのもの。**親の実体は `money>=12→deposit` の手書きif-elseで、`deposit_rate`/
  `demurrage` を一切見ない**(接地は原理的にBCからは来ない)。
- **BC annealing(#20)**: 親離れをスケジュール化(0.30→0.05)。離乳しても行動は
  regime分離せず、預金密度はむしろ低下。**POWERED-NO**。盲目teacherは*密度*を
  ブートストラップできるが*随伴性*は無理、自己プレイPGは随伴性を勾配で表現できるが
  まだ*勝てない*——どちらの単一チャネルもこのスケールで接地を閉じない(run-20 README)。
- **A1=接地teacher(#25)**: #20の本命診断(issue #10(c))。regime**盲目**の親を
  regime**認識**の grounded heuristic に差し替え(controlで預金・demurrageでREST)、
  正解regimeを毎ep教示、BCは0.3固定(チャネルを開けたまま)。結果 **POWERED-NO**——
  預金回数 control 167 ≒ cf 160、`fraction_grounded 0.00`、`wilcoxon_p 1.0`。
  **接地した教師を与えても BC で随伴性は伝わらない**。ボトルネックは「接地teacherの
  不在」ではなく、run-20 が事前登録した fail 分岐=**方策/表現の学習可能性**。
  一次資料 `docs/runs/run-25/`。
  - ⚠️ **交絡の注意**: この run から訓練安定化で状態表現に `LayerNorm` を導入
    (下記)。scale情報を捨てるので、regime信号が scale に乗っていたら消えた可能性。
    regime は token パターン(預金/残高の推移)に出るはずで LayerNorm でも保たれ、
    かつ #17–20(正規化なし)と同じ POWERED-NO なので整合的だが、表現学習可能性を
    「無理」と断ずる前に with/without-norm 対照 or 正規化状態の decodability probe を
    owe(表現学習可能性ラインの第1タスク)。
- **訓練不安定の根本修正(#21/#22 のクラッシュ原因)**: brain の `encode_state` に
  出力正規化が無く、訓練で表現の大きさが発散→価値MSE/生の好奇心(wm_loss)/policy
  logits がNaN→`multinomial` が例外→`decide()` の素の except が握り潰して「原因不明の
  heuristic フォールバック」化。メタ安定なので運init次第で ep1〜ep163 のどこでも落ちる
  (ドライバがRNG未seed)。修正: brain=パラメータ無し `F.layer_norm`(状態)+非有限loss
  時は更新スキップ / engine=RNG seed 固定+握り潰し例外をstderrへ+fatalに真因を出す。
  記憶の byte-identity と baseline 決定性テストは不変。#25 が200ep完走で **CI実証済み**。
- 一次資料: `docs/runs/run-1[5-9]/`・`run-20/`・`run-25/`(各 README+battery.json)、
  `control-margin-1`・`contingency-calib-1`、叙述は `docs/GROUNDING.md`、経緯は #130。

**次の分岐**:
(a) **表現の学習可能性ライン(POWERED-NO の事前登録フォローアップ)**: まず上記の
LayerNorm 交絡チェック(正規化状態の regime decodability probe)、次に観測エンコード/
memoryless方策が「demurrage→預けない」を*表現・学習*できるかを直接測る。メトリクスの
再チューニングではない。
(b) **記憶ライン(v1b=run #26 訓練中)**: `memory_into_policy=true` で「この状況の過去の
帰結」を想起して h に足す。memoryless方策の信号の薄さ(ticksをまたぐ随伴性)への構造的
回答になりうるか——#17–20/#25 に対して接地が動くかを測る。結果待ち。
(c) 本物のLLM親——ただし CI 訓練ループは CPU・LLM未接続なので要環境整備。

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
