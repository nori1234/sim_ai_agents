# セッション引き継ぎ (HANDOFF)

> 系の全体像（3リポ＝1つの系・器官マップ）は [`docs/SYSTEM.md`](SYSTEM.md) が単一ソース。
> ここは運用引き継ぎ。叙述と深掘りは [`GROUNDING.md`](GROUNDING.md) 冒頭。**最終更新: 2026-07-22**。

## 最上位の枠組み

**AGI ← 接地 ← (3) エピソード内のレジーム推論 ＋ (4) その行動化。**
接地の操作的定義＝「隠された世界の法則を、自分が生きた*不可逆な帰結*から推論し、行動を変える
（訓練の再生でない）」。知覚は解決済（符号化に随伴性 94% 保持＝decodability probe）。**壁は
(3)+(4)**。以降すべてはこの一点に奉仕する。

## 合否メトリクス ＝ G2（最重要・これで全アームを採点）

- **G1（excess / norm_contingency）は反射で通過可能**と実証。盲目 floor は `money≥12→預金` の
  記憶なし規則で、しきい値を上げるだけ (T=20) で G1 **+0.716**＝floor(+0.52)超え。⇒「excess>0」は
  接地を証明しない。floor の随伴性は**軌道乖離 100%・レジーム検出 0%**（同一富ビン内の預金率は
  両regime同一、cf は demurrage で貧しくなり決定点が低ビンに寄るだけ）。
- **G2 ＝ 富マッチ随伴性**（同一富ビンでの預金率差 control−cf）。記憶なし方策は **G2≤~0**（純富規則で
  ≡0、実 floor は decide 優先度×regime の小残差で ≈−0.05〜−0.09、**決して正にならない**）。判別子：
  しきい値 12→20 で G1↑・**G2 据置**。∴ **正の G2 は反射で到達不能＝真の接地**（＝北極星そのもの）。
- 実装済：`money_matched_contingency` / `measure_money_matched_contingency`
  (`emergence/grounding.py`)、battery 配線 (`scripts/train_neural_grounding.py` の `[G2]` 行・
  `battery.json.money_matched_contingency`)、`tests/test_money_matched_contingency.py`。
- 一次資料 **`docs/runs/metric-trajectory-confound-1/`**。

## 現在地

記憶+credit+最適化 族は **G1≈+0.09 で頭打ち**（v2a #29 が天井、`docs/runs/run-31-40-sweep/` 9セル：
LSH増bitは0へsmear・軟化criticは密度↔ノイズ・GAE平坦）。**族は正しい問題を解いていなかった**
（floor は履歴推論せず反射で +0.52）。∴ 標的を G2、機構を (3)+(4) へ移した。

## 実験結果 grid #41–#48（8並列・完了）＝ 壁は (3)推論 と判明（`docs/runs/grid-41-48-mechanism-task/`）

機構×課題×belief強度×days の8アーム、全て **G2≈0・UNDETERMINED＝接地せず**。決定打は **D1**：
B2信念ヘッドを `_priv` で直接BCE教師しても、eval の **belief_decode_accuracy=0.501（偶然）／
平均信念 control 0.500 / cf 0.501＝両regime無分離**。∴ **記憶recall特徴（`mean_reward_deposit`等）
は今エピソードのレジーム証拠を学習可能な形で運んでいない＝(3)推論が失敗**。
- #41 v2a base: G2 +0.007、預金率 0.665/0.660（両regime同率＝再生）。
- #44 v2a+B2 C1: 密度崩壊（発火1.5–19%）、G1 −0.079（やや逆）。C1は設計通り反射不能→推論
  できず"預金を止めるだけ"。
- **根本原因（8本分析で確定）＝state-keyed recall は構造的にレジーム盲目**：想起は state類似で引く
  ＝レジーム非依存なので control/cf が同一バケット→混合平均→信号ゼロ。レジームは「自分が・この一生
  で」という**時間的/自伝的構造**にしか無く、recall はそれを捨てる。∴ recall基盤の B2/critic 亜種は
  再走させない。
- **手段ランク（目標＝D1>0.5 の信念分離）**：
  - **M2（実装済・#49-51走行中）**：felt な帰結を観測に露出（`obs.economy.deposit_yield`＝demurrage−/
    interest+）。方策が推論せず*反応*できるか＝**定義A判別**（#49 memoryless+felt が肝）。engine フラグ、
    tokenizer 自動露出、determinism 不変。
  - **M6（実装済・走行中）**：`obs_window` で系列backbone に直近K観測を食わせ**軌道を積分**＝最も
    native。窓per-episodeリセット・on-policy一貫（obs_tok保存）・既定1でbyte-identical・テスト有。
  - **M1（未実装）**：リカレント信念、**M4（未実装）**：自伝的記憶キー。
- **定義判断が保留**：接地＝A「felt帰結への反応で可」／B「隠れ法則の推論を要求」。#49(M2 memoryless)と
  M6 の結果で、どちらの定義でも動くかをデータで切り分ける。判定は常に **G2＋D1**。

## 全方位アーム（4介入点・全て G2 採点）

- **C1 反射不能タスク**（engine）：demurrage の帰結を観測可能な spendable money から切断し、regime を
  履歴からしか推論できなくする（＝floor の G1 も 0 に落ちる課題）。**次に実装**、torch-free で floor 検証可。
- **A1 リカレント信念**（brain・メタRL/RL²）：エピソード内 (obs,act,rew) を要約する走る状態を方策へ。
- **B1 情報探索の内発報酬**：regime 不確実性↓ の行動に報酬＝能動的実験者。
- **D1 内部信念プローブ**（診断）：隠れ状態から regime をデコード＝**(3)推論と(4)行動化を分離**。
- 降格：記憶の風味違い（bit数 / critic mix）は圧力なしにアーキだけ弄り反射のまま＝低EV。

## 体制（スコープ）— 旧「brain は読めない別チーム」は無効

このセッションは **3リポ全部を担当ブランチ `claude/investigation-t0zm4z` で保有・編集**する：
`sim_ai_agents`(engine/世界)・`llm_model_agi`(brain)・`agent_agi`(memory)。3つとも `/home/user/` に
clone 済・import 可。手動中継は不要。ユーザー（研究ディレクター）に合意を取るのは**両リポの共有
標的を変える時だけ**（例：G1→G2 のメトリクス変更）。

## 不変の規律（CLAUDE.md も参照）

1. **決定論ベースライン**：opt-in 全OFF で `tests/test_baseline_contract.py` がバイト不変。新機構は
   必ずフラグ裏・既定OFF（zero-init encoder・損失+0.0 で担保）。
2. **マージ**：squash-merge → 直後に dev で `scripts/sync-branch.sh`（終えるまで未完）。
3. **事前登録の尊重**：判定ルール・タスクのマージンは、結果を見てから黙って回さない。
4. **事実と意見は必ず切り分けて書く**（ユーザー標準要求）。

## 実行

- 訓練+バッテリ：`neural-train-battery.yml` を workflow_dispatch。`sandbox=true, sole_banker=true,
  demurrage_per_day=0.25`。3リポの ref はいずれも `@claude/investigation-t0zm4z`。
- v2a hparams：`{"memory":"episodic","memory_into_policy":true,"memory_key_mode":"state_lsh",
  "memory_lsh_bits":12}`（+B2 は `"belief_head":true`）。`build_brain` が hparams→AgentConfig を
  汎用転送するので新フラグはコード変更なしで流れる。
- torch-free 診断（各 数秒〜分）：`scripts/threshold_landscape.py`・`money_matched_contingency.py`・
  `deposit_oracle.py`・`train_neural_grounding.py --preflight-only`。
- 既知インフラ修正：`encode_state` に `F.layer_norm`（表現発散→NaN→無言 heuristic フォールバックの
  根治、RNG seed固定・例外可視化込み）／memory recall O(全件)→O(tag)。**無言フォールバックを疑ったらここ**。
- 大きい tool-result は `/root/.claude/.../tool-results/*.txt` に落ちる → jq/python で抽出。

## 一次資料

**`docs/runs/metric-trajectory-confound-1/`（最新 frontier）**・`run-31-40-sweep/`・
`run-28-30-memory-critic/`・`run-25/`。叙述＝`GROUNDING.md` 冒頭（訂正履歴ごと保存、上書き消去しない）。
経緯＝issue #130、地図＝#99。記憶設計＝`agent_agi/docs/10`、critic＝`llm_model_agi/docs/PRIVILEGED_CRITIC.md`。
過去の run 詳細（#14–25 の逐次ブロッカー潰し）は各 `docs/runs/` と GROUNDING.md に格納済。
事業的・思想的な *why*（フィジカルAI 4層の堀＝接地機構／離散→連続の質的断絶）＝
`docs/notes/why-grounding-is-the-moat.md`。
