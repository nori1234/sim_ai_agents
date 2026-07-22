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

## 走行中の実験（G2採点、各 ~3.5–4h）

- **#41** v2a baseline（記憶+`state_lsh` 12bit）＝ G2 の基準値（記憶が推論に使われているか初可視化）。
- **#42** v2a + **B2 信念ヘッド**＝ 記憶特徴（`mean_reward_deposit`＝今epの被弾帰結）→ P(cf) 予測 →
  **方策に注入**、`_priv`(真regime)を教師に BCE補助損失。特権criticの鏡像（あちらは value専用で漏れ
  なし、B2 は方策に*予測した信念のみ*＝deploy時は推論、漏れなし）。llm_model_agi の `belief_head` フラグ。

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
