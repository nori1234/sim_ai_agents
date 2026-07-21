# SYSTEM — 3リポジトリ＝1つの系（統一アーキテクチャ・真実の単一ソース）

> このファイルは **3つのリポジトリを貫く背骨**。個別リポの README / 設計は必ずここに整合させる。
> プローズは古びうる——数値の一次資料は各リポの `docs/runs/` や issue、契約は `NEURAL_CONTRACT.md` が正。
> 最終更新: 2026-07-21。

## 0. 目的（達成可能な定義での「AGI」）

SF的な万能知能ではなく、**達成可能な現実的ゴール**として：

> **不可逆な帰結のある世界に *接地* した（訓練データの再生ではない）エージェントが、
> 記憶し・継続的に学習し、やがて共有する記号を通じて他者と協力する。**

これは主流（巨大LLM＋検証可能報酬RL＋道具使用）とは別の、**発達 × 接地**の少数派の賭け。
2026のフロンティア（世界モデル/模擬世界・agent memory・weak-to-strong）とは道具レベルで整合し、
極小スケール＋発達的枠組みで行う点が独自（背景は `agent_agi/docs/09` の②節）。

## 1. 器官マップ — リポジトリと役割の対応

**重要**: GitHubリポ名・パッケージ名は**当初目的の名残でバラバラ**。役割（器官）で読むこと。
名前は据え置き、概念層でここに統一する（リポ名リネームは将来の別タスク）。

| 器官 | 役割 | GitHubリポ | パッケージ名 | 名前についての注記 |
|---|---|---|---|---|
| **世界／訓練場** | 決定論的な多エージェント社会＋行動プリミティブ。接地の**測定器** | `sim_ai_agents` | `emergence-world` | READMEは整合済。3名称が別なだけ |
| **脳** | HierMamba 発達エージェント（方策/価値/世界モデル/ワーキング記憶） | `llm_model_agi` | `llm_model_agi` | 名前は「日本語LLM自作」の名残。役割は**脳** |
| **記憶（器官）** | 永続の長期記憶（外部海馬）。＋**体/装備**（道具） | `agent_agi` | `agent_agi` | 名前は「単体自律AGIエージェント」の名残。役割は**記憶器官** |

記憶が「体」でなく独立器官な理由、記憶の種類別の宿り先（ワーキング＝脳／永続＝記憶器官／
手続き＝体）は `agent_agi/docs/09_integration_grounding.md`（器官マップ）で確定済み。

## 2. どう繋がるか（結線）

```
        ┌──────────── 世界 (emergence-world / sim_ai_agents) ────────────┐
        │  帰結のある社会・47動詞・反実仮想ルール・record/replay・society層 │
        └───────▲───────────────────────────────────────────┬───────────┘
                │ 観測(obs)                            行動(Action)│
        ┌───────┴──────── 神経系（結線）─────────────────────▼───────────┐
        │  agent.adapters.emergence（llm側）= obs→トークン / 決定→動詞     │
        │  契約: NEURAL_CONTRACT.md（sim⇄llm、round-trip CI 緑）           │
        └───────▲───────────────────────────────────────────┬───────────┘
                │                                            │
        ┌───────┴──────── 脳 (llm_model_agi) ────────────────▼───────────┐
        │  HierMamba＋方策/価値/世界モデル＋（壊れている）ワーキング記憶    │
        │  親(teacher)=HeuristicBrain（regime盲目・issue#10 R2）           │
        └───────────────────────┬────────────────────────────────────────┘
                                │ 永続記憶の読み書き（v0設計・未実装）
        ┌───────────────────────▼──────── 記憶器官 (agent_agi) ───────────┐
        │  MemoryStore（永続・外部海馬）。v0は決定論バックエンド・既定OFF   │
        └─────────────────────────────────────────────────────────────────┘
```

- **今つながっているのは 世界⇄脳**（契約 v1.x、21+2の相互参照、CIで訓練→接地測定）。
- **記憶器官(agent_agi)は未接続**（相互参照0）。v0設計は `agent_agi/docs/09`、実装は green-light 待ち。

## 3. 当初計画 → 現状（正直なドリフト記録）

| リポ | 当初計画 | 現在の役割 | ドリフトの扱い |
|---|---|---|---|
| llm_model_agi | 日本語LLMを自作（Transformer超え） | 世界で育つ**発達的接地の脳** | 実コーパス事前学習は**未実施のまま**（issue #4）。名前は旧目的の名残 |
| agent_agi | 単体の自律AGIエージェント（記憶ファースト） | 大系の**記憶器官＋体/装備** | 記憶ファースト設計がそのまま接合面に。名前は旧目的の名残 |
| sim_ai_agents | 「どのAI脳がどんな社会を作るか」の創発シム | 接地を反証可能に測る**測定器/訓練場** | #118で再基準化済み。最も整合。名称のみ3つ別 |

## 4. ロードマップ（2本の線＋合流）と現在地

**線A: 個体の接地**（世界⇄脳）
- run #15–#20 まで POWERED-NO。個体RLの梯子（分散低減#18-19・BC annealing#20）は必要だが不十分。
- 次候補: 接地した親（**on-policy蒸留**推奨・素のBCより強い）／探索強化／**privileged critic**
  （engineは訓練時に真regimeを持つ＝密なstep信号）。一次資料 `docs/GROUNDING.md`・issue #130。

**線B: 記憶／体・装備**（脳⇄記憶器官）
- v0記憶 = 設計 LOCKED（`agent_agi/docs/09`）。既定OFF・byte-identical・決定論。実装 green-light 待ち。
- run #20 が「記憶なし方策は遅延帰結の信号をほぼ持てない」と示し、線Aの対策(3)と**合流**。

**合流点 = 記憶**。線Bの第一歩＝線Aの対策候補3。だから最初に固めた対象。

**線C: 協力する社会**（stage 3）
- sim に足場あり（faith/preach・規範#37・評判・噂#96・cooperation）。**接地が前提**なので着手は最後。

## 5. 全リポ共通の不変条件（規律）

1. **決定論baseline byte-identical**（sim）: opt-in全OFFで `tests/test_baseline_contract.py` 不変。
   新機構はフラグ裏・既定OFF。record/replay は bit一致。
2. **契約を壊さない**: `NEURAL_CONTRACT.md`（sim⇄llm）、v0記憶Protocol（脳⇄記憶器官）。
3. **マージ手順**: squash-merge → 直後に `scripts/sync-branch.sh`（sim）／各リポで branch を merged main に同期。
4. **事実と意見を分ける**・**事前登録を尊重**（結果を見てから黙って回さない）。
5. **一次資料はリポに置く**（`docs/runs/`・issue #130/#99）。チャットは要約＋ポインタ。

## 6. 役割語彙（正典・全文書/コメントで統一）

world（世界/訓練場）＝sim ／ brain（脳）＝llm ／ memory（記憶器官）＝agent_agi ／
body・equipment（体/装備）＝sim動詞＋agent_agi.action＋adapter ／ society（社会）＝sim society層 ／
nervous system（神経系/結線）＝`agent.adapters.emergence`＋契約。

## 7. ポインタ

- 世界/接地の叙述と全run: `sim_ai_agents` `docs/GROUNDING.md`・`docs/runs/`・引き継ぎ `docs/HANDOFF.md`・地図 issue #99・経緯 issue #130
- 脳の総まとめ: `llm_model_agi` `docs/PROJECT_SUMMARY.md`・監査 `docs/RESOURCE_AUDIT.md`
- 記憶器官の統合仕様(v0)＋器官マップ＋2026整合: `agent_agi` `docs/09_integration_grounding.md`
- 契約: `sim_ai_agents` `docs/NEURAL_CONTRACT.md`
