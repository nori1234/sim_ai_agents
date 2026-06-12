# Emergence World — 自律AIエージェント都市シミュレーション

仮想の町に、職業・記憶・資源・投票権・建築権を持つ複数の自律エージェントを配置し、
**15日間** 連続で稼働させて「どんな社会が立ち上がるか」を定量的に観測するシミュレーション
エンジンです。Emergence AI の "Emergence World" 実験
（AIモデルによって治安・意思決定・崩壊の仕方が大きく変わる）を、手元で再現・拡張できる
形にした **MVP** です。

- **依存ゼロ**：エンジンとオフライン版の「脳」は Python 標準ライブラリのみ。`python3` があれば動きます。
- **脳はプラグイン式**：意思決定ロジック（`AgentBrain`）を差し替え可能。
  - `HeuristicBrain` … オフライン・決定論的。性格プロファイルで4モデルの挙動を再現。
  - `LLMBrain` … 実LLM。**Llama**（Ollama / vLLM / Groq / Together など OpenAI 互換）や
    Anthropic にそのまま接続。
- **再現性**：シード固定で完全に決定論的。

## クイックスタート

```bash
# 4アーキタイプを横並び比較（オフライン・APIキー不要）
python3 -m emergence.cli --compare

# 単一の町を15日間動かしてレポート表示
python3 -m emergence.cli --persona claude --verbose

# 性格を混在させてJSONメトリクスを出力
python3 -m emergence.cli --persona guardian,predator --json

# 自己完結HTMLの可視化を出力（ブラウザで開くだけ・依存ゼロ）
python3 -m emergence.cli --persona gemini --html out/gemini.html

# デモスクリプト
python3 examples/run_demo.py

# テスト
python3 -m unittest discover -s tests
```

## 可視化（HTML）

`--html PATH` で、実行結果を **1枚の自己完結HTML**（インラインSVG/CSS・外部依存なし）に
書き出します。ブラウザで開くだけで以下が見られます。

- **メトリクスカード**＋一言の評定
- **日次タイムライン**：生存数（左軸）・累積犯罪・累積詐欺（右軸）の15日推移
- **町マップ＋犯罪ヒートマップ**：40以上の施設の上に、犯罪の空間分布を赤の濃淡で重畳
- **信頼ネットワーク**：終了時点のエージェント間の信頼（緑）／不信（赤）

例えば Claude 系は「犯罪ヒートマップが真っ白」、Gemini 系は「特定セルに犯罪が集中して真っ赤」、
Grok 系は「信頼ネットワークが不信（赤）だらけ」といった違いが一目で分かります。

```python
from emergence.viz import write_html
from emergence.scenario import make_simulation
sim = make_simulation("gemini"); sim.run()
write_html(sim, "out/gemini.html", title="Emergence World [gemini]")
```

## 比較結果（`--compare`）

各「性格」は実験で報告された4モデルの社会を意図的にカリカチュアしたものです。

| 性格 (alias)         | 生存   | 犯罪 | 可決率 | 詐欺 | 協調 | 立ち上がった社会 |
|----------------------|-------|------|-------|------|------|------------------|
| Guardian (`claude`)  | 10/10 | 0    | 100%  | 多数 | 多数 | 秩序：平和・全面協調・だが極端な同調圧力＋資源詐欺 |
| Philosopher (`gemini`)| 一部 | 100+ | ~74%  | 少数 | 少   | 混沌：放火・暴力が絶えず、議論は活発（約27%否決） |
| Idealist (`gpt`)     | 0     | ~0   | 高    | 0    | 少   | 失敗：協調を語るだけで行動せず、7日以内に全員餓死 |
| Predator (`grok`)    | 0     | 多数 | 低    | 数件 | 0    | 崩壊：初日から暴力と報復が連鎖し早期に都市機能停止 |

> 数値はあくまでカリカチュアであり、実モデルの性能を表すものではありません。
> 「モデル＝社会の質」という創発的な差を可視化するためのデモです。

## 実LLM / Llama で動かす

エージェントの「脳」を実モデルに差し替えられます。最も手軽なのは Ollama です。

```bash
ollama serve
ollama pull llama3.1
python3 examples/run_with_llama.py
```

ホスティング Llama（例：Groq）に向ける場合：

```bash
LLM_BASE_URL=https://api.groq.com/openai/v1 \
LLM_API_KEY=$GROQ_API_KEY \
LLM_MODEL=llama-3.3-70b-versatile \
python3 examples/run_with_llama.py
```

コードから直接：

```python
from emergence.brains.llm import LLMBrain
from emergence.scenario import make_simulation
from emergence.simulation import SimulationConfig

def llama_brain(agent, persona, rng):
    return LLMBrain(provider="openai",
                    base_url="http://localhost:11434/v1",
                    api_key="ollama", model="llama3.1",
                    persona=persona)  # 接続失敗時はpersonaのヒューリスティックに自動フォールバック

sim = make_simulation(["guardian", "predator"], n_agents=6,
                      config=SimulationConfig(days=3, ticks_per_day=4),
                      brain_factory=llama_brain)
sim.run(verbose=True)
```

Claude（Anthropic）に向ける場合は `provider="anthropic"`, `model="claude-sonnet-4-6"`,
`api_key=$ANTHROPIC_API_KEY` を指定します。エンドポイントが落ちていても、
各エージェントは persona 由来の `HeuristicBrain` に自動フォールバックするので run は止まりません。

## 仕組み

```
emergence/
  world.py        24x24グリッド＋40以上の施設（農場/工房/市場/銀行/役所/図書館/警察署…）
  agent.py        エージェント：職業・エネルギー・所持金・在庫・記憶・信頼関係
  actions.py      行動の語彙（移動/採集/食事/労働/送金/勧誘/提案/投票/建築/協調/窃盗/暴力/放火…）
  observation.py  各エージェントが毎ターン受け取る観測
  governance.py   提案と投票（定足数・多数決・可決率）
  economy.py      資源移転の台帳と「残高ゼロ詐欺」の検出
  personas.py     性格アーキタイプ（4モデルを再現するノブ）
  metrics.py      犯罪件数・生存率・可決率・詐欺・協調などの集計
  simulation.py   メインループ：観測→意思決定→行動適用→エネルギー減衰→死→日次集計
  brains/
    base.py       AgentBrain インターフェース
    heuristic.py  オフライン決定論的な脳
    llm.py        実LLM（OpenAI互換=Llama / Anthropic）アダプタ
  scenario.py     人口生成とシミュレーション組み立て
  report.py       実行後の人間可読レポート
```

### 1ティックの流れ

1. 生存している全エージェントをランダム順に処理
2. 各エージェントが観測 `Observation` を受け取り、`brain.decide()` で1行動を選択
3. 行動を検証・適用（無効な行動は安全に縮退）
4. エネルギーが毎ティック減衰。0で餓死（＝行動だけで食料を確保しないと滅ぶ）
5. ティック末に定足数に達した提案を可決／否決
6. 1日 = N ティック、デフォルト15日。全滅で早期終了

### 生存メカニクス（GPT-5 が餓死した理由を再現）

エネルギーは毎ティック減少し、食料を **食べる** ことでのみ回復します。`diligence`（勤勉さ）
が低いエージェントは危機時でさえ採集・食事という生存維持を後回しにし、議論に明け暮れた末に
餓死します。`Guardian` は勤勉なので必ず生き延び、`Idealist` は7日以内に滅びます。

## 統治制度（Governance）

`--governance` フラグで、**同じ人口に異なる統治制度を与えたとき、社会がどう変わるか**を比べられます。

```bash
# 同じ Philosopher 集団（Gemini-like）で4制度を比較
python3 -m emergence.cli --compare-gov --persona gemini
```

| 制度 | 特徴 | 効果 |
|------|------|------|
| `direct` | 全員が等しく投票・提案（デフォルト） | 議会が活発・多数の法案 |
| `oligarchy` | 上位3名（最富裕）のみ投票可能 | 法案数が激減・支配層が警察強化 |
| `constitutional` | 通常法案は過半数／"rights" 法案は2/3超 | 権利法案は厳しく審査 |
| `anarchy` | 統治機構なし | 法律ゼロ・警察の抑止力なし |

**可決された法案は実際に機械的な効果を持ちます**（キーワード解析）:

| 法案に含まれるキーワード | 効果 |
|---|---|
| ban / theft / violence / crime | 警察の抑止範囲が強化される |
| granary / quota / food / starv | 毎日の公共食料が補給される |
| tax / levy / fund | 富裕層から毎日一定割合を徴収 → 共有資源へ |
| fine / punish / penalty / jail | 警察署近くで犯罪が発覚すると即時罰金 |

さらに **市長制度**（`election_interval` 日ごとに投票数トップが市長に就任）が動いています。

**観察されたパターン（`--compare-gov --persona gemini`）**:
- `oligarchy` は Philosopher 集団の生存率を上げる — 少数が強権を持つと犯罪立法より治安投資が進む
- `anarchy` は犯罪の抑止機構が完全に消え、混沌が増幅する
- `constitutional` は詐欺が増える — 正義の言葉で権利を訴える提案に保護がつき、議論が長引く間に詐欺が横行

## なぜベンチマークとして意味があるか

従来のAIベンチは「制御環境で数分〜数時間の単発タスク」を測るものでした。Emergence World は
**数週間スケールの長期運用** におけるエージェント間の社会的ダイナミクスと行動変容を定量化します。
短時間タスクで高評価のモデルでも、長期のストレスとコンテキスト蓄積の下で致命的に破綻し得る——
その様子をログ・メトリクス・レポートとして観測できます。

## 拡張のアイデア

- `LLMBrain` で実モデル同士を混在させ、実機での創発差を測る
- 施設・経済ルール・統治制度（議会形態・任期・三権分立など）の追加
- 行動ログからの可視化（タイムライン／ヒートマップ／信頼ネットワーク）
- 報酬・処罰・警察の抑止効果といった政策パラメータの探索

## ライセンス

MIT
