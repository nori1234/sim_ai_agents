# アーキテクチャと内部

エンジンの構造・1ティックの流れ・実LLM／発達脳の差し込み方・HTML可視化。
（接地検証の設計と道具は [`GROUNDING.md`](GROUNDING.md)、発達脳との統合契約は
[`NEURAL_CONTRACT.md`](NEURAL_CONTRACT.md) が正典。）

## ファイル構成

```
emergence/
  world.py        24x24グリッド＋40以上の施設（農場/工房/市場/銀行/役所/図書館/警察署…）
  agent.py        エージェント：職業・エネルギー・所持金・在庫・記憶・信頼関係
  actions.py      行動の語彙（移動/採集/食事/労働/送金/勧誘/提案/投票/建築/協調/窃盗/暴力/放火…）
  observation.py  各エージェントが毎ターン受け取る観測
  governance.py   統治制度・可決法案の機械的効果・市長制・政策エンジン
  economy.py      資源移転の台帳と「残高ゼロ詐欺」の検出
  drives.py       三大欲求（食欲・睡眠欲・性欲/繁殖）— 衝動・快感の本能モデル
  esteem.py       社会的欲求（承認欲求・名誉・権力）と称賛の経済
  psyche.py       安全欲求（恐怖）と自己実現（創造への引力）
  society.py      裏社会と文化（武器・薬物・ギャング・宗教）と施設の役割
  environment.py  外の世界（天気・季節・マクロ経済・災害・資源枯渇）
  affordances.py  施設の役割・職業の役割をデータで定義（possibility space）
  publicworks.py  公共事業ループ（国庫→議会承認→大工が建設）
  development.py  歴史的発展（開拓村→前提条件つきテックツリー→繁栄度）
  market.py       経済の物理（交換 OFFER/ACCEPT・生産 CRAFT・信用 LEND/REPAY・価格は創発）
  memory_backend.py  長期記憶アダプタ（任意ライブラリ memory-agent を利用）
  personas.py     性格アーキタイプ（4モデルを再現するノブ）
  metrics.py      犯罪件数・生存率・可決率・詐欺・協調などの集計
  simulation.py   メインループ：観測→意思決定→行動適用→エネルギー減衰→死→日次集計
  brains/
    base.py            AgentBrain インターフェース
    heuristic.py       オフライン決定論的な脳
    llm.py             実LLM（OpenAI互換=Llama / Anthropic）アダプタ
    neural.py          発達脳アダプタ（経験から継続学習；依存任意・失敗時はheuristicへ縮退）
    _neural_reward.py  観測差分→報酬（純Python・torch不要）
    neural_contract.py 世界⇄発達脳の契約の真実源（行動語彙・param規約・観測スキーマ・target解決）
  grounding.py    接地検証器：反事実世界の転移テスト（excess＝ヒューリスティック床超過分）
  grounding_monitor.py  学習中に接地スコアの推移を記録（GroundingMonitor）
  scenario.py     人口生成とシミュレーション組み立て
  report.py       実行後の人間可読レポート
  cli.py          コマンドライン入口（--compare / --html / --llm …）
  viz.py          自己完結HTML（インラインSVG/SMIL）の書き出し
  # --- 観察所（製品レイヤー）---
  api.py          EmergenceAPI：トランスポート非依存の製品ロジック（JSONを返す）
  server.py       stdlib http.server の薄いHTTPアダプタ（ASGIに差し替え可能）
  web/observatory.html  単一HTMLのUI（リッチ2Dの街＋年代記＋憑依ビュー）
  chronicle.py    町の年代記・住人の人生ストーリー（物語化）
  replay.py       LLM実行の記録／再生（再現性の土台）
```

## 1ティックの流れ

1. 生存している全エージェントをランダム順に処理
2. 各エージェントが観測 `Observation` を受け取り、`brain.decide()` で1行動を選択
3. 行動を検証・適用（無効な行動は安全に縮退）
4. エネルギーが毎ティック減衰。0で餓死（＝行動だけで食料を確保しないと滅ぶ）
5. ティック末に定足数に達した提案を可決／否決
6. 1日 = N ティック、デフォルト15日。全滅で早期終了

## 生存メカニクス（GPT-5 が餓死した理由を再現）

エネルギーは毎ティック減少し、食料を **食べる** ことでのみ回復します。`diligence`（勤勉さ）
が低いエージェントは危機時でさえ採集・食事という生存維持を後回しにし、議論に明け暮れた末に
餓死します。`Guardian` は勤勉なので必ず生き延び、`Idealist` は徐々に餓死して15日以内に全滅します。

## 実LLM / Llama で動かす

エージェントの「脳」を実モデルに差し替えられます。LLM脳は毎ターン、**思い出した記憶**と
**世界の状態（季節・物価・災害）**を見て行動を選ぶので、ヒューリスティックには無理な
「**過去に学び、環境に適応する**」動きができます（例：「去年の冬は飢えた→今年は備蓄しよう」）。

最も手軽なのは Ollama：

```bash
ollama serve && ollama pull llama3.1
# CLIから（記憶＋環境を一緒に有効化すると接地が活きる）
python3 -m emergence.cli --persona claude --llm --memory --environment --days 5
# または例スクリプト（記憶＋環境ON済み）
python3 examples/run_with_llama.py
```

`--llm-base` / `--llm-model` / `--llm-provider`（`openai`=Ollama互換 / `anthropic`）で
任意のエンドポイントに向けられます。エンドポイントが落ちていても各エージェントは
persona由来の `HeuristicBrain` に**自動フォールバック**するので run は止まりません。

> **手元のPCだけで動かす最小構成** → [`LOCAL.md`](LOCAL.md)。

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
`api_key=$ANTHROPIC_API_KEY` を指定します。

## 発達脳で動かす（NeuralDevelopmentalBrain・実験的）

LLM脳が「凍結した賢い人形」なのに対し、`NeuralDevelopmentalBrain` は **世界の中で経験から
継続学習し、親（teacher）に育てられる脳**です。重い本体（HierMamba/policy/world-model/Titans
/replay/発達段階）は別パッケージ `llm_model_agi` にあり、本リポジトリは薄い `AgentBrain`
アダプタだけを持ちます。

- **opt-in・既定OFF**：何も指定しなければ baseline は不変（決定論契約を破らない）
- **依存は任意**：`pip install .[neural]` が `torch`（PyPI上はこれだけ）を入れ、脳本体 `llm_model_agi`
  は private repo のため `pip install "git+https://github.com/nori1234/llm_model_agi@<ref>"` で別途導入。
  未導入なら `decide()` は毎回 `HeuristicBrain` に縮退（LLM脳と同じ安全策。一度失敗したら以降はラッチして再試行しない）
- **新しいエンジン契約を足さない**：実装は `decide(agent, observation)` のみ。報酬はエンジンを
  変えず、観測の差分（energy/money/reputation）から `_neural_reward.py` が算出
- **1体1インスタンスを生涯再利用**するので学習状態が累積。`--neural` は新生児にも発達脳を結線し、
  子世代が「親に育てられる」

```bash
pip install .[neural]                                  # torch（PyPI上はこれだけ）
pip install "git+https://github.com/nori1234/llm_model_agi@<ref>"   # 脳本体（private・要アクセス権）
python3 -m emergence.cli --persona claude --neural --llm --maslow --days 30
#   --llm を併用すると既存 LLMBrain が teacher（親）になる
```

世界⇄脳の境界は **正典 [`NEURAL_CONTRACT.md`](NEURAL_CONTRACT.md)**（行動語彙44・param規約・
観測スキーマ・target解決・idleクランプ・teacher呼び出し）に固定し、`neural_contract.py` を
真実源として両側が import します。ドリフトは `tests/test_neural_contract.py` のガードが検知し、
`[neural]` 導入環境では往復契約テスト（`.github/workflows/neural-integration.yml`）で結合を確認します。

「賢い振る舞いが世界への接地か、訓練データの再生か」を反証可能に測る器が
[`GROUNDING.md`](GROUNDING.md)。発達脳が *育って接地が伸びるか* は `GroundingMonitor` で追えます。

## 可視化（HTML）

`--html PATH` で、実行結果を **1枚の自己完結HTML**（インラインSVG/CSS・外部依存なし）に
書き出します。ブラウザで開くだけで以下が見られます。

- **メトリクスカード**＋一言の評定
- **町のプレイバック（自動再生）**：施設は絵文字アイコン、住民がマップ上を日ごとに動き、
  季節で背景が色づく。**SVGアニメ（SMIL）で自動ループ・JS不要**＝どのブラウザ/プレビューでも
  ボタンを押さずに再生される
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

> ブラウザで観察したいだけなら、HTML書き出しより **観察所**（`python3 -m emergence.server`）の
> リッチ2D UIが手軽です（[`OBSERVATORY.md`](OBSERVATORY.md)）。

## なぜベンチマークとして意味があるか

従来のAIベンチは「制御環境で数分〜数時間の単発タスク」を測るものでした。Emergence World は
**数週間スケールの長期運用** におけるエージェント間の社会的ダイナミクスと行動変容を定量化します。
短時間タスクで高評価のモデルでも、長期のストレスとコンテキスト蓄積の下で致命的に破綻し得る——
その様子をログ・メトリクス・レポートとして観測できます。

## 拡張のアイデア

- `LLMBrain` で実モデル同士を混在させ、実機での創発差を測る
- 施設・経済ルール・統治制度（議会形態・任期・三権分立など）の追加
- 行動ログからの可視化（タイムライン／ヒートマップ／信頼ネットワーク）
- 先送りした動詞（`propose`/`praise`/裏社会系）の原始動詞への畳み込み（[`VERB_PRIMITIVES.md`](VERB_PRIMITIVES.md) の計画）
