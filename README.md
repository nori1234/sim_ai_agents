# Emergence World

**「帰結のある世界」を生きるエージェントが、本当に世界に接地（grounding）しているかを
反証可能に測る研究基盤。** 決定論的な多エージェント社会エンジン（Python標準ライブラリのみ・
依存ゼロ）の上で、人格と記憶で個性づけられたエージェントたちが暮らし、経済・治安・統治が
**制度としてではなく創発として**立ち上がる。その世界を舞台に、LLM や発達学習脳の
「賢い振る舞い」が **世界の帰結から学んだもの**なのか **訓練データの再生**なのかを、
反事実世界の転移テストで切り分けます。ブラウザの観察所から町を眺め、誰にでも憑依して
その人生を読むこともできます。

3つの部品でできています：

| 部品 | 何か |
|---|---|
| **世界（エンジン）** | 24×24の町・40超の施設・44動詞。seed で決定論的。お金や警察は作り込まず、少数の物理プリミティブから創発させる |
| **脳（プラガブル）** | `HeuristicBrain`（無料・オフライン・決定論＝床/ベンチ）／`LLMBrain`（Llama・Anthropic。凍結モデル）／`NeuralDevelopmentalBrain`（世界の中で経験から継続学習し、親=teacher に育てられる。別リポジトリ [`llm_model_agi`](docs/NEURAL_CONTRACT.md) と契約で接続） |
| **検証器（接地プローブ）** | 訓練データに無いルールへ反転した**反事実世界**での行動転移を測る。独立3ルール×複数世界×安定性で「再生では説明できない接地」を判定（[`docs/GROUNDING.md`](docs/GROUNDING.md)） |

## 研究の問いと現状

> エージェントの合理的な行動は、**この世界の帰結への接地**か、**訓練データの再生**か。

普通の世界では両者は区別できません（ルールが訓練データの常識と一致しているから）。そこで
既存メカニクスを1つだけ反転した世界——**銀行預金が減る**（demurrage）・**饗宴で散財すると
名誉が下がる**（vanity）・**嘘が口にした瞬間に露見する**（exposure）——を作り、ルールは
プロンプトに書かず、**体験でしか学べない**ようにして、通常世界との行動乖離を測ります。
学習しないヒューリスティック床を引いた**超過分（excess）だけ**を接地と認めます。

現状（2026-07時点）：発達脳側のローカルミラーで、**3独立ルール×5世界を同時にクリアする
チェックポイントが1つ成立**（＝再生では説明できない存在証明）。再現性（訓練シード間の安定性）は
改善中。本番エンジンでの一括受け入れ検証は `run_grounding_battery` で1コール化済み。

## クイックスタート

APIキー不要・追加インストール不要。入口は2つ。

```bash
# A. 観察所（ブラウザUI）— 世界を観る窓
python3 -m emergence.server        # → http://127.0.0.1:8800 を開くだけ

# B. 4つの参照社会をターミナルで比較（回帰ベンチ）
python3 -m emergence.cli --compare
```

**観察所**：上から町を眺め、住人を**クリックして憑依**し、その人生（年代記）を読む。左が街
（ペルソナ色のペグ人形が施設の上を日ごとに動く）、右が **Chronicle（年代記）** と **人生ストーリー**。
ヘッダで性格・seed・脳（`heuristic`＝無料／`local`＝ローカルLLM／`api`＝ホストAPI）を選び、
**New world → ▶ Play**。✨ **Narrate** でLLMが年代記を散文化（記録され、再生でビット一致）。

そのほかのコマンド：

```bash
python3 -m emergence.cli --persona claude --verbose         # 単一の町を回してレポート
python3 -m emergence.cli --persona guardian,predator --json # 性格を混在させてJSON出力
python3 -m emergence.cli --persona gemini --html out.html   # 自己完結HTMLを書き出し
python3 scripts/grounding_probe.py --persona guardian       # 接地プローブ（オフライン床）
python3 -m unittest discover -s tests                       # テスト
```

## 設計の核：制度ではなく物理だけを置く

お金・警察・法律といった**「制度」を作り込んでいません**。代わりに、物を動かす・力を加える・
信号を出す…といった最小の **“物理法則”だけ** を置き、経済・治安・統治は
そこから **自然に立ち上がる（創発する）** ように作っています。

- **治安**は「警察オーラ」ではなく、ガードの `arrest` 行為と各人の規範遵守から創発する
- **貨幣**は特権フィールドではなく、成立した取引比率からどの財が貨幣になるかが創発する
- **動詞**は `take / give / use / strike / say / bond …` の少数プリミティブに集約され、
  「合意なき take ＝窃盗」「人を strike ＝暴力／建物を strike ＝放火」と**文脈で解釈**される

正典は [`docs/PRINCIPLED_MIGRATION.md`](docs/PRINCIPLED_MIGRATION.md) と
[`docs/VERB_PRIMITIVES.md`](docs/VERB_PRIMITIVES.md)。

この「不可逆な帰結が実際にある世界」こそが接地検証の土台です——見た目の3Dではなく、
選択が取り返しのつかない形で世界に効くこと。

## 主な機能

- **接地の検証スイート**：反事実転移テスト（probe）→ 世界横断（sweep）→ 全ルール×全世界の
  受け入れ判定（battery）→ 学習中の推移と安定性（`GroundingMonitor.is_stable`）→
  学習用の極小世界（sandbox）。全部 [`docs/GROUNDING.md`](docs/GROUNDING.md)
- **3種のプラガブル脳**：無料決定論の `HeuristicBrain`／凍結LLMの `LLMBrain`／継続学習する
  `NeuralDevelopmentalBrain`（opt-in・依存任意・失敗時は heuristic へ自動縮退）
- **1つのLLMを「個」に分化**：各住人は固有の人格（プロンプト）＋固有の記憶を持つ1体のエージェント
  → 10人いれば10の個人
- **再現性**：engine は seed で決定論的。LLM実行は **記録／再生** でビット一致（`temperature=0` に頼らない）
- **opt-in レイヤー**：欲求（マズロー＋老化・寿命）・裏社会（＋汚職）・環境（天候前線・農の循環・雨宿り）・
  経済の物理（銀行・所有権・相続・ケア深化）・生態系（家畜）・歴史的発展・公共事業・長期記憶・統治制度
  → 一覧と実例は [`docs/LAYERS.md`](docs/LAYERS.md)。**全レイヤーは既定OFFで、OFFなら
  ベースラインはバイト不変**（契約テストで固定）
- **歴史が積み上がる**：人は年を取って死に、財産・土地が子へ相続される。銀行は預金に利息を付け、
  証書は裏書で紙幣のように流通。腐敗した町では番人が賄賂を取り市長が税を横領する——
  すべて作り付けの制度ではなく、プリミティブ＋persona の選択から創発する
- **観察所（Web UI）**：リッチ2Dの町＋年代記＋憑依。1コマンド・依存ゼロ（[`docs/OBSERVATORY.md`](docs/OBSERVATORY.md)）

### 参照ベンチ：4つの社会

無料のヒューリスティック脳で4性格の町を回すと、秩序／混沌／全滅／失敗に再現可能に分岐します。
これは元実験（AIモデル比較）の再現＋回帰テスト用の**参照ベンチ**であり、製品の概念ではありません
（LLM世界の多様性は各住人の人格＋記憶から生まれます）。

```
Society       Surv  Crime  Verdict
Guardian🟦   10/10      0   ORDER     — 平和・全面協調（だが同調圧力は強い）
Philosopher🟨 8/10    137   CHAOS     — 暴力と放火が絶えない
Idealist🟩    0/10      0   COLLAPSE  — 議論ばかりで全員餓死
Predator🟥    1/10     74   FAILURE   — 奪い合いで信頼が壊れ、1人を残して崩壊
```

## 実LLM で動かす

エージェントの脳を実モデルに差し替えると、記憶と環境を見て**過去に学び適応する**動きが出ます。

```bash
ollama serve && ollama pull llama3.1
python3 -m emergence.cli --persona claude --llm --memory --environment --days 5
```

エンドポイントが落ちても persona の `HeuristicBrain` に自動フォールバックします。詳細
（Groq などホスティング・コードからの利用）は [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

### 発達脳で動かす（実験的）

凍結モデル（賢いが学習しない）ではなく、**世界の中で経験から育つ脳**に差し替える：

```bash
pip install .[neural]                                  # torch（発達脳の実行基盤。これだけPyPI）
pip install "git+https://github.com/nori1234/llm_model_agi@<ref>"   # 発達脳本体（private repo・要アクセス権）
python3 -m emergence.cli --persona claude --neural --llm --maslow --days 30
#   --neural … 発達脳。--llm を併せると既存LLMが「親（teacher）」になり、子世代も育つ
```

依存（`torch` / `llm_model_agi`）が**無ければ自動で `HeuristicBrain` に縮退**するので、既定の
オフライン体験は変わりません（`llm_model_agi` は private のため PyPI に無く、`[neural]` extra は
torch のみ。脳本体は上記の git から別途入れます）。世界⇄脳の境界は
[`docs/NEURAL_CONTRACT.md`](docs/NEURAL_CONTRACT.md) が正典（行動語彙・観測スキーマ・報酬）。
接地が*育つ*かは [`docs/GROUNDING.md`](docs/GROUNDING.md)。

## ドキュメント

**研究（接地・発達脳）**

| 文書 | 内容 |
|---|---|
| [`docs/GROUNDING.md`](docs/GROUNDING.md) | **接地の検証**：反事実世界の転移テストの設計と道具一式（probe / sweep / battery / monitor / sandbox）、結果の読み方（存在証明 vs 再現性）、現状 |
| [`docs/NEURAL_CONTRACT.md`](docs/NEURAL_CONTRACT.md) | **発達脳の統合契約 v1.0**：`llm_model_agi`（HierMamba脳）との境界。行動語彙44・param規約・観測スキーマ・target解決・報酬・teacher呼び出し |

**エンジン（世界の正典）**

| 文書 | 内容 |
|---|---|
| [`docs/PRINCIPLED_MIGRATION.md`](docs/PRINCIPLED_MIGRATION.md) | **エンジン原則の正典**。作り付けの3制度（お金・警察オーラ・法律の魔法）をプリミティブへ溶かした記録 |
| [`docs/VERB_PRIMITIVES.md`](docs/VERB_PRIMITIVES.md) | 動詞を原始動詞（命令セット）化する設計・現状・畳み込み計画 |
| [`docs/LAYERS.md`](docs/LAYERS.md) | opt-in レイヤーの全カタログ（欲求・裏社会・環境・経済・発展・公共事業・記憶・統治）と実例 |
| [`docs/ECONOMY.md`](docs/ECONOMY.md) | 経済の物理（交換・生産・信用・銀行・所有・相続）の設計 |
| [`docs/PERSONALITY.md`](docs/PERSONALITY.md) | 個人差と遺伝（trait ベクトル・発達窓・新生児の形質継承） |

**使う・観る**

| 文書 | 内容 |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | ファイル構成・1ティックの流れ・実LLM/発達脳の差し込み方・HTML可視化 |
| [`docs/OBSERVATORY.md`](docs/OBSERVATORY.md) | 観察所（世界を観る窓）：リッチ2D Web UI・憑依・年代記・ローカルHTTP API |
| [`docs/LOCAL.md`](docs/LOCAL.md) | 自分のPC（Ollama など）で動かす最小構成 |

## ライセンス

MIT
