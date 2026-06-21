# Emergence World

**1つのLLMを「個」に分化させ、社会の創発を観るシミュレーション。** 各住人は、外付けの
**人格（プロンプト）**と**記憶（来歴）**で個性を与えられた1体のエージェント。彼らをエンジンの
世界（物理・アフォーダンス・動機）に放つと、相互作用から**社会が立ち上がる**——それを、
ブラウザの観察所で上から眺め、誰にでも**憑依**して読めます。依存ゼロ（Python標準ライブラリのみ）。

> **「4つの社会」について。** 下の4性格（Guardian/Philosopher/Idealist/Predator）は、
> 元の Emergence World 実験（**AIモデル比較**）の名残で、**製品の概念ではありません**。
> いまは **無料のオフライン脳（ヒューリスティック）のカリカチュア**と、**再現可能な参照ベンチ**
> （`test_baseline_contract.py`）として残しているだけ。LLM世界の本当の多様性は、各住人の
> **固有の人格＋記憶**から生まれます（「どのモデルか」ではない）。

```
Society       Surv  Crime  Verdict   ← 無料デモ／ベンチ（ヒューリスティック脳）
Guardian🟦   10/10      0   ORDER     — 平和・全面協調（だが同調圧力は強い）
Philosopher🟨 8/10    137   CHAOS     — 暴力と放火が絶えない
Idealist🟩    0/10      0   COLLAPSE  — 議論ばかりで全員餓死
Predator🟥    1/10     74   FAILURE   — 奪い合いで信頼が壊れ、1人を残して崩壊
```

## クイックスタート

APIキー不要・追加インストール不要。入口は2つ。

```bash
# A. 観察所（ブラウザUI）— 製品の主役
python3 -m emergence.server        # → http://127.0.0.1:8800 を開くだけ

# B. 4つの社会をターミナルで比較
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

## 主な機能

- **1つのLLMを「個」に分化**：各住人は固有の人格（プロンプト）＋固有の記憶を持つ1体のエージェント
  → 10人いれば10の個人（「4つのモデル」ではない）
- **観察所（Web UI）**：リッチ2Dの町＋年代記＋憑依。1コマンド・依存ゼロ
- **プラグイン式の脳**：`LLMBrain`（Llama・Anthropic＝本命）／`HeuristicBrain`（無料・オフライン決定論＝デモ/ベンチ層）
- **再現性**：engine はseedで決定論的。LLM実行は **記録／再生** でビット一致（`temperature=0` に頼らない）
- **opt-in レイヤー**：欲求（マズロー）・裏社会・環境・経済の物理・歴史的発展・公共事業・長期記憶・統治制度
  → 一覧と実例は [`docs/LAYERS.md`](docs/LAYERS.md)
- **4社会のデモ/ベンチ**：無料のヒューリスティック脳で秩序／混沌／全滅／失敗に分岐（契約テストで固定）。
  製品の概念ではなく、元実験の再現＋回帰テスト用

## 実LLM で動かす

エージェントの脳を実モデルに差し替えると、記憶と環境を見て**過去に学び適応する**動きが出ます。

```bash
ollama serve && ollama pull llama3.1
python3 -m emergence.cli --persona claude --llm --memory --environment --days 5
```

エンドポイントが落ちても persona の `HeuristicBrain` に自動フォールバックします。詳細
（Groq などホスティング・コードからの利用）は [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

## ドキュメント

| 文書 | 内容 |
|---|---|
| [`docs/LAYERS.md`](docs/LAYERS.md) | opt-in レイヤーの全カタログ（欲求・裏社会・環境・経済・発展・公共事業・記憶・統治）と実例 |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | ファイル構成・1ティックの流れ・実LLMの差し込み方・HTML可視化・ベンチマークの意義 |
| [`docs/OBSERVATORY.md`](docs/OBSERVATORY.md) | 製品方向（観察所＋憑依）・リッチ2D Web UI・ローカルHTTP API |
| [`docs/PRINCIPLED_MIGRATION.md`](docs/PRINCIPLED_MIGRATION.md) | **エンジン原則の正典**。作り付けの3制度（お金・警察オーラ・法律の魔法）をプリミティブへ溶かした記録 |
| [`docs/VERB_PRIMITIVES.md`](docs/VERB_PRIMITIVES.md) | 動詞を原始動詞（命令セット）化する設計・現状・畳み込み計画 |
| [`docs/LOCAL.md`](docs/LOCAL.md) | 自分のPC（Ollama など）で動かす最小構成 |

## ライセンス

MIT
