# 自分のPCで動かす（最小構成）

APIの従量課金なしで、**手元のPCだけ**でAIの住民を生かす最小手順です。GPUは不要
（CPUで動きます。小さいモデルなら数分で終わります）。

## 1. Ollama を入れて起動

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh
# macOS: https://ollama.com/download （または brew install ollama）

ollama serve        # 別ターミナルで起動したままにする（macOSアプリ版は自動）
```

## 2. 走らせる

```bash
./run_local.sh
```

これだけです。スクリプトが自動で:
1. Ollama が起動しているか確認
2. 小型モデル（既定 `llama3.2:3b`・約2GB）を未取得なら `ollama pull`
3. **記憶＋環境ON・永続化**で小規模シミュレーション（5人×3日）を実行

`memory-agent`（長期記憶ライブラリ）を未インストールなら先に:
```bash
git clone https://github.com/nori1234/memory-agent && pip install -e ./memory-agent
```

## 3. 重さの調整（環境変数で上書き）

| 目的 | コマンド |
|------|----------|
| **最軽量**（非力なPC） | `LLM_MODEL=qwen2.5:1.5b AGENTS=4 DAYS=2 ./run_local.sh` |
| 既定（バランス） | `./run_local.sh` |
| しっかり長期 | `LLM_MODEL=llama3.1:8b DAYS=15 ./run_local.sh` |
| 性格を変える | `PERSONA=gemini ./run_local.sh` |

## どれくらい時間がかかる？

LLM呼び出しは **1ティックずつ逐次**（住民が順番に行動して世界を書き換えるため）。
目安は `agents × days × ticks` 回の呼び出し：

- 既定（5×3×3＝45回）× CPUで3Bモデル（約1〜3秒/回）→ **およそ1〜3分**
- `--days 15` などにすると比例して伸びます

速くしたい場合の選択肢：より小さいモデル（`qwen2.5:1.5b`）、GPUのあるマシン、
あるいは vLLM で並列配信（将来）。

## セッションを跨いで思い出す

`run_local.sh` は記憶を `town.db` に永続化します。**もう一度実行すると、町は前回の
人生を覚えています**（別の住民が前回の出来事を思い出す）。記憶をリセットしたいときは
`town.db` を消すだけ。

## うまく動かないとき

| 症状 | 対処 |
|------|------|
| `Ollama not reachable` | 別ターミナルで `ollama serve` を起動 |
| `model not found` | スクリプトが自動 pull します。手動なら `ollama pull llama3.2:3b` |
| 応答が遅い・重い | `LLM_MODEL=qwen2.5:1.5b` に下げる／`AGENTS`/`DAYS`/`TICKS` を小さく |
| JSONが崩れて行動が変 | 小型モデルの限界。`llama3.2:3b` 以上を推奨。崩れた応答は自動でヒューリスティックにフォールバックするので**止まりはしません** |
| LLMを使わず確認したい | `python3 -m emergence.cli --persona claude --compare`（オフライン・即終了） |

## 仕組みの要点

- 住民は毎ターン「**思い出した記憶**」と「**季節・物価・災害**」を見て行動を決める
- エンドポイントが落ちても各住民は性格別ヒューリスティックに**自動フォールバック**して
  run は止まらない
- 既定はすべて**追加ライブラリ不要**。記憶機能だけ任意ライブラリ `memory-agent` を使う
