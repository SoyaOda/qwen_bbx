# Serena MCP 新規プロジェクトセットアップガイド

## 概要
Serena MCPは、セマンティックコード検索と編集機能を提供する強力なコーディングエージェントツールキットです。このガイドでは、新しいプロジェクトでSerena MCPをセットアップする手順を説明します。

## 前提条件

### 1. Python環境
- **必須バージョン**: Python 3.11.x（3.11.0〜3.11.9）
- 注意: Python 3.12以降はサポートされていません

### 2. UV パッケージマネージャのインストール
Serenaはuv（Rustで書かれた高速Pythonパッケージマネージャ）を使用します。

```bash
# macOS/Linux の場合
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows の場合（PowerShell）
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Homebrew（macOS）の場合
brew install uv
```

## セットアップ方法

### 方法1: Claude Code（推奨）

#### ステップ1: Claude Codeでの設定
Claude Codeを使用している場合、以下のコマンドでSerenaを追加：

```bash
claude mcp add serena -- uvx --from git+https://github.com/oraios/serena serena start-mcp-server --context ide-assistant
```

プロジェクト固有の設定を行う場合：
```bash
claude mcp add serena -- uvx --from git+https://github.com/oraios/serena serena start-mcp-server --context ide-assistant --project $(pwd)
```

#### ステップ2: Claude Codeの再起動
設定を反映させるため、Claude Codeを再起動します。

### 方法2: Claude Desktop

#### ステップ1: 設定ファイルの編集
Claude Desktopの設定に以下を追加：

**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Linux**: `~/.config/claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "serena": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/oraios/serena",
        "serena-mcp-server"
      ]
    }
  }
}
```

#### ステップ2: Claude Desktopの再起動
設定を反映させるため、Claude Desktopを再起動します。

### 方法3: Cursor

#### ステップ1: MCP設定ファイルの作成
以下のいずれかの場所に設定ファイルを作成：
- **グローバル設定**: `~/.cursor/mcp.json`
- **プロジェクト固有**: `.cursor/mcp.json`

```json
{
  "mcpServers": {
    "serena": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/oraios/serena",
        "serena-mcp-server"
      ]
    }
  }
}
```

## プロジェクトの初期化

### 1. プロジェクトのアクティベート
新しいプロジェクトを開いた後、AIアシスタントに以下のいずれかを伝えます：

```
「プロジェクト /path/to/my_project をアクティベートして」
```

または、既にアクティベート済みの場合：

```
「プロジェクト my_project をアクティベートして」
```

### 2. プロジェクト設定の確認
プロジェクトがアクティベートされると、以下のファイルが自動生成されます：

- **`.serena/project.yml`**: プロジェクト固有の設定
- **`serena_config.yml`**: グローバル設定（すべてのプロジェクトのリスト）

### 3. .gitignoreへの追加
`.serena`ディレクトリにはプロジェクトに関するメモリが保存されるため、`.gitignore`に追加します：

```bash
echo ".serena/" >> .gitignore
```

## プロジェクト設定のカスタマイズ

### `.serena/project.yml`の例

```yaml
name: "my_project"  # プロジェクト名（アクティベート時に使用）
path: "/path/to/my_project"
language: "python"  # 主要言語
read_only: false    # 読み取り専用モード（trueで編集無効）
disabled_tools: []  # 無効化するツールのリスト
```

### セキュリティ設定

#### 読み取り専用モード
コードの分析のみを行い、編集を防ぐ場合：

```yaml
read_only: true
```

#### 特定ツールの無効化
シェルコマンド実行を無効化する例：

```yaml
disabled_tools:
  - execute_shell_command
```

## 大規模プロジェクトの最適化

### インデックスの作成
大規模プロジェクトの場合、パフォーマンス向上のためインデックスを作成：

```bash
uvx --from git+https://github.com/oraios/serena serena project index
```

## 利用可能な主要ツール

### ファイル操作
- `read_file`: ファイルの読み取り
- `create_text_file`: 新規ファイル作成
- `replace_regex`: 正規表現による置換
- `list_dir`: ディレクトリ一覧
- `find_file`: ファイル検索

### シンボル操作（コード分析）
- `find_symbol`: シンボル検索
- `get_symbols_overview`: シンボル概要取得
- `replace_symbol_body`: シンボル本体の置換
- `find_referencing_symbols`: 参照検索
- `insert_before_symbol`/`insert_after_symbol`: シンボル前後への挿入

### プロジェクト管理
- `activate_project`: プロジェクトアクティベート
- `write_memory`: メモリ書き込み
- `read_memory`: メモリ読み取り
- `list_memories`: メモリ一覧

### その他
- `execute_shell_command`: シェルコマンド実行
- `search_for_pattern`: パターン検索

## トラブルシューティング

### 1. 接続の問題
macOSでClaude Codeが自動接続しない場合：
```
/mcp - Reconnect
```

### 2. ログの確認
- **Windows**: `%APPDATA%\Claude\logs\mcp.log`
- **macOS**: `~/Library/Logs/Claude/mcp.log`
- **Linux**: `~/.config/claude/logs/mcp.log`

### 3. 初期指示の読み込み
Claudeが自動的に指示を読み込まない場合：
```
「Serenaの初期指示を読んで」
```
または
```
/mcp__serena__initial_instructions
```

## ベストプラクティス

1. **バージョン管理**: 必ずGitなどのバージョン管理システムを使用
2. **バックアップ**: 重要な作業前にバックアップを作成
3. **メモリの活用**: プロジェクト固有の知識をメモリに保存
4. **段階的な導入**: まず読み取り専用モードで試してから編集機能を有効化

## 複数プロジェクトの管理

### プロジェクト切り替え
```python
# AIアシスタントに伝える
「プロジェクト project_a をアクティベート」
「プロジェクト /home/user/project_b をアクティベート」
```

### プロジェクト一覧の確認
`serena_config.yml`ファイルで管理されているプロジェクト一覧を確認できます。

## まとめ

Serena MCPは強力なコード分析・編集ツールですが、適切な設定とセキュリティ対策が重要です。プロジェクトの規模と要件に応じて、読み取り専用モードやツールの無効化を検討してください。

---

*最終更新: 2025年1月*
*Serena バージョン: v1.0.52以降推奨*