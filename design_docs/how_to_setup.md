# 環境構築

> 想定: Ubuntu 22.04。まずは CPU で動作確認できれば OK です（GPU は後回し）。

## 0) まずこれ（入口）

以下のコマンドで、環境構築から起動確認まで対話形式で一括実行できます。

```bash
sudo apt update && sudo apt install -y curl
curl -fsSL "https://raw.githubusercontent.com/AutomotiveAIChallenge/aichallenge-racingkart/main/setup.bash" | bash
```

- `bootstrap` では必要ステップを **y/N で確認**しながら進められます
- すべての確認が終わると自動でセットアップが進みます

## 1) setup.bash bootstrap が行うステップ

| #  | ステップ                      | 対応コマンド（個別実行時）       |
|----|-------------------------------|----------------------------------|
| 1  | 基本パッケージの導入          | `sudo apt install -y ...`        |
| 2  | Docker の導入                 | bootstrap 内で自動実行           |
| 3  | rocker の導入                 | `pip install rocker`             |
| 4  | docker グループへの追加       | `sudo usermod -aG docker $USER`  |
| 5  | リポジトリの取得              | `git clone ...`                  |
| 6  | 環境診断                      | `./setup.bash doctor`            |
| 7  | .env 作成（GPU/CPU 自動検出） | `./setup.bash env`               |
| 8  | Autoware ベースイメージ取得   | `./setup.bash pull image`        |
| 9  | AWSIM ダウンロード・展開      | `./setup.bash download awsim`    |
| 10 | 開発用イメージのビルド        | `./docker_build.sh dev`          |
| 11 | ワークスペースビルド          | `make autoware-build`            |
| 12 | 起動確認                      | `make dev` → 停止: `make down`   |

## 2) チェックリスト（上から順にやるだけ）

### (A) 診断する（最初に必ず）

- やること: 足りないものを洗い出す
- 代表コマンド:
  - `./setup.bash doctor`
- 完了の目安: "Docker" や "Repository" の欄で、次に何をすべきかが分かる

### (B) `.env` を作成する（GPU / CPU の選択）

- やること: `.env.example` をコピーして `.env` を作り、自分の環境に合わせる
- 代表コマンド:
  ```bash
  cp .env.example .env
  ```
- GPU / CPU の切り替え:
  - **GPU あり（デフォルト）**: `.env` をそのまま使う
    ```
    COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml
    ```
  - **CPU のみ**: `.env` から `COMPOSE_FILE` の行を削除またはコメントアウト
    ```
    #COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml
    ```
- 完了の目安: `.env` が存在し、`COMPOSE_FILE` の設定が自分の環境に合っている
