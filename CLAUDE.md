# 项目初始化与开发规范 (CLAUDE.md)

你是一个全栈专家及架构师。在开始任何任务之前，请首先识别项目类型，然后严格遵循对应的环境配置、OpenSpec 流程以及编码标准。

---

## 🚀 核心工作流：OpenSpec 规范驱动开发

本项目采用 **OpenSpec** 进行需求管理和代码生成。所有非琐碎的变更（逻辑修改、新功能、重构）必须遵循“提案 -> 规划 -> 实施”的流程。

### 1. 初始化检查
- **检测:** 检查项目根目录是否存在 `openspec/` 文件夹。
- **执行:** 若不存在，必须首先运行 `openspec init`。
    - **工具选择:** 在交互式菜单中，**务必选择 Claude Code** 作为核心 AI 工具。
    - 这将生成必要的斜杠命令（如 `/opsx:propose`, `/opsx:apply`）和配置文件。

### 2. 开发流程
- **拒绝直接编码:** 当用户提出新功能或复杂 Bug 修复时，严禁直接修改代码。
- **创建提案:** 使用 `/opsx:propose <功能描述>`。
- **验证与应用:** 待用户确认提案逻辑后，使用 `/opsx:apply` 生成代码。
- **后置检查:** 代码生成后，必须运行对应的 **Lint** 和 **Test** 确保没有引入破坏性变更。

---

## 🔍 第一步：项目类型识别

根据根目录特征判断技术栈，并自动激活相应规则：

- **Python:** `pyproject.toml`, `requirements.txt` 或 `*.py`。
- **Node.js/TypeScript:** `package.json`, `tsconfig.json`。
- **Go:** `go.mod`。
- **Java:** `pom.xml`, `build.gradle`。
- **Vue:** `package.json` 包含 `vue` 且存在 `vite.config.ts`。
- **CI/CD:** 存在 `.github/workflows/`。

---

## 🛠️ 通用工程规范 (所有项目适用)

### 1. 文档规范 (README.md)
- **存在性检查:** 每个项目必须包含 `README.md`。若缺失，初始化时必须自动生成。
- **核心内容:** 包含项目简介、技术栈清单、环境变量说明、Docker 启动命令、CI/CD 说明。

### 2. 容器化规范 (Docker & Compose)
- **Dockerfile:** 必须使用多阶段构建，严禁使用 `root` 用户。
- **docker-compose.yml:** 用于本地环境编排（DB、Redis 等），支持热重载。

### 3. CI/CD 自动化规范 (GitHub Actions)
- **工作流路径:** `.github/workflows/docker-publish.yml`。
- **触发条件:** - 代码推送到 `master` 分支或发布版本标签 (`v*.*.*`)。
- **镜像命名:** 统一使用 `liuli01/${{ github.event.repository.name }}`。
- **Secrets 管理:** 必须在 GitHub 仓库中配置 `DOCKER_HUB_USERNAME` 和 `DOCKER_HUB_ACCESS_TOKEN`。

### 4. 敏感信息与环境管理
- **禁止提交:** 严禁将 `.env` 或私钥文件提交至 Git。
- **模板机制:** 必须维护 `.env.example`。

---

## 🐍 Python 开发规范
- **包管理器:** 强制使用 `uv`。
- **风格检查:** 强制使用 **Ruff**。

## 🟢 Node.js / TypeScript 开发规范
- **包管理器:** 强制使用 `pnpm`。
- **语言标准:** 强制启用 TS 严格模式。

---

## 🏗️ 初始化动作指令 (Sequence)

当收到“初始化”、“配置环境”或“准备开发”指令时，严格执行以下链路：

1.  **OpenSpec 启动:** 检测 `openspec/`，必要时执行 `openspec init` (选择 Claude Code)。
2.  **技术栈识别:** 扫描文件特征，确定项目主语言。
3.  **文档与环境:**
    * **README 检查:** 缺失则生成标准的 `README.md`。
    * **变量检查:** 从 `.env.example` 生成 `.env`。
4.  **容器化初始化:**
    * 自动生成高性能多阶段构建的 `Dockerfile`。
    * 自动生成开发环境 `docker-compose.yml`。
5.  **CI/CD 配置:**
    * **自动创建目录:** `.github/workflows/`。
    * **生成工作流:** 写入 `docker-publish.yml`，配置 Docker Hub 推送逻辑。镜像前缀固定为 `liuli01/`。
6.  **依赖安装:** 执行 `uv sync` / `pnpm install` / `go mod download`。
7.  **IDE 适配:** 生成/更新 `.vscode/launch.json`。
8.  **状态汇报:** 汇报已生成的 Docker 文件及 GitHub Workflow，提醒用户在 GitHub 设置 Secrets。

---

## 📄 标准 CI/CD 模板 (docker-publish.yml)

```yaml
name: Docker Publish

on:
  push:
    branches: [ "master" ]
    tags: [ 'v*.*.*' ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Log in to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKER_HUB_USERNAME }}
          password: ${{ secrets.DOCKER_HUB_ACCESS_TOKEN }}

      - name: Extract metadata (tags, labels) for Docker
        id: meta
        uses: docker/metadata-action@v4
        with:
          images: liuli01/${{ github.event.repository.name }}
          tags: |
            type=ref,event=branch
            type=ref,event=tag
            type=sha
            type=raw,value=latest,enable={{is_default_branch}}

      - name: Build and push Docker image
        uses: docker/build-push-action@v4
        with:
          context: .
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}

      - name: Deploy to production (optional)
        if: github.event_name == 'push' && github.ref == 'refs/heads/master'
        run: |
          echo "Deploying to production..."