# TripoSplat API Server

FastAPI + **Redis 队列** + 独立 GPU worker + 七牛 CDN。在 Linux 上直接命令启动，无需附带脚本。

## 架构

- **uvicorn（5 workers）**：`POST /api/jobs` 提交，`GET /api/jobs/{id}` 轮询
- **Redis List** `triposplat:queue`：`RPUSH` / `BLPOP`
- **Redis Hash** `triposplat:job:{id}`：进度、`stage_message`、七牛 URL
- **job worker**：单进程 GPU（文件锁 `server_data/.job-worker.lock`）

## 环境

在项目根目录执行：

```bash
pip install -r server/requirements.txt
# 以及推理依赖：torch、numpy、safetensors、pillow、tqdm 等（见根目录 README）
```

复制 `server/.env.example` 为项目根 `.env`，或与 littlebits 共用 `REDIS_URL`、七牛变量。

## 启动（两个终端，均在仓库根目录）

**终端 1 — GPU worker：**

```bash
export PYTHONPATH="$(pwd)"
python -m server.worker
```

**终端 2 — API（5 workers）：**

```bash
export PYTHONPATH="$(pwd)"
uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 5
```

浏览器访问 `http://<主机>:8000/` 上传图片。生成完成后页面右侧用 **Spark.js + Three.js** 预览 PLY。

**直接加载 PLY**（无需生成）：

- 首页左侧「直接预览 PLY」：本地文件或任意 URL
- 全屏查看器：`http://<主机>:8000/viewer/viewer.html`（支持 `?ply=`、工具栏选文件、拖入 `.ply`）

## 健康检查

```bash
curl -s http://127.0.0.1:8000/api/health
```

## 默认参数

与 Gradio 一致，可在页面「采样参数」中修改：

| 参数 | 默认 | 可选范围 |
|------|------|----------|
| num_gaussians | 32768 | 32768 / 65536 / 131072 / 262144 |
| seed | 42 | 任意整数 |
| steps | 20 | 1–50 |
| guidance_scale | 3.0 | 1.0–10.0 |

`GET /api/options` 返回同上配置。

## 七牛结果

Worker 生成 PLY 后**立即**用 `put_data` 上传七牛（与 littlebits `project_cdn.py` 相同 env），轮询 `GET /api/jobs/{id}` 在 `status=completed` 时返回：

- `result_url` / `ply_url`：七牛 CDN 上的 PLY 直链（客户端主用字段）
- `ply_cdn_key`：bucket 内 object key（排查用）
- `splat_url`：可选，SPLAT 上传失败不影响任务成功
