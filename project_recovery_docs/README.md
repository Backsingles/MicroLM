# MicroLM 轻量恢复文档包

生成日期：2026-05-23  
对应提交：`88c5dffa806d048c129671eeaa0c7c3b194377b0`  
工作区：`E:\MicroLM`

这个目录是 **文档级恢复包**，不是项目备份包。它不复制源码、模型权重、日志大文件、WSL 镜像或下载包，而是用结构化文档说明项目所有关键细节：项目目标、模块职责、数据 pipeline、训练流程、评测指标、部署方式、关键产物、已知坑、恢复步骤和文件清单。

## 适用场景

- 项目交接：让后来者理解项目从哪里开始、做到哪里、为什么这样做。
- 灾后重建：在有 Git 仓库或可重新拉取源码的前提下，用文档恢复项目结构、配置、数据准备和运行流程。
- 简历/答辩准备：快速查找技术路线、指标、实验结论和工程难点。
- 后续开发：明确哪些模块是自研实现，哪些是迁移到 HF/PEFT/vLLM 生态。

## 目录说明

| 文件 | 内容 |
|---|---|
| `00_RECOVERY_OVERVIEW.md` | 恢复总纲：项目是什么、恢复边界、最短阅读路径 |
| `01_ARCHITECTURE_AND_MODULES.md` | 仓库结构、核心模块、源码职责、关键设计 |
| `02_DATA_PIPELINES.md` | MiniMind 与 InstructIE 数据来源、清洗、派生、采样、格式协议 |
| `03_TRAINING_AND_INFERENCE.md` | tokenizer、pretrain、SFT、LoRA、推理、chat、KV Cache |
| `04_EVALUATION_AND_DEPLOYMENT.md` | 通用评测、结构化评测、schema repair、vLLM 部署和 benchmark |
| `05_ARTIFACTS_AND_RESULTS.md` | 关键产物路径、结果指标、报告索引、实验状态 |
| `06_DECISIONS_LIMITS_AND_BUGS.md` | 关键决策、能力边界、已知坑、修复教训 |
| `07_REPRODUCTION_RUNBOOK.md` | 从空环境恢复项目的步骤、命令、验收清单 |
| `08_RESUME_CLAIMS_EVIDENCE.md` | 简历项目描述逐条证据链、面试回答口径、数字风险和建议改写 |
| `manifests/` | Git 跟踪文件清单、提交统计、文件类型统计 |

## 最重要的三件事

1. 当前项目的权威状态是提交 `88c5dffa806d048c129671eeaa0c7c3b194377b0`。
2. 自研 MicroLM 主线用于证明训练/推理全链路能力；Qwen 主线用于结构化输出和部署。
3. 这不是二进制备份。模型权重、原始数据、WSL 镜像等需要从仓库提交、外部源或本地产物重新获得。

## 恢复优先级

如果只想快速恢复项目认知，按下面顺序读：

1. `00_RECOVERY_OVERVIEW.md`
2. `07_REPRODUCTION_RUNBOOK.md`
3. `01_ARCHITECTURE_AND_MODULES.md`
4. `02_DATA_PIPELINES.md`
5. `04_EVALUATION_AND_DEPLOYMENT.md`
6. `08_RESUME_CLAIMS_EVIDENCE.md`

如果要继续开发，额外读：

1. `06_DECISIONS_LIMITS_AND_BUGS.md`
2. `05_ARTIFACTS_AND_RESULTS.md`
3. `08_RESUME_CLAIMS_EVIDENCE.md`
4. `manifests/git_tracked_files.txt`
