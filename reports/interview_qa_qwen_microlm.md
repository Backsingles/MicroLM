# MicroLM / Qwen LoRA 结构化抽取项目面试拷打问答稿

> 适用场景：根据简历项目被深挖时使用。回答风格按面试口述整理，重点是把两条技术线、数据口径、LoRA 参数量、评测口径和部署指标讲清楚。

## 先统一三个口径

1. **MicroLM 和 Qwen 是两条并行线，不是前后依赖。**  
   MicroLM 线证明我能从 tokenizer、模型、训练循环、SFT、LoRA、KV Cache 自己搭一条闭环；Qwen 线证明我能把同一套方法迁移到 HF / PEFT / vLLM 生态，并落到结构化信息抽取任务。

2. **`1MB / 8.3MB` 说的是 adaptor 存储，不是参数个数。**  
   MicroLM LoRA 约 262K 可训练参数，占 31.7M 的 0.83%；Qwen LoRA 是 2,179,072 可训练参数，占 1.55B 的 0.14%，adaptor 文件约 8.3MB。

3. **评测指标要分清楚。**  
   `4 模型 × 40 prompt × 4 指标` 是结构化 prompt 行为评测；`valid JSONL` 是内容级字段/键值 F1 评测。简历里如果写 `1000 条验证样本`，需要准备对应结果文件；目前仓库可追溯的正式内容级结果是 200 条 valid JSONL，Field F1 约 0.784，Pair F1 约 0.673。

---

## 一、项目定位

### 1. 标题写 Qwen2.5-1.5B，但第一条又说 31.7M MicroLM。这两个模型是什么关系？

这是项目的两条并行验证线。MicroLM 是我从零搭的 31.7M 小模型链路，用来证明底层能力：BPE tokenizer、decoder-only Transformer、RoPE、SwiGLU、预训练、SFT、LoRA 和 KV Cache 都自己实现。Qwen2.5-1.5B 是迁移线，用成熟基座和 HF / PEFT / vLLM 工具栈，把同样的训练-评测-部署方法落到结构化信息抽取任务上。

面试时我会强调：MicroLM 不是 Qwen 的前置模型，也不是用 MicroLM 初始化 Qwen；它们共享的是工程方法和评测思路。

### 2. 这个项目最终目标到底是什么？

最终目标不是单纯训练一个小模型，也不是只跑一次 LoRA，而是打通一条完整 LLM 工程闭环：数据处理、tokenizer、预训练、SFT、LoRA、自动评测、推理优化和服务化部署。

后半段我把任务聚焦到结构化信息抽取，是因为它比通用聊天更适合作为简历项目：JSON 可解析率、字段 F1、Pair F1、Strict% 都能自动量化，能证明训练确实改变了模型行为。

### 3. 如果一句话描述你的贡献，你希望面试官记住什么？

我从零实现了一条 MicroLM 训练闭环，并把这套方法迁移到 Qwen2.5-1.5B 上，用 LoRA 完成结构化抽取后训练，最终形成了可评测、可部署、可复现的训练-评测-推理服务闭环。

---

## 二、数据与 Pipeline

### 4. MiniMind 的 141 万条语料和 InstructIE 的 171K 样本分别用于什么阶段？有没有混用？

没有混用。MiniMind 用在 MicroLM 自研线，主要做通用语言建模预训练和后续 SFT 验证；InstructIE 用在 Qwen 迁移线，专门做结构化信息抽取 LoRA SFT。

我这样拆是因为两类数据目标不同：MiniMind 提供通用语言分布，适合验证小模型训练闭环；InstructIE 带 schema 和 gold JSON，适合训练结构化输出。

### 5. SHA1 切分具体是对什么做 SHA1？为什么不用随机 seed split？

预训练阶段对样本文本或稳定样本标识做 SHA1 哈希，再按哈希值确定落入 train 还是 valid。这样每条样本的归属是确定性的，多次清洗或重跑不会因为 shuffle 顺序改变而漂移。

随机 seed split 也能复现，但更依赖输入顺序；哈希切分对数据追加、重跑和增量处理更稳，也更利于避免同一文本跨集合泄漏。

### 6. 六步 Pipeline 里哪一步最容易引入数据泄漏？

最容易出问题的是采样和切分附近，也就是过滤、分层采样、valid 切分这几步。如果先派生任务再切分，同一条原始样本的不同派生版本可能同时进入 train 和 valid，造成泄漏。

所以我的处理思路是先做标准化和跨集重复过滤，再做质量分层和受控采样；最终 valid 从候选集中独立切分，并固定随机种子。严格说，精确重复能处理，语义近似泄漏还需要 embedding 相似度去重继续增强。

### 7. 28.5K 训练样本和 1.5K 验证样本是怎么从 171K 来的？为什么只取这么多？

InstructIE 原始中文 train 约 171K。先经过标准化、硬过滤、per-topic P99 软过滤、质量分层，然后只保留 high quality 样本。再从每条样本派生四类任务：ie_extraction、text_to_json、format_following、schema_repair。最后按任务类型、topic、质量和复杂度分层采样 30K，并按 95/5 切成 28.5K train 和 1.5K valid。

只取 30K 是一个质量优先的选择。LoRA 对数据格式和分布很敏感，直接全量训练会放大 topic 不均衡、长尾噪声和跨集重复；先用干净、均衡、可审计的数据把 schema-following 学稳，收益更可控。

### 8. JSON 校验 100% 是校验训练标签 JSON 合法，还是模型输出 JSON 合法？

这里要分两层说。数据 Pipeline 里的 JSON 校验 100%，指训练目标，也就是 assistant 的 gold output 能被 `json.loads` 正常解析，是合法 JSON。它保证训练标签干净。

模型输出的 Parse% 是评测或部署阶段的指标，例如 vLLM 稳定性验证里 Parse% 100%，指模型实际输出能解析成 JSON。训练标签 100% 合法不等于模型输出一定 100% 合法，这两个口径不能混。

---

## 三、Tokenizer 与预训练

### 9. 6400 词 BPE tokenizer 为什么是 6400？对中文信息抽取会不会太小？

6400 是模型容量和 token 效率之间的折中。MicroLM 只有 31M 参数，如果词表太大，embedding 和 lm_head 会吃掉太多参数；6400 能覆盖常用中文字符和高频子词，同时把参数预算留给 Transformer 层。

它对通用小模型训练是合适的，但对结构化 JSON 输出确实偏小，容易导致序列更长、符号和字段 tokenization 不够高效。这也是后来把结构化抽取迁移到 Qwen 的原因之一。

### 10. 316M tokens 预训练一个 31.7M 参数模型，训练了多少 epoch？batch size、context length、学习率、硬件是什么？

正式配置里 train tokens 是 316,095,996，batch size=8，context length=512，max_iters=50,000。按每步 8×512 计算，训练过程实际看到约 204.8M token，相当于约 0.65 个 epoch 的 token 更新量。不过数据加载是随机 block sampling，不是严格顺序扫一遍，所以我会说是“约 0.65 epoch 等效 token budget”。

学习率是 2e-4，min_lr=2e-5，warmup_iters=2000，cosine decay，weight_decay=0.1，gradient clipping=1.0，CUDA 单卡训练。

### 11. 预训练 loss 从 8.85 降到约 2.4，这是 train loss 还是 val loss？per-token CE 吗？有没有 perplexity？

这是 next-token prediction 的 per-token cross entropy，训练日志里 train 和 val 都有记录。起点大约是 train_loss 8.844、val_loss 8.856；后期 val loss 在 2.x 区间波动，简历里写约 2.4 是收敛量级口径。

困惑度可以用 `exp(loss)` 估算：8.85 对应约 7000，2.4 对应约 11。这个下降说明模型确实学到了语料 token 分布，但它不是大模型级别的强通用能力。

### 12. 怎么判断 31.7M MicroLM 学到了语言能力，而不是只是在数据上拟合？

我看三类证据。第一，train/val loss 都明显下降，不只是 train loss 单边下降。第二，SFT 后在固定 prompt 人工评估中，pretrain 到 SFT 的回答相关性和指令跟随有提升。第三，KV Cache、REPL、生成脚本能在未见 prompt 上生成基本连贯文本。

但我也会承认边界：31M 小模型容量有限，结构化 JSON Parse% 为 0%，复杂指令、长上下文和严格 schema-following 不是它的强项。

---

## 四、LoRA 训练

### 13. LoRA SFT 仅训练 0.83% 参数，为什么括号写 1MB？口径怎么算？

0.83% 是 MicroLM LoRA 的可训练参数占比，不是文件大小。MicroLM 基座约 31.47M 参数；在 attention 的 q/k/v/output 投影层加 rank=8 LoRA 后，可训练参数约 262K，占约 0.83%。

1MB 指 adaptor 存储量。262K 参数如果按 FP32 存，大约 1.05MB；如果按 FP16 会更小。所以面试里要说“0.83% 可训练参数，adaptor 约 1MB”，不能说“1MB 参数”。

### 14. Qwen LoRA 0.14% 参数、8.3MB，对应多少 trainable parameters？rank、alpha、target modules 是什么？

Qwen LoRA 可训练参数是 2,179,072，占 1,545,893,376 总参数的约 0.14%。配置是 r=8，alpha=16，dropout=0.05，target modules 是 `q_proj`, `k_proj`, `v_proj`, `o_proj`。

8.3MB 是 adaptor 存储量。2.18M 参数按 FP32 约 8.7MB，序列化后报告里约 8.3MB。

### 15. 为什么 target 这些模块？有没有试过只训 q_proj/v_proj，或者加 FFN？

我选择 attention 四个投影层，是保守但有效的 PEFT 配置。结构化抽取主要依赖指令、schema、输入文本之间的对齐关系，attention 投影层对这种对齐行为影响直接；同时参数量可控，风险低。

没有系统跑完整的 target module ablation。只训 q/v 可能更省参数，但容量更低；加 gate/up/down FFN 可能增强表达能力，但参数和过拟合风险也更高。这个项目优先跑通可靠闭环，所以先选 q/k/v/o。

### 16. 2000 steps 后训练够吗？有没有过拟合迹象？继续训会怎样？

够作为正式收口训练。Qwen LoRA 正式日志里，val loss 从 step 100 的约 0.4025 降到 step 2000 的约 0.1553，后期仍在缓慢下降，没有出现 train loss 降而 val loss 反弹的明显过拟合信号。

继续训练可能还有小幅收益，但边际收益下降。真正要提升字段和值的正确性，优先级不一定是继续堆 step，而是做 schema alias 归一化、实体规范化、失败样本分析和约束解码。

如果简历保留“1.115 → 0.843”，要准备对应日志；仓库正式 run 更建议写“val loss 0.4025 → 0.1553”或“验证 loss 持续下降，正式 run 降幅约 61%”。

### 17. 有没有做 full fine-tune、prefix tuning、QLoRA 或不同 rank 的 ablation？如果没有，为什么？

有一部分对比：MicroLM 全参 SFT vs MicroLM LoRA，Qwen base vs Qwen LoRA，hardcase refinement 前后对比，部署前后稳定性对比。没有完整做 full fine-tune、prefix tuning、QLoRA 和 rank 网格。

原因是项目目标是建立可解释闭环，不是刷超参榜。Qwen2.5-1.5B 用 FP16 LoRA 单卡能训通，QLoRA 的 4-bit 量化会引入额外数值变量；full fine-tune 成本更高，也更容易过拟合。后续可以补 rank=4/8/16、target modules 和 QLoRA 的 ablation。

---

## 五、评测

### 18. 字段级 JSON F1 从 0.057 到 0.541，base 为什么这么低？是不是 prompt 没写好？

base Qwen 强在通用理解，不等于强在特定 schema 的严格结构化抽取。它可能能输出合法 JSON，但字段命名、实体嵌套方式、schema 对齐和 gold JSON 不一致，严格 F1 会很低。

prompt 确实会影响结果，所以我用统一 prompt、统一解码参数、统一评分脚本来比较 base 和 LoRA。更稳妥的说法是：base 低不代表 Qwen 不懂文本，而是没有适配本项目的 InstructIE 输出风格和评分协议。

### 19. Parse% 100%，但字段 F1 只有 0.541，说明模型能输出 JSON 但字段错很多。主要错在哪里？

是的，Parse% 只说明格式合法，不代表内容正确。字段 F1 或 Pair F1 低，主要来自几类问题：实体边界不一致、字段 alias 不一致、值规范化不一致、复杂关系漏抽、列表或嵌套结构和 gold 不一致，以及模型多输出幻觉字段。

所以我会同时报 Parse%、Exact Match、Field F1 和 Pair F1。只报 Parse% 会显得包装指标。

### 20. “4 模型 × 40 Prompt × 4 指标”和“1000 条验证样本”是什么关系？

这是两套评测，不能混成一个实验。`4 模型 × 40 Prompt × 4 指标` 是 prompt-level 结构化行为评测，用 40 条人工设计的结构化 prompt 检查 JSON 可解析率、缺字段率、幻觉字段率、Strict% 等。

验证样本评测是 data-level 内容评测，从 valid JSONL 里抽样，对模型输出和 gold JSON 做 Field F1 / Pair F1 / Exact Match。简历里建议写成“在 40 条结构化 prompt 和 held-out valid JSONL 上分别评测”，不要让面试官误以为 40 prompt 每条又跑了 1000 样本。

### 21. 四个指标具体是什么？怎么算？

结构化 prompt 评测的四个主指标是：

1. JSON 可解析率：清理 markdown fence 后能否被 `json.loads` 解析。
2. 缺字段率：schema 要求字段在输出中没有出现的比例。
3. 幻觉字段率：输出了 schema 不允许字段的比例。
4. Strict Schema 命中率：JSON 可解析、必填字段齐全、无多余字段、枚举约束满足。

另外还有辅助的 Alias-Strict%，会把“姓名/name”“位于/location”这类字段别名归一化后再算严格命中。

### 22. 字段级 F1 是 micro-F1 还是 macro-F1？空字段和多值字段怎么处理？

当前 `evaluate_qwen_valid_jsonl.py` 是逐样本计算 precision、recall、F1，然后对样本取平均，所以更接近 macro average，而不是把所有样本的 TP/FP/FN 汇总后算 micro-F1。

解析失败时 F1 记 0。pred 和 gold 都为空时 PRF 记 1，这是避免空目标样本被错误惩罚。多值字段会 flatten 成路径或 key-value pair 集合比较；字段顺序不影响字典比较，但列表值如果语义无序，当前评分仍偏严格，后续可以做排序或集合化规范。

### 23. 怎么保证评测脚本没有偏向 LoRA 模型？

核心是统一输入和统一评分。base 和 LoRA 使用同一批 prompt、同一套 schema、同一套 generation 参数、同一套 `json.loads` 和 flatten 逻辑。评测脚本不读取模型名称来改变评分规则。

同时我保留 base、LoRA、MicroLM SFT、MicroLM LoRA 四模型对比，让 LoRA 的收益不是单模型自说自话。需要承认的一点是 prompt 可能来自任务开发过程，所以更严格的做法是再加一套冻结的 hidden test prompt。

---

## 六、推理部署

### 24. KV Cache 加速 3.86x 是在哪个模型上测的？输入输出长度是多少？

这是在自研 MicroLM 上测的，不是 Qwen/vLLM。checkpoint 是 `outputs/sft_baseline/ckpt_final.pt`，CPU float32，batch size=1，用 no-cache generate 和 cache generate 对比 wall-clock。

输入 prompt 长度覆盖 16/32/64/128/256 tokens，生成长度覆盖 32/64/128/256 tokens，共 20 组，每组预热 1 次、正式跑 3 次取平均。平均加速 3.86x，最大 9.08x 出现在 prompt=256、gen=256。

### 25. vLLM 部署用了什么参数？tensor parallel、max model len、dtype？

默认部署加载 `outputs/qwen_lora_merged_final`，OpenAI-compatible API，host=`0.0.0.0`，port=8000，tensor parallel size=1，max_model_len=4096，dtype=auto。当前脚本还默认关闭 FlashInfer sampler，以适配 WSL/RTX 50 系列环境里 `nvcc` 依赖问题。

推理请求建议 temperature=0、top_p=1、max_tokens=256，并使用 `response_format={"type":"json_object"}` 做格式约束。

### 26. “低延迟服务化”低到什么程度？P50/P95、QPS、显存占用是多少？

我不会只说“低延迟”，会报已经测到的数字：vLLM benchmark 覆盖 5 组配置，0 errors；单请求 128/64 的 TTFT 约 0.53s，512/128 的 TTFT 约 0.43s，1024/256 的 TTFT 约 1.24s；吞吐约 12 到 31 tok/s，4 并发和 8 并发都稳定完成。

同时我会承认：当前报告里更完整的是平均耗时、TTFT、吞吐和错误率，P50/P95 和显存峰值还没有固化成主表。如果面试官追生产级 SLA，我会说这是后续需要补的压测项。

### 27. smoke 5/5 是什么测试？五个 case 全过能证明服务稳定吗？

smoke 5/5 包括 health check、simple chat、structured extraction、multi-turn、structured_response_format 五项，证明服务能启动、能响应、能对话、能做核心结构化任务，并且支持 JSON response_format。

它只能证明最小链路可用，不能证明生产稳定。生产稳定还需要并发压测、长输入、非法 schema、超时、重试、异常 JSON、显存压力和持续运行测试。我在项目里补了 benchmark 和 40 条稳定性验证，但严格生产级还可以继续扩。

---

## 七、真实性压力测试

### 28. 如果让你手写 RoPE 核心公式，你能写出来吗？

能。RoPE 对 Q/K 的偶奇维度成对做旋转。对位置 `m` 和第 `i` 个二维子空间：

```text
theta_i = base^(-2i / d)
R(m, i) = [[cos(m theta_i), -sin(m theta_i)],
           [sin(m theta_i),  cos(m theta_i)]]
```

把 `[x_2i, x_2i+1]` 乘上这个旋转矩阵。注意力里用旋转后的 `q_m` 和 `k_n` 做内积，结果会包含 `m-n` 的相对位置信息，这就是 RoPE 适合自回归模型的原因。

### 29. SwiGLU 相比 GELU FFN 多了什么？参数量怎么变？

普通 GELU FFN 可以写成：

```text
FFN(x) = W2 * GELU(W1 * x)
```

SwiGLU 是门控结构：

```text
SwiGLU(x) = W_down * (SiLU(W_gate * x) ⊙ W_up * x)
```

它多了一条 gate 分支，用逐元素乘法控制信息通过。参数上，普通 FFN 是 `2 * d_model * d_ff`，SwiGLU 是 `3 * d_model * d_ff_swiglu`。通常会把 SwiGLU 的中间维度设得更小，让总参数量接近普通 4d FFN。MicroLM 里 `d_model=512, d_ff=1344`，SwiGLU 约 3×512×1344=2.06M，和 4d GELU FFN 的 2×512×2048=2.10M 接近。

### 30. LoRA 为什么能只训练低秩矩阵？推理时怎么 merge？merge 前后输出是否完全一致？

LoRA 的假设是任务适配所需的权重更新近似低秩。原始线性层是 `y = Wx`，LoRA 冻结 `W`，只训练低秩增量：

```text
y = Wx + (alpha / r) * B A x
```

训练后可以把增量合并回权重：

```text
W' = W + (alpha / r) * B A
```

eval 模式下没有 dropout，数学上 merge 前后输出等价；实际工程里因为 dtype、量化、kernel 计算顺序不同，可能有极小浮点误差。

### 31. vLLM 的 PagedAttention 解决了什么问题？和普通 KV Cache 的区别是什么？

普通 KV Cache 是按请求缓存历史 token 的 K/V，能避免重复计算，但服务化并发时容易遇到显存碎片、连续大块分配和不同请求长度不齐的问题。

PagedAttention 把 KV cache 像操作系统分页一样切成 block，用 block table 管理每个请求的缓存。它解决的是 KV cache 的显存管理问题，配合 continuous batching，可以让新请求动态加入、完成请求动态退出，提高 GPU 利用率。

### 32. 如果模型输出非法 JSON，你的修复策略是什么？

评测时不能偷偷修，解析失败就记 parse=false，Field F1 / Pair F1 记 0，这样指标诚实。

线上服务可以分层处理：第一层用 temperature=0、`response_format=json_object` 降低非法 JSON 概率；第二层做直接 `json.loads`、markdown fence 清理、首个 JSON span 提取；第三层做 schema projection、alias normalize、类型归一化和缺字段补齐；第四层对失败样本走一次 self-repair prompt。再往后可以引入 grammar constrained decoding 或 JSON schema constrained decoding。

---

## 最后背诵版

这个项目最容易被拷打的点是口径。我的回答策略是：MicroLM 负责证明从零实现能力，Qwen 负责证明迁移和部署能力；MiniMind 和 InstructIE 不混用；`1MB/8.3MB` 是 adaptor 存储，不是参数个数；Parse% 只代表 JSON 合法，不代表字段正确；4×40 prompt 评测和 valid JSONL F1 评测是两套实验；vLLM 的 smoke 只能证明链路通，真正性能要看 benchmark 和稳定性验证。

