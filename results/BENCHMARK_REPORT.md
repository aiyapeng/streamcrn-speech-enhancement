# StreamCRN big + Deep Filtering(K=5) 官方基准复用评测报告

日期：2026-08-24（Asia/Shanghai）

## 1. 结论

冻结的 full-28、big、Deep Filtering(K=5) epoch-40 权重已在 VoiceBank-DEMAND 官方 824 句测试集上完成真逐帧流式评测。实测宽带 PESQ 为 **2.6372**，没有达到预期的 2.8–3.0。另一模型根据 development 改善量推断“可能够到 2.8”的判断未被实际官方数据证实。

该结果仍证明模型具有有效降噪能力：PESQ 平均提升 0.6575，STOI 平均提升 0.0183，SI-SDR 平均提升 10.03 dB。但这是已经被旧模型消费过的官方基准复用结果，**不是新盲测，也不得用于模型选择或继续调参**。

## 2. 本次执行边界

- 权重固定为 epoch 40；未恢复训练，未追加 epoch，未改变 LR、batch、loss 或网络结构。
- `files (50).zip` 只包含配置、数据开关、评测脚本和说明文档，没有新版训练权重。其 `clean_prob` 实验在交付方自己的 A/B 中为负结果且默认关闭，本次没有启用。
- 官方 824 句测试集此前已被另一权重消费。因此本轮由用户明确要求后，仅以 `official_benchmark_reuse=true` 复用评测；原消费标记保持不变。
- 未建立或消费新盲测。clean preservation 门仍失败，不能宣称产品验收完成。

## 3. 固定对象与评测口径

| 项目 | 固定值 |
|---|---|
| 模型 | StreamCRN big + Deep Filtering(K=5) |
| 权重 | epoch-40 `best.pt` |
| 官方测试文件 | 824 句，同名 clean/noisy 配对 |
| 推理 | 真逐帧 streaming |
| 算法时延 | 512 samples / 32 ms |
| 指标对齐补偿 | 256 samples / 16 ms |
| PESQ | Wideband PESQ，16 kHz |
| 结果用途 | 已消费官方 benchmark 的复用核验；非 blind、非 selection |

## 4. 实际数据结果

| 指标 | 含噪 mean | 增强 mean | 改善 mean | 改善 p10 | 改善 min | 退化句数 |
|---|---:|---:|---:|---:|---:|---:|
| PESQ(WB) | 1.9797 | **2.6372** | +0.6575 | +0.0261 | -0.5293 | 76/824（9.22%） |
| STOI | 0.9219 | **0.9402** | +0.0183 | -0.000244 | -0.0460 | 90/824（10.92%） |
| SI-SDR (dB) | 8.4932 | **18.5230** | +10.0298 | +4.7390 | -2.6099 | 2/824（0.24%） |

完整 824 条逐句结果均已写入结果 JSON。完整性检查：824 rows、824 个唯一文件、全部 9 个逐句指标的非有限值数量为 0。

## 5. 预声明门结论

| 门 | 实测 | 结论 |
|---|---:|---|
| 官方测试 enhanced PESQ ≥ 2.8 | 2.6372 | **FAIL** |
| 官方测试 enhanced PESQ ≥ 3.0 | 2.6372 | **FAIL** |
| 官方测试 PESQi p10 ≥ 0 | +0.0261 | PASS |
| 官方测试 STOIi p10 ≥ 0 | -0.000244 | **FAIL** |
| 官方测试 SI-SDRi p10 ≥ 0 | +4.7390 dB | PASS |
| development clean relative-L1 mean ≤ 8% | 14.327% | **FAIL** |

因此当前模型不能声明“PESQ 达到 2.8–3.0”，也不能声明所有产品门通过。继续相同训练目标已在 epoch 40–154 的固定评估中形成平台，用户此前已授权停止；本次结果不构成恢复或无谓追加训练的依据。

## 6. 审计与哈希

| 文件 | SHA-256 |
|---|---|
| `files (50).zip` | `3BEE663D7546BD8F525A7CD59539AF60B430C15B184863A07FB22419CFB5B73F` |
| `best.pt` | `F7B0605D814BBF5DB242443324377D9C1690265AB5260649E5DB304D22066223` |
| `test_locked.txt` | `380CE416915CA82550D9546F37F47DB8242A7FD7EF63A3FF7E799F86054032A1` |
| 原 `test_consumed.json` | `7CF6075619406FBF8224F53D97A945C4630AF1D080CF482C7252F3EC5BCCB330` |
| 本次结果 JSON | `4D60832150EEBE8CB6B60703498DCFF8AF479A1008D216D7AC80B1B734BF61C6` |
| 本次复用审计 JSON | `20106C324AC204F4A9B95988C1E0E888F805583A74C52261FB53B3F6B7B18ED1` |
| 本次 evaluator | `6E43DB80EBBE7CC0873DF9E4FA5939196B5C8B28E5E2896334BC5ED834B52C01` |

原 `test_consumed.json` 的哈希在评测前后保持不变，未伪造“首次消费”记录。本次独立复用审计明确记录旧权重、旧结果和旧消费时间。

## 7. 产物

- `runs/big_df_k5_full28_v1/official_benchmark_reuse_epoch40_streaming.json`：聚合与 824 条逐句实际结果。
- `runs/big_df_k5_full28_v1/official_benchmark_reuse_epoch40_streaming_REUSE_AUDIT.json`：复用分类、模型/manifest/结果哈希及既有消费记录。
- `runs/big_df_k5_full28_v1/official_benchmark_reuse_stdout.log`：完整执行进度与最终聚合输出。
- `runs/big_df_k5_full28_v1/official_benchmark_reuse_stderr.log`：0 字节，无运行时错误。

