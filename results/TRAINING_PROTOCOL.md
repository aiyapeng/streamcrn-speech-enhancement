# Full28 big-DF K=5 预注册复现协议

冻结日期：2026-08-23。训练开始后不得依据 development 或历史 official test 改 batch、LR、DF order、损失、训练轮数或说话人切分。

## 数据

- 唯一训练来源：官方 VoiceBank-DEMAND `clean/noisy_trainset_28spk_wav`，11572 个同名配对。
- 归档固定值：clean 2486057279 bytes / MD5 `d2d5a45ec32f8fcbf201bde0447e20ba`；noisy 2830205201 bytes / MD5 `1fca9e8bafb8cd069f6653c6d92f9e9c`。
- fit：除 p226、p287 外的 26 名训练说话人。
- development：p226、p287；阶段选点使用固定哈希选择的 256 句，冻结后再跑两人的完整 development。
- VCTK-Corpus-0.92.zip 不是本实验的配对训练数据，不得替代 VoiceBank-DEMAND。
- official 824 句 test 已在上一模型迭代中消费，本实验禁止读取或重复评测。

## 模型与训练

- 随机初始化，不加载或继承任何外部/旧 checkpoint 权重。
- StreamCRN big：channels `[24,48,64,96,96]`，GRU hidden=256、layers=2，Deep Filtering order=5。
- 16 kHz、FFT/window=512、hop=256、`center=False`、sqrt-Hann WOLA。
- batch=32，segment=3 s，AdamW，LR=8e-4，warmup=2000 个真实 optimizer step，AMP。
- 固定 300 epochs；cosine 以完整 300 epoch 总有效步计算；每 20 个完整 epoch 做一次固定 development-selection 评估。
- Windows DataLoader workers=0，避免此前多进程无 traceback 退出。
- 仅允许因进程中断从同一 `last.pt` 精确恢复 optimizer/scheduler/scaler/global-step；OOM、非有限 loss 或配置/hash 变化必须停机审计。

## 选点与门禁

- candidate best：development PESQi 最大，且 mean STOIi ≥ 0、mean SI-SDRi > 0。
- development signal gate：mean PESQi ≥ 0；PESQi/STOIi/SI-SDRi 的 p10 均 ≥ 0；clean relative-L1 mean ≤ 8%。
- PESQ 2.8 是最终来源隔离测试口径，不直接套用到含噪基线不同的 development。
- signal/clean 门未全部通过：不创建或消费新盲测，不宣称最终达标。
- signal/clean 门全部通过：冻结权重、模型/配置/协议/评估器哈希后，另建来源隔离且单次消费的新盲测；不得回用 official 824 test。

## 部署门

- 6 项因果/流式/DF 回归测试必须全部通过。
- ONNX checker 通过，ORT/PyTorch max abs error ≤ 1e-5，连续状态循环通过，CPU RTF < 1。
- 当前架构算法时延为 32 ms；若产品合同要求 ≤20 ms，必须明确判为部署时延失败，不能用 RTF 掩盖。
