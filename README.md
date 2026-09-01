# Causal StreamCRN Speech Enhancement

基于复数 STFT 的因果流式语音增强模型。编码器仅在频率维下采样，以因果卷积和 GRU 建模时序信息，并使用 K=5 Deep Filtering 重建当前帧及历史谱信息。

## 目录

- `model.py`：StreamCRN、逐帧归一化和 Deep Filtering
- `stft.py`：sqrt-Hann WOLA 离线/流式前端
- `dataset.py`：在线混合与成对语音数据接口
- `losses.py`：SI-SDR 与压缩复数谱联合损失
- `scripts/`：训练、评测、推理、ONNX 导出和流式测试
- `checkpoints/`：最终 K=5 模型
- `results/`：固定协议和公开基准复用结果

## 环境

```bash
python -m venv .venv
pip install -r requirements.txt
```

## 流式推理

```bash
python -m scripts.infer_stream \
  --ckpt checkpoints/streamcrn_df_k5.pt \
  --in noisy.wav \
  --out enhanced.wav
```

运行因果性和流式一致性测试：

```bash
python -m scripts.test_streaming
```

## 已记录结果

完整 VoiceBank-DEMAND 28 说话人训练协议包含 11,572 对语音，按说话人划分 26 人训练集和 2 人开发集。公开 824 句基准复用评测结果如下：

| 指标 | 含噪 | 增强后 |
| --- | ---: | ---: |
| PESQ | 1.9797 | 2.6372 |
| STOI | 0.9219 | 0.9402 |
| SI-SDR | 8.49 dB | 18.52 dB |

算法时延为 32 ms。该 824 句结果属于公开基准复用评测，不作为新盲测结果。

VoiceBank-DEMAND 数据集未包含在仓库中。
