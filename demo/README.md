# 语音增强示例

`clean_reference.wav` 为 SemaAlign-TTS 项目生成的语音。脚本以固定随机种子加入宽带噪声和低频干扰，再通过仓库内的 StreamCRN 检查点逐帧增强。

```bash
python demo/run_demo.py
```

可直接对比：

- [`noisy_input.wav`](noisy_input.wav?raw=1)：3 dB 目标信噪比的带噪输入
- [`enhanced_output.wav`](enhanced_output.wav?raw=1)：流式增强输出
- [`clean_reference.wav`](clean_reference.wav?raw=1)：仅用于听感和 SI-SDR 对照

`metrics.json` 记录该示例增强前后的 SI-SDR 与算法时延。
