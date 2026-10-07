# Fast Local Voice Cloning：音色选择与快速本地克隆

为中文故事、知识动画和视频旁白提供一条已经试听验证的轻量本地路线：

**ZipVoice-Distill + 4 步 + 原生 1.05 倍速 + Apple MPS 主模型 + CPU Vocos。**

本方案不调用云端配音服务，不上传参考声音；也没有训练一份名为“A”的专属音色模型。
“A”是本地试听的参考录音编号，不是通用声音 ID。

## 交付内容

- [音色选择与复刻工作流](docs/VOICE_SELECTION_CN.md)：录音、选段、A/B 比较、音色保存与复现边界。
- [详细中文技术方案](docs/TECHNICAL_PLAN_CN.md)：架构、参数、安装、推理、停顿、实测、质量检查和公开范围。
- [配置模板](profiles/a-1.05-4steps.example.json)：已确认的 1.05 倍速、4 步等运行参数。
- [本地推理脚本](scripts/zipvoice_distill_local.py)：所有资源路径显式传入，拒绝覆盖已有输出。
- [兼容性约束](requirements-tested.constraints.txt)：本次测试过的关键依赖版本，不是完整锁文件。
- [原创示例稿](examples/story.txt)：早餐店视频服务故事；虚构教学示例。
- [去敏实测记录](evidence/benchmarks.json)：区分生成、初始化和分句额外开销。
- [公开素材边界](docs/PUBLICATION_SCOPE.md)：配置可以公开，参考录音、声音特征和权重不随本包上传。

本仓库公开发布方案、代码和参数模板；不包含参考录音、生成试听音频或模型权重。音色资产仍需单独决定。

## “快”与“复刻”的含义

- **快**：本机案例生成约一分钟旁白，包含加载约 11 秒；不是跨设备速度保证。
- **选择音色**：使用自己的不同参考录音，不是内置声音商城。
- **克隆**：通用模型结合本地参考进行零样本合成，不需要本次额外训练。
- **复刻**：保存必要素材和版本以复现相近结果；公开配置本身不等于公开音色。

当前提供命令行工程与技术文档；没有声称已经实现录音前端、常驻服务或多用户音色管理。

## 已确认的结果

在 Apple M5 / 32 GB / Python 3.12.14 / PyTorch 2.5.1 上，一次本地测试：

| 设置 | 成品时长 | 模型初始化＋生成 |
|---|---:|---:|
| 4 步、1.05 倍、全文提交 | 62.414 秒 | 11.045 秒 |
| 4 步、1.25 倍、同稿全文提交 | 52.456 秒 | 9.016 秒 |

1.05 倍试听得到使用者认可；这不等于普适的真人感评分。用时不含 MP3 转码、响度处理和检查。
更改硬件、正文、参考录音或依赖版本后需重新测量。

## 重要限制

1. **上传配置不会自动上传音色。** 复现同一声音还需要同一参考录音及准确逐字稿。
2. 当前第三方参考录音不包含在仓库中，也不发布其生成试听音频或声音特征。
3. 当前版本不支持用自然语言指令精确指定“每句停 0.5 秒”。仅改标点的实测改善有限。
4. 参考短录音无法保证情绪、重音或长篇说话习惯都与原声音一致。
5. 代码、模型权重和声音素材的许可是三件不同的事，不能相互替代。

## 开始使用

按[安装与运行](docs/TECHNICAL_PLAN_CN.md#4-环境与安装)准备本地资源，然后执行：

```bash
python scripts/zipvoice_distill_local.py \
  --config profiles/a-1.05-4steps.example.json \
  --repo-root vendor/ZipVoice \
  --model-dir models/zipvoice/zipvoice_distill \
  --vocos-dir models/vocos \
  --reference private/reference.wav \
  --reference-text-file private/reference.txt \
  --text-file examples/story.txt \
  --output local-output/story-1.05.wav
```

`private/`、`models/`、`vendor/`、`local-output/` 及音频/权重文件默认被忽略，不应强行加入公开提交。

## 测试

```bash
python -m unittest discover -s tests -v
```

配置测试不需要模型。推理需要下载模型、安装依赖并提供自己的已授权参考录音。

## 上游

- [ZipVoice 官方仓库](https://github.com/k2-fsa/ZipVoice)
- [本方案核对的固定源码版本](https://github.com/k2-fsa/ZipVoice/tree/2f7326fbfe999a3ad179e3f1af82a424d4a62819)
- [ZipVoice 模型](https://huggingface.co/k2-fsa/ZipVoice)
- [Vocos 24 kHz 模型](https://huggingface.co/charactr/vocos-mel-24khz)

本包是视频旁白集成方案，不是 ZipVoice 官方项目或音色素材合集。
