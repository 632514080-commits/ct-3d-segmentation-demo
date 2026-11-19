# 3D CT 病灶分割 Demo

这是一个面向高精密影像设备算法岗位的学习项目。它用合成的小动物 CT 体数据，走通“数据生成与版本管理 → 防泄漏切分 → 传统算法基线 → 3D U-Net → 评测与错误分析 → TorchScript/ONNX 部署”的完整链路。

它不是实际工作经历，也不能写成已经参与过商业 CT 产品。面试时应表述为：**为了理解目标岗位，我独立完成的学习型 Demo**。

## 一键运行

```bash
conda activate multimodal311
cd /Users/cathy/Documents/ChatGPT/人生规划问题/ct_3d_segmentation_demo
pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_demo.py --epochs 10 --device cpu
```

本项目默认使用 CPU。部分 macOS/PyTorch 组合虽然能检测到 Apple MPS，但不完整支持 `Conv3d`，会在模型第一层报错。只有确认当前版本支持 3D 卷积时，才尝试 `--device mps`。

首次完整运行会生成：

- `data/manifest.json`：数据版本、subject/case、train/val/test 切分和内容哈希；
- `artifacts/model.pt`：训练检查点；
- `artifacts/test_metrics.csv`：逐病例 Dice、IoU、Sensitivity；
- `artifacts/summary.json`：总体结果与最差病例；
- `artifacts/prediction.png`：切片、真值、预测可视化；
- `artifacts/model.ts`、`model.onnx`：部署模型；
- `artifacts/deployment.json`：输入规格、延迟和模型大小。

部署文件还会记录 ONNX 与 PyTorch 输出的最大绝对误差；只有误差小于 `1e-4` 才标记为一致性通过。导出成功不等于模型可用，数值对齐是上线门槛。

## 已验证运行结果

以下结果来自 `multimodal311` 环境、CPU、10 个 epoch 的完整运行：

| 项目 | 结果 |
|---|---:|
| 测试体数据数 | 10 |
| 3D U-Net 平均 Dice | 0.8036 |
| 传统 HU 阈值基线平均 Dice | 0.9573 |
| 最差病例 | mouse-005-v2 |
| TorchScript CPU 推理延迟 | 约 83.7 ms |
| TorchScript 模型大小 | 约 0.115 MB |
| ONNX 导出 | 成功 |
| ONNX 与 PyTorch 最大绝对误差 | 9.54e-7 |
| ONNX 数值一致性 | 通过 |

传统算法在这组合成数据上优于深度学习模型并不是失败。这里的病灶灰度规则明确、数据量很小，HU 阈值天然占优；它说明产品方案应先建立简单、可解释的基线，再判断复杂模型是否带来真实增益。真实 CT 数据存在伪影、噪声、跨设备域偏移和复杂边界，不能据此推断传统算法在真实场景中仍会保持同样优势。

![3D CT lesion segmentation example](artifacts/prediction.png)

## 你必须真正理解的六件事

1. **为什么是 3D 而不是 2D**：体素上下文能帮助区分单张切片里外观相似的结构，但显存与标注成本更高。真实项目常用 patch、混合精度和滑窗推理。
2. **为什么按 subject 切分**：同一只动物的多次扫描高度相似；若分别落入训练集和测试集，指标会虚高。这比随机按图像切分更重要。
3. **为什么 Dice 不够**：小病灶中类别极不平衡；还要看 sensitivity、假阳性、病灶级召回及最差病例。空标签病例要单独分析。
4. **组合方案如何设计**：传统 HU 阈值提供快速、可解释基线；3D U-Net处理复杂边界；低置信度病例进入人工修正，修正结果回流下一版训练集。
5. **如何保证可复现**：固定随机种子，保存 manifest、数据哈希、代码版本、参数、模型权重和逐病例结果；禁止用测试集调阈值。
6. **如何接入产品**：定义固定输入 `[N,C,D,H,W]`、归一化规则、输出阈值和版本号；先导出 TorchScript/ONNX，再由统一推理服务做超时、显存、日志与版本回退。

## 15 分钟演示顺序

1. 打开 `manifest.json`，讲 subject 级切分和数据泄漏；
2. 展示 `traditional_baseline`，讲可解释基线与形态学后处理；
3. 展示 `TinyUNet3D` 和 Dice+BCE，讲小目标与类别不平衡；
4. 打开 `prediction.png`、`test_metrics.csv`，重点讲最差病例而不是只报均值；
5. 打开 `deployment.json`，说明接口规格、延迟、模型大小与回退策略；
6. 主动说明局限：合成数据不代表真实设备域，下一步需接入真实 DICOM/NIfTI、spacing 重采样、窗宽窗位、标注一致性与跨设备外部验证。

## 和岗位要求的映射

| 岗位要求 | Demo 中的证据 |
|---|---|
| CT/3D 分割 | 3D U-Net、体数据输入、逐体素预测 |
| AI + 传统算法 + 人工修正 | HU 阈值/形态学基线 + U-Net；README 描述低置信度人工复核闭环 |
| 数据治理与防泄漏 | manifest、数据版本/哈希、subject 级切分、泄漏单测 |
| 训练评测与错误分析 | 固定种子、逐病例指标、最差病例定位、可视化 |
| 部署与版本回退 | TorchScript/ONNX、输入契约、延迟与模型大小；保留 checkpoint |
| 软件协作 | 明确输入输出，可进一步封装 FastAPI/C++ 推理接口 |

## 面试时可以这样介绍

“我的主经历是数据与模型全生命周期治理，不是医学影像研发。为了验证迁移能力，我做了一个学习型 3D CT 分割 Demo。数据层面我特意模拟了同一 subject 多次扫描，并按 subject 切分，避免泄漏；算法层面先用 HU 阈值和形态学做可解释基线，再训练轻量 3D U-Net；评测不只看平均 Dice，还保留逐病例结果和最差病例；工程层面导出 TorchScript 和 ONNX，并记录输入规格、延迟与模型版本。这个 Demo 不能替代真实 CT 项目经验，但让我能快速进入真实数据的 spacing、标注一致性、跨设备泛化和部署问题。”

## 高频追问与答题边界

- **真实 CT 为什么不能直接这么做？** 合成数据没有设备噪声、金属伪影、重建核差异和跨中心域偏移。真实流程必须读取 DICOM 元数据、统一方向与 spacing、处理窗宽窗位，并做跨设备测试。
- **小病灶 Dice 为什么波动大？** 少量边界体素就会显著改变交并比，因此同时看病灶级召回、表面距离和临床/检测容忍度。
- **人工修正怎么回流？** 保存原预测、修订 mask、操作者与时间戳；复核后进入新数据版本，按 subject 去重，再进行回归评测，不能直接覆盖旧标签。
- **TensorRT 没做怎么办？** 如实说当前在 macOS 上验证 TorchScript/ONNX；TensorRT 需要 NVIDIA 环境，迁移时会固定 CUDA/TensorRT 版本、验证数值误差，并比较 FP32/FP16 的速度与精度。
- **你的优势是什么？** 不要说自己已经是影像专家。强调你做过数据治理、模型监控、可解释性、跨团队落地，短板是影像领域积累，但已经用可运行 Demo 建立了共同语言。

## 建议的学习节奏

- 第 1 天：跑通项目，理解体数据 shape、HU、mask、Dice；
- 第 2 天：逐行读数据生成、切分与泄漏测试；
- 第 3 天：读 3D U-Net、loss 和评测，故意制造一个错误并定位；
- 第 4 天：理解 TorchScript/ONNX 输出，练习 15 分钟演示；
- 第 5 天：按上面的高频追问录音复盘，确保不背稿、不夸大。
