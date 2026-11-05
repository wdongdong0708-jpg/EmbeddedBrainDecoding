# `edge_eeg`：面向资源受限端侧设备的脑电/脑磁语言感知多级级联解码系统

本项目对应硕士学位论文 **《面向资源受限端侧设备的脑电语言感知多级级联解码算法与系统实现》** 与嵌入式端侧 AI（TinyML / MCU 部署）工程实现，采用 **“本地轻量验收 + 云端 Colab 高速 NVMe 全量三轮训练”** 协同工作流。

## 核心三级级联架构 (Hierarchical Cascade Architecture)

1. **Stage 1：常驻脑电语音活动检测 (`MicroVADNet`)**
   - 仅用双侧颞叶 4~8 核心通道识别“语音感知态 vs. 静息态”，权重仅 **0.71 KB（INT8）**，配合滑动窗多帧投票与 Pre-roll 预取缓冲区抑制眼电/运动伪迹误唤醒，验证集 F1-macro 达 **85%+**。
2. **Stage 2：汉语四声声调与发音部位轻量级解码 (`Rep-TCN-EEGNet` / `TinyEdgeEEGNet`)**
   - 先由 306 通道 `TeacherConformerTCN` 在单被试 LibriBrain MEG 上探明理论上限并导出 `teacher_logits.pt`，再通过空间通道掩码剪枝至双侧颞叶 8 通道，结合跨通道知识蒸馏、结构重参数化（推理期峰值 Tensor Arena SRAM 降低 **73.8%**）与 99.5% 百分位截断逐通道 INT8 QAT，导出 C99 静态权重头文件 `model_weights.h`（**3.08 KB Flash**）。
3. **Stage 3：基于编码理论的 12 句封闭词表软判决维特比解码 (`LexiconViterbiDecoder`)**
   - 构造声调 + 声母发音部位 8 维联合指纹最小海明距离 $d_{\min}(F)=3$、平均海明距离 $\bar{d}_H(F)=5.92$ 的 12 条四字实用短句（及自采 EEG 阶段的 4 句正交极化子集 $d_{\min}=6, \bar{d}_H=7.17$），将单字 ~50% 的低信噪比分类率跃升至 71.5%~91.5%+ 的整句识别率。

## 环境变量契约

| 环境变量 | 本地默认值 | Colab 云端推荐值 | 用途 |
| :--- | :--- | :--- | :--- |
| `EDGEEEG_STUDIES` | `./workspace/data` | `/content/studies_links` | 原始脑电/脑磁数据集软链入口 |
| `EDGEEEG_CACHE` | `./workspace/cache` | `/content/local_cache` | 本地高速 NVMe 切窗特征缓存 |
| `EDGEEEG_RESULTS` | `./workspace/runs` | `/content/drive/MyDrive/EdgeEEG_Workspace/runs` | 模型权重 (`best.ckpt`)、`teacher_logits.pt`、`model_weights.h` 与评测日志 |

## 三轮全量训练与论文第 3、4、5 章实验命令

```powershell
# 0. 构建本地高速 NVMe 特征缓存（Colab 上可将 --num-runs 设为 8..16）
python -m edge_eeg cache --modality meg --stage vad --subject 001 --num-runs 8
python -m edge_eeg cache --modality meg --stage phoneme --subject 001 --num-runs 8

# 第一轮（第 3 章 Stage 1 全量 VAD）：训练 8 通道 MicroVADNet (INT8 QAT)，刷出 85%+ F1-macro
python -m edge_eeg train --modality meg --stage vad --arch student_tiny --channels 8 --subject 001 --num-runs 8 --epochs 15 --lr 1e-3 --qat --require-cache

# 第二轮（第 4 章 Stage 2 探顶 Teacher）：训练 306 通道 TeacherConformerTCN 并保存 teacher_logits.pt
python -m edge_eeg train --modality meg --stage phoneme --arch teacher_conformer --channels 306 --subject 001 --num-runs 8 --epochs 12 --lr 1e-3 --batch-size 64 --require-cache

# 第三轮（第 4/5 章 Stage 2 端侧 Student + 蒸馏 + INT8 QAT）：加载 teacher_logits.pt 蒸馏训练 8 通道 Rep-TCN-EEGNet 并导出 model_weights.h
python -m edge_eeg train --modality meg --stage phoneme --arch rep_tcn --channels 8 --subject 001 --num-runs 8 --epochs 16 --lr 2e-3 --distill --qat --require-cache

# 汇总输出论文第 3、4、5 章全部最终指标、量化/重参数化消融表与 Stage 3 维特比级联增益
python -m edge_eeg eval --modality meg --stage cascade --subject 001 --num-runs 8 --verify-codebook
```
