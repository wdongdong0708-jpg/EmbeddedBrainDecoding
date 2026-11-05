# `edge_eeg`：面向资源受限端侧设备的脑电/脑磁语言感知多级级联解码系统

本项目对应硕士学位论文 **《面向资源受限端侧设备的脑电语言感知多级级联解码算法与系统实现》** 与嵌入式端侧 AI（TinyML / MCU 部署）工程实现，采用 **“本地零数据轻量 Smoke 调试 + 云端 Colab 高速 NVMe 缓存训练”** 协同工作流。

## 核心三级级联架构 (Hierarchical Cascade Architecture)

1. **Stage 1：常驻脑电语音活动检测 (`MicroVADNet`)**
   - 仅用双侧颞叶 4~8 核心通道识别“语音感知态 vs. 静息态”，权重小于 4 KB（INT8），配合滑动窗多帧投票与 Pre-roll 预取缓冲区抑制眼电/运动伪迹误唤醒。
2. **Stage 2：汉语四声声调与发音部位轻量级解码 (`TinyEdgeEEGNet` / `Rep-TCN-EEGNet`)**
   - 先由 306 通道 `TeacherConformerTCN` 在单被试 LibriBrain MEG 上探明理论上限，再通过空间通道掩码剪枝至双侧颞叶 8 通道，训练 100% 兼容 TFLite Micro / CMSIS-NN 的深度可分离网络。
3. **Stage 3：基于编码理论的 12 句封闭词表软判决维特比解码 (`LexiconViterbiDecoder`)**
   - 精心构造声调 + 声母发音部位 8 维联合指纹最小海明距离 $d_{\min}(F)=4$、平均海明距离 $\bar{d}_H(F)=5.89$ 的 12 条四字实用短句（及自采 EEG 阶段的 4 句正交极化子集），将单字 ~50% 的低信噪比分类率跃升至 85%~95%+ 的整句识别率。

## 环境变量契约

| 环境变量 | 本地默认值 | Colab 云端推荐值 | 用途 |
| :--- | :--- | :--- | :--- |
| `EDGEEEG_STUDIES` | `./workspace/data` | `/content/studies_links` | 原始脑电/脑磁数据集软链入口 |
| `EDGEEEG_CACHE` | `./workspace/cache` | `/content/local_cache` | 本地高速 NVMe 切窗特征缓存 |
| `EDGEEEG_RESULTS` | `./workspace/runs` | `/content/drive/MyDrive/EdgeEEG_Workspace/runs` | 模型权重 (`last.ckpt`/`best.ckpt`) 与评测日志 |

## 本地快速验证命令 (Local Smoke Test)

```powershell
# 1. 验证第 2.1 节 12 句黄金词表海明距离矩阵与第 5.1 节 Stage 3 维特比纠错增益
python -m edge_eeg eval --stage cascade --verify-codebook

# 2. 本地无数据集 3 秒跑通 Stage 1 (8通道 VAD) 与 Stage 2 (8通道音素/声调) Smoke Test
python -m edge_eeg smoke --modality meg --stage vad --arch student_tiny --channels 8 --epochs 3
python -m edge_eeg smoke --modality meg --stage phoneme --arch student_tiny --channels 8 --epochs 3

# 3. 本地验证 306 通道 Teacher 大模型前向/反向传播与显存统计
python -m edge_eeg smoke --modality meg --stage phoneme --arch teacher_conformer --channels 306 --epochs 2
```
