# 灾害推文分类：TF-IDF Baseline 与 BERT 微调

[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-blue)](https://www.kaggle.com/code/jiatyan/baseline-tf-idf-logistic-and-bert)

本仓库对应一个完整的 Kaggle Notebook，用于参加 [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started) 竞赛。Notebook 中包含了从数据探索、简单基线模型到 BERT 微调的完整流程，并针对多 GPU 训练、阈值调优等实际问题给出了解决方案。竞赛排名前3%

<img width="1855" height="898" alt="image" src="https://github.com/user-attachments/assets/9ed0e753-23bc-4806-8ebf-cca230ad1e9c" />

## 📋 项目简介

- **任务**：判断一条推文是否描述了真实灾难（二分类）
- **评价指标**：F1 分数
- **数据**：约 10,000 条标注推文，字段包括 `text`、`keyword`、`location`
- **方案**：
  1. 基于 TF-IDF + 逻辑回归的快速基线
  2. 使用 HuggingFace Transformers 微调 BERT（`bert-base-uncased`）
  3. 在验证集上自动搜索最佳分类阈值，优化 F1

## 🔗 Notebook 地址

所有代码和详细说明均在一个 Kaggle Notebook 中：  
[https://www.kaggle.com/code/jiatyan/baseline-tf-idf-logistic-and-bert](https://www.kaggle.com/code/jiatyan/baseline-tf-idf-logistic-and-bert)

## 📁 Notebook 结构

Notebook 按顺序包含以下主要部分：

1. **数据探索分析（EDA）**
   - 目标变量分布
   - 缺失值检查
   - 文本长度（字符数、单词数）
   - 关键词与灾难比例的关系
   - 灾难/非灾难推文词云

2. **Baseline：TF-IDF + Logistic Regression**
   - 文本清洗（去除 URL、HTML、标点等）
   - TF-IDF 向量化（n-gram 1-2，最大特征 10000）
   - 使用交叉验证评估并调整阈值
   - 生成提交文件

3. **BERT 微调**
   - 数据预处理（最小清洗，保留标点和大小写）
   - 自定义 `TweetDataset` 与 `DataLoader`
   - 加载 `bert-base-uncased` 并添加分类头
   - 多 GPU 训练（`DataParallel`）与损失处理
   - 学习率调度（warmup + 线性衰减）
   - 早停与最佳模型保存
   - 在验证集上搜索最佳阈值
   - 测试集预测与提交

4. **常见错误与解决方案**  
   Notebook 中注释详细，且针对以下问题给出了处理：
   - `AdamW` 导入错误
   - `encode_plus` 属性错误
   - 多 GPU 下损失向量问题
   - 模型加载时键名不匹配

## 🚀 如何使用

### 在 Kaggle 上运行

1. 打开 Notebook 链接，点击 **Copy & Edit** 创建自己的副本。
2. 确保 Notebook 的 Accelerator 设置为 **GPU**（推荐 T4 x2 或 P100）。
3. 依次运行所有单元格即可。最终会生成 `submission.csv` 文件，可直接提交。

### 在本地运行（需自行调整路径）

若想在本地环境运行，需要安装依赖：

```bash
pip install pandas numpy scikit-learn matplotlib seaborn torch transformers
```

并将数据文件放在指定路径（或修改代码中的路径）。

## 🧠 主要技术细节

### 1. 文本预处理
- **Baseline**：较为激进，去除标点、URL、HTML 实体，只保留字母和空格。
- **BERT**：最小化清洗，仅去除 URL、HTML 标签和 @提及，保留 `#` 标签、标点和大小写，以充分利用 BERT 的 tokenizer。

### 2. 模型训练
- **优化器**：PyTorch 自带的 `AdamW`（注意不是从 transformers 导入）
- **学习率**：2e-5，配合线性预热和衰减
- **批量大小**：32
- **最大序列长度**：128（基于文本长度分布）
- **早停**：监控验证 F1，若连续 1 个 epoch 无提升则停止

### 3. 多 GPU 训练注意事项
- 使用 `nn.DataParallel` 包装模型。
- 损失函数返回的是向量，必须调用 `.mean()` 转为标量再进行反向传播。
- 保存模型时使用 `model.module.state_dict()` 去除 `module.` 前缀，以便后续单卡加载。
- 推理阶段加载模型时，需重新创建**未包装**的模型实例。

### 4. 阈值调优
- 由于评价指标是 F1，默认阈值 0.5 并非最佳。
- 在验证集上使用 `precision_recall_curve` 计算不同阈值下的 F1，选取最大值对应的阈值。
- 将最优阈值应用到测试集预测。

## 📊 结果

- **TF-IDF + Logistic Regression**：验证集 F1 约 0.78 ~ 0.80
- **BERT 微调**：验证集 F1 可达 0.82 ~ 0.85（公开测试集）

## ❗ 常见问题

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| `ImportError: cannot import name 'AdamW' from 'transformers'` | 新版 transformers 移除了 AdamW | 改用 `from torch.optim import AdamW` |
| `AttributeError: 'BertTokenizer' object has no attribute 'encode_plus'` | 直接调用 tokenizer 更稳定 | 将 `tokenizer.encode_plus(...)` 改为 `tokenizer(...)` |
| `RuntimeError: grad can be implicitly created only for scalar outputs` | DataParallel 返回向量损失 | 获取 loss 后添加 `loss = loss.mean()` |
| 加载模型时键名不匹配（`Missing key(s) ... module.bert...`） | 保存与加载的模型包装不一致 | 保存时用 `model.module.state_dict()`，加载时用未包装模型 |
| 加载预训练模型时出现 `UNEXPECTED`/`MISSING` 警告 | 预训练与任务层不同 | 正常现象，可忽略 |

## 🙏 致谢

- Kaggle 提供竞赛平台与数据
- HuggingFace 提供 Transformers 库
- 数据集来源于 figure-eight 的 “Data For Everyone” 项目

---

欢迎 Star 和 Fork，如有问题请提 Issue。
