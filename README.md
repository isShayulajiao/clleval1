# 2025 CCL25-CLLEval  
# 面向中国文学领域的大语言模型性能评测基准

<div>
<div align="left">
    <a target='_blank'>Kang Wang<sup>1</sup></span>&emsp;
    <a target='_blank'>Gang Hu<sup>1</sup></span>&emsp;
    <a target='_blank'>QingQing Wang<sup>1</sup></a>&emsp;
    <a target='_blank'>Ke Qin<sup>1</sup></a>&emsp;
        
</div>
<div>
<div align="left">
    <sup>1</sup>Yunan University&emsp;
    <sup>3</sup>Sichuan University&emsp;
    <sup>3</sup>Wuhan University&emsp;

</div>
<div align="left">
    <img src='https://i.postimg.cc/DfB8jxV1/ynu.png' alt='Yunnan University Logo' height='80px'>&emsp;
    <img src='https://i.postimg.cc/CLtkBwz7/57-EDDD9-FB0-DF712-F3-AB627163-C2-1-EF15655-13-FCA.png' alt='Wuhan University Logo' height='80px'>&emsp;
    <img src='https://i.postimg.cc/NjKhDkGY/DFAF986-CCD6529-E52-D7830-F180-D-C37-C7-DEE-4340.png' alt='Sichuan University Logo' height='80px'>&emsp;
    
</div>


## 评测组织者
* 胡刚（hugang@ynu.edu.cn），云南大学  


## 联系人及联系方式
* 王康(wangkang1@stu.ynu.edu.cn)，云南大学硕士研究生

## 目录

1. [评测背景](#1-评测背景)  
2. [任务介绍](#2-任务介绍)  
    2.1 [现代文学批评倾向](#21-现代文学批评倾向)  
    2.2 [现代文学批评挖掘](#22-现代文学批评挖掘)  
    2.3 [古代文学知识理解](#23-古代文学知识理解)  
    2.4 [文学阅读理解](#24-文学阅读理解)  
    2.5 [文学语言理解](#25-文学语言理解)  
    2.6 [文学作品风格预测](#26-文学作品风格预测)  
    2.7 [文学语言风格转换](#27-文学语言风格转换)
3. [评价标准](#3-模型评测)  
4. [模型评测](#4-模型评测)  
5. [评测赛程](#评测赛程)  
6. [结果提交](#结果提交)      
7. [报名提交方式及申请本次评测语料使用权](#报名提交方式及申请本次评测语料使用权)
8. [奖项设置](#奖项设置)   


---



## 1 评测背景


* 大预言模型已经成为自然语言处理领域中不可或缺的工具。它们展示了在文本生成、命名实体识别、机器翻译等任务中的多样化能力，并在许多应用场景中取得了显著进展。然而，大型语言模型在中国文学研究中的应用，远远不止于文本生成和自动翻译。例如，通过大模型进行现代文学作品的自动分类和风格分析，可以帮助研究者更好地理解作家的创作特征和文学风格。尽管大模型在中国文学研究中展现出了广阔的前景，其应用仍面临诸多挑战。

* 在处理古典汉语文本时，大模型常因语言的复杂性和多义性而遇到困难；而中国文学文本的高度专业性要求大模型能够有效理解复杂的文学语言及其概念。同时，分析现代文学批评倾向、进行现代文学批评挖掘、理解古代文学知识、预测文学作品风格、进行文学语言风格转换，以及提升文学阅读理解和文学语言理解等任务，迫切需要更多样化和丰富的中文教学语料库。

* 新一代的大语言模型，如Llama-2/3、Qwen1.5/2和InternLM-2，在自然语言理解和执行各种任务方面表现出色。然而，数据的稀缺性和标记化的挑战使得这些模型在处理复杂文学任务时仅能依靠自然语言指令完成相对简单的任务。尤其在中国文学领域，它们的应用和评估面临着巨大挑战，尤其是在多模态开发、教学数据集的多样性等方面。大部分大语言模型是为通用领域设计的，这暴露出专门针对中国文学领域的模型的短缺，这可能会限制相关研究的进一步发展。 此外，缺乏专门为中国文学研究设计的教学数据集，使得研究人员在教学调整时缺乏有效的支持；而缺乏评估基准也使得无法全面、准确地评估和比较大语言模型在中国文学任务中的表现。这对于研究人员来说，形成了一个巨大的瓶颈。

* 为了进一步提升大模型在中国文学语言解析方面的能力，并增强其在细粒度评估上的表现，我们精心构建了七个高质量的数据集。这些数据集不仅覆盖了丰富的文学文本类型，还涉及了多种语言理解任务，旨在推动大模型对中国文学语言的深度理解与精准分析。通过这些数据集的引入，我们期望能够在语言结构、文化内涵和文学风格等方面实现模型的全面提升，从而为中文自然语言处理领域的发展作出贡献。



**评测数据集** :

- [CLLEval (CLLEval_aclue)](https://huggingface.co/datasets/ChanceFocus/flare-zh-afqmc)
- [CLLEval_(CLLEval_author_2_class)](https://huggingface.co/datasets/ChanceFocus/flare-zh-stocka)
- [CLLEval (CLLEval_oa)](https://huggingface.co/datasets/ChanceFocus/flare-zh-corpus)
- [CLLEval (CLLEval_oa2)](https://huggingface.co/datasets/ChanceFocus/flare-zh-fineval)
- [CLLEval (CLLEval_cft)](https://huggingface.co/datasets/ChanceFocus/flare-zh-fe)
- [CLLEval (CLLEval_ner_re)](https://huggingface.co/datasets/ChanceFocus/flare-zh-nl)
- [CLLEval (CLLEval_tsla)](https://huggingface.co/datasets/ChanceFocus/flare-zh-nl2)


---

## 2 任务介绍

CLLEval提供的7个数据集涵盖了不同类型和风格的中国文学数据，提供了从现代文学批评到古代文学知识的多维度任务。这种多样化的数据集结构对大模型的评估带来了多重优势。不仅丰富了模型的训练和测试内容，涵盖了中国文学的特定文化背景和语言结构，使得大模型的评估结果更加精准，尤其是在语言的细粒度分析和风格迁移方面，而且通过包括文言文等语言结构复杂的文本，这些数据集能够测试大模型对古代汉语的处理能力。数据集的基本信息如下表所示：


|  Data   |               Task               | Text Types |   Raw   | Instruction | Test  |  License   | Source |
|:-------:|:--------------------------------:|:----------:|:-------:|:-----------:|:-----:|:----------:|:--------:|
|   OA2   |             现代文学批评倾向             |    现代文     |  1,014  |     141    |  141  |            |        |
|   OA1   |             现代文学批评挖掘             |    现代文     |  1,014  |     829     |  829  |            |        |
|  ACLUE  |             古代文学知识理解             |    文言文     | 49,660  |   49,660    | 2,000 |    MIT     | [1]    |
|   cft   |              文学阅读理解              |    现代文     | 29,013  |   29,013    | 2,000 | CC-BY-SA-4.0 | [2]    |
| NER_re  |              文学语言理解              |    现代文     | 28,894  |   27,864    | 2,750 |   Public   | [3]    |
| author  |             文学作品风格预测             |    现代文     | 30,324  |   30,324    | 2,000 |   Public   | [4]    |
|  tsla   |             文学语言风格转换             |    文言文     | 972,467 |   972,467   | 2,000 |   MIT   | [5]    |



1. https://github.com/isen-zhang/ACLUE
2. Cui, Y., Liu, T., Chen, Z., Wang, S., & Hu, G. (2016). Consensus attention-based neural networks for Chinese reading comprehension. arXiv preprint arXiv:1607.02250.
3. Xu, J., Wen, J., Sun, X., & Su, Q. (2017). A discourse-level named entity recognition and relation extraction dataset for chinese literature text. arXiv preprint arXiv:1711.07010.
4. https://blog.csdn.net/zcp0216/article/details/122063405?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522170159041816800192270328%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=170159041816800192270328&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-2-122063405-null-null.nonecase&utm_term=%E4%B8%AD%E6%96%87%E6%96%87%E5%AD%A6%E6%95%B0%E6%8D%AE%E9%9B%86&spm=1018.2226.3001.4450
5. https://github.com/NiuTrans/Classical-Modern

### 2.1 现代文学批评倾向

该数据集的内容来源于现代报刊中的文学评论文章，人工经过精心筛选和提取。首先，我们从多个主流报刊中挑选了涵盖不同文学流派、风格和时期的评论内容，以确保数据的多样性和代表性。所涉及的文学作品包括经典名著、当代畅销书籍以及重要文学事件的评论，广泛覆盖了各类文学作品。随后，数据采集团队对相关评论进行了手动筛选和整理，确保每条数据的准确性和相关性。所有摘录的评价都经过精确标注，并进行了必要的格式化处理，以便后续研究使用。通过这一精细的采集和整理过程，我们最终构建了一个系统且富有深度的文学评价数据集(1922-1949_Origin_AllDocs)，为文学评论分析、情感分析等研究领域提供了宝贵的资源。我们基于该数据集构建了两个文学评价数据集：现代文学批评倾向(OA1)和现代文学批评挖掘(OA2)。
现代文学批评倾向和现代文学批评挖掘CLLEval只提供了测试集作为此次任务集的域外数据集，用来评估模型在新任务和领域中的准确性和鲁棒性以及模型的泛化能力。

### 任务内容
该数据集专注于现代文学批评中的情感分析任务，是一个判断文学评价中的情感倾向是积极、中性还是消极的过程。主要以20世纪早期的文学评论为内容，涵盖了多个文学作品的评论实例。每个实例包括一个标识符、评论文本、任务提示以及情感分类标签（如“积极”、“中性”、“消极”）。通过该数据集，研究人员可以对文学批评文本进行情感倾向的自动识别，推动中文文学评论的情感分析技术发展。数据集的标注确保了每条评论的情感分类准确，为文学批评情感分析、情感分类等领域的研究提供了丰富的素材和支持。

### 数据样例
现代文学批评倾向评测任务提供了JSON格式的数据。以下为相应的数据样例：
![现代文学批评挖掘](https://github.com/isShayulajiao/CCL25-CLLEval/blob/main/oa2.jpg)


### 评价指标

对于**现代文学批评倾向**的评测任务，我们采用了 **准确率 (ACC)**、**F1 Score**、**Macro F1** 和 **MCC（(Matthews Correlation Coefficient）** 来全面评估模型的表现。以下是对这些指标的详细说明，以及它们在情感倾向分析中的具体应用：

**准确率 (ACC)** 通过计算预测正确的样本占总样本的比率来测量模型的文本分类效果
计算公式如下：

$$
ACC = \frac{\text{正确预测的样本数}}{\text{总样本数}}
$$

评价指标的说明：
**高ACC值** 表示模型能够精准地判断文学批评中的情感倾向。
**低ACC值** 可能体现模型在理解文本内容或提取出文学批评的情感倾向方面存在不足。

### **F1 值 (F1 Score)**
**F1 值** 是精确率 (Precision) 和召回率 (Recall) 的调和平均，用于综合评估模型在识别情感正例上的表现，尤其适用于类别不平衡的情感分析任务。它能够在精确率和召回率之间取得平衡，体现模型识别正面或负面情感的整体效果。

**F1 值** 计算公式如下：

$$
F1 = 2 \times \frac{\text{准确率} \times \text{召回率}}{\text{准确率} + \text{召回率}}
$$

在情感倾向分析中，F1 值帮助评估模型识别特定情感（如正面或负面）的精度和完整性。较高的 F1 值表明模型不仅能够准确地识别情感类型，还能覆盖尽可能多的情感样本。

**评价指标的说明：**
- **高 F1 值** 表示模型能有效、全面地识别出评论中的情感倾向。
- **低 F1 值** 则可能表明模型在情感识别上存在遗漏或误判。

### **Macro F1**
**Macro F1** 是将各情感类别的 F1 值平均，用来衡量模型在不同情感类别上的一致性。宏 F1 值能有效反映多类别情感分析任务中模型的均衡表现，尤其在类别比例不均衡的情感数据中，Macro F1尤为重要。

**Macro F1** 计算公式为：

$$
\text{Macro F1} = \frac{1}{N} \sum_{i=1}^N F1_i
$$

在现代文学批评向任务中，Macro F1用于观察模型在各类情感（如积极、消极、中性）上的表现是否均衡。

**评价指标的说明：**
- **高Macro F1值** 表明模型在不同情感类别上都具备稳定的识别能力。
- **低Macro F1值** 则可能说明模型在部分情感类别上表现不佳，表现出类别间的偏差。


### **MCC（Matthews Correlation Coefficient，马修斯相关系数）**
**MCC** 是一种全面的二分类指标，在情感分析中适合处理类别不均衡的数据，尤其适用于区分明显的正面与负面情感的任务。

**MCC** 计算公式如下：

$$
\text{MCC} = \frac{\sum_{k=1}^{K} \sum_{l=1}^{K} \sum_{m=1}^{K} C_{kk} \cdot C_{lm} - C_{kl} \cdot C_{mk}}{\sqrt{\left(\sum_{k=1}^{K} \sum_{l=1}^{K} C_{kl}\right) \cdot \left(\sum_{k=1}^{K} \sum_{l=1}^{K} C_{lk}\right) \cdot \left(\sum_{k=1}^{K} \sum_{l=1}^{K} C_{kl}\right) \cdot \left(\sum_{k=1}^{K} \sum_{l=1}^{K} C_{lk}\right)}}
$$

**公式符号解释**

- **\( K \)**：类别数（例如，二分类时 \( K = 2 \)，三分类时 \( K = 3 \) 等）。
- **\( C_{kk} \)**：混淆矩阵的对角线元素，表示模型正确分类为第 \( k \) 类的样本数（即真阳性数）。
- **\( C_{kl} \)**：第 \( k \) 类被误分类为第 \( l \) 类的样本数（即第 \( k \) 类的假阳性数）。
- **\( C_{lm} \)** 和 **\( C_{mk} \)**：分别表示第 \( l \) 类和第 \( m \) 类之间的交叉误分类数，用于多分类任务的误分类计数。

在情感倾向分析中，MCC 评估模型对各情感类别的平衡性。相比其他指标，MCC 能更有效地反映模型在复杂情感分析任务中的稳定性和整体适应性。

**评价指标的说明：**
- **高 MCC 值** 表示模型在正负情感类别之间具有良好的平衡，分类准确。
- **低 MCC 值** 则可能表明模型在某些类别上产生了过多误判，情感倾向识别不准确。

---


## 2.2 现代文学批评挖掘

### 任务内容
现代文学批评挖掘是一个从评论文本中提取出被评论的出版物名称的过程。该数据集同样来自于文学评价数据集(1922-1949_Origin_AllDocs)，以20世纪早期的文学评论为主要内容，包含多个评论实例，每个实例包括一个标识符、评论上下文、任务提示以及对应的出版物名称。通过该数据集，研究者可以开展出版物名称识别和文本信息抽取任务。数据集的标注涵盖了评论文本中涉及的作品名称，为中文文本的自动信息提取与分析提供了重要的资源。

### 数据样例
现代文学批评挖掘评测任务提供了JSON格式的数据。以下为相应的数据样例：
![现代文学批评挖掘](https://github.com/isShayulajiao/CCL25-CLLEval/blob/main/oa1.jpg)

### 评价指标

在本任务中，准确率用于衡量模型在从评论文本中正确识别和提取被评论的出版物名称的能力。准确预测的标签是指模型在对给定的评论文本进行任务处理后，能准确地识别出文本中涉及的出版物名称并匹配真实标注的数量。

评价指标的说明：
* 高准确率 表示模型能够高效、精准地识别出评论中的出版物名称。
* 低准确率 可能表明模型在理解文本内容、区分相关信息或提取出版物名称方面存在困难。

---


### 2.3 古代文学知识理解
### 任务内容
### 评价标准
### 2.4 文学阅读理解
### 任务内容
### 评价标准
### 2.5 文学语言理解
### 任务内容
### 评价标准
### 2.6 文学作品风格预测
### 任务内容
### 评价标准
### 2.7 文学语言风格转换
### 任务内容
### 评价标准

---


## 3 评价标准

各数据集使用的评价指标如下图所示：

| Task                      | Data     | Test  | Metrics                          |
|:---------------------------:|:--------:|:-----:|:---------------------------------:|
| 现代文学批评倾向            | OA2      | 829   | ACC, F1, macro-F1, mcc                                |
| 现代文学批评挖掘            | OA      | 141   | ACC           |
| 古代文学知识理解            | ACLUE    | 2,000 | ACC, F1, macro-F1, mcc           |
| 文学阅读理解                | cft      | 2,000 | ACC                               |
| 文学语言理解                | NER_re   | 2,750 | entity_f1                         |
| 文学作品风格预测            | author   | 2,000 | ACC, F1, macro-F1, mcc           |
| 文学语言风格转换            | tsla     | 2,000 | BERTScore-F1, BARTScore          |


各指标详细计算过程详情请看[**第二章任务介绍**](#2-任务介绍)

## 4 模型评测


#### 本地部署
```bash
git clone https://github.com/isShayulajiao/CLLEval1.git
cd CLLLM
pip install -r requirements.txt
cd PIXIU/src/literature-evaluation
pip install -e .[multilingual]
```


#### 自动化任务评测

在评估之前，请下载[BART模型检查点](https://drive.google.com/u/0/uc?id=1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m&export=download)到'src/metrics/BARTScore/bart_score.pth'。

对于自动化评测，请按照以下步骤操作：

1. Huggingface Transformer

   若要评估托管在 HuggingFace Hub 上的模型（例如，Baichuan2-7B-Base），请使用以下命令：

```bash
python src/eval.py \
    --model hf-causal-vllm \
    --tasks flare_oa2 \
    --model_args use_accelerate=True,pretrained=Qwen/Qwen2-7B,tokenizer=Qwen/Qwen2-7B,max_gen_toks=1024,use_fast=False,dtype=float16,trust_remote_code=True 
```

各模型使用的model参数如下表所示：


| Model                   | model 参数       |
|:-------------------------:|:------------------:|
| bloomz_7b1              | hf-causal-vllm   |
| Baichuan-7B             | hf-causal-llama  |
| Llama-2-7b-hf           | hf-causal-llama  |
| Baichuan2-7B-Base       | hf-causal-vllm   |
| Qwen-7B                 | hf-causal-vllm   |
| Xunzi-Qwen1.5-7B        | hf-causal-vllm   |
| Qwen1.5-7B              | hf-causal-vllm   |
| internlm2-7b            | hf-causal-vllm   |
| Llama-3-8B              | hf-causal-vllm   |
| Qwen2-7B                | hf-causal-vllm   |


2. API

```bash
export OPENAI_API_SECRET_KEY=YOUR_KEY_HERE
python eval.py \
    --model gpt-3.5-turbo \
    --tasks CLLEval_aclue,CLLEval_author_2_class,CLLEval_oa
```

---


## 5 评测赛程

## 5.1 赛程

&emsp;&emsp;2024年4月10日：报名截止；

&emsp;&emsp;2024年4月11日：通过邮件的方式将测试集和结果提交链接发送给报名队伍，提交结果榜单详见各赛道Github主页；

&emsp;&emsp;2024年5月15日：测试集结果提交截止；

&emsp;&emsp;2024年5月31日：公布参赛队伍的成绩和排名（并非最终排名，还需参考技术报告）；

&emsp;&emsp;2024年6月：提交技术报告；

&emsp;&emsp;2024年7月：评测论文审稿&评测论文录用通知&评测论文Camera-Ready版提交；

&emsp;&emsp;2024年8月：CCL 2024评测研讨会；

## 5.2 报告格式

- 报告可由中文或英文撰写。
- 报告统一使用CCL 2025 的[论文模版](http://cips-cl.org/static/CCL2024/downloads/ccl2023_template.zip)。
- 报告应至少包含以下四个部分：模型介绍、评测结果、结果分析与讨论和参考文献。

---


## 6 结果提交

参赛队伍在测试集上参与评测，结果集使用ccleval评测框架最终生成的数据格式。

提交的压缩包命名为**参赛队伍名称 + CCL2025-CLLEval 评测结果**，其中包含七个任务的预测文件以及两个excel指标文件。
result.zip  
OA1.json  
OA2.json  
ACLUE.json  
cft.json  
NER_re.json  
author.json  
tsla.json  
metrics.xlsx  
metrics_Avg.xlsx  
1. 七个任务的预测文件需严格命名为OA1.json、OA2.json、ACLUE.json、cft.json、NER_re.json、author.json和tsla.json。  
2. 请严格使用**参赛队伍名称 + CCL2025-CLLEval 评测结果**对OA1.json、OA2.json ACLUE.json、cft.json、NER_re.json、author.json、tsla.json、metrics.xlsx和metrics_Avg.xlsx进行压缩，即要求解压后的文件不能存在中间目录。 选⼿可以只提交部分任务的结果，如只提交“现代文学批评倾向 ”任务，未预测任务的分数默认为0。  
3. metrics.xlsx和metrics_Avg.xlsx文件填写示例，请在[这里](https://drive.google.com/u/0/uc?id=1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m&export=download)下载
本次评测结果集采用邮件方式进行提交。参赛队伍负责人以  
**队伍名称 + CCL2025-CLLEval2025结果提交**为邮件标题，将结果文件直接发送到 **wangkang1@stu.ynu.edu.cn** ，以提交时间截止前您的最后一次提交结果为准。

### 排名

1. 所有评测任务均采用百分制分数显示，小数点后保留2位。
2. 排名取各项任务得分的平均值，即： 

$$\text{平均得分} = \frac{\sum_{i=1}^{n} \text{任务得分}_i}{n}$$

其中，**任务得分** 的取值为该任务所有评估指标的平均值。具体地，对于每个任务：  

$$\text{任务得分} = \frac{\sum_{j=1}^{m} \text{指标得分}_j}{m}$$

其中：
- n 表示任务总数。
- m 表示该任务的评估指标总数。
- $\text{指标得分}_j$表示该任务的第 j 个指标得分。

---


## 7 报名提交方式及申请本次评测语料使用权


---


## 8 奖项设置

本次评测将设置一、二、三等奖，中文信息学将会为本次评测获奖队伍提供荣誉证书。







