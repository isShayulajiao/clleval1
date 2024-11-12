# 2025 CCL25-clleval     面向中国文学领域的大语言模型性能评测基准

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
4. [评价标准](#3-模型评测)  
5. [模型评测](#4-模型评测)  
6. [评测赛程](#评测赛程)  
7. [结果提交](#结果提交)    
8. [奖项设置](#奖项设置)    
9. [撰写技术报告](#撰写技术报告)    
10. [报名提交方式及申请本次评测语料使用权](#报名提交方式及申请本次评测语料使用权)






---



## 1 评测背景


* 大预言模型已经成为自然语言处理领域中不可或缺的工具。它们展示了在文本生成、命名实体识别、机器翻译等任务中的多样化能力，并在许多应用场景中取得了显著进展。然而，大型语言模型在中国文学研究中的应用，远远不止于文本生成和自动翻译。例如，通过大模型进行现代文学作品的自动分类和风格分析，可以帮助研究者更好地理解作家的创作特征和文学风格。尽管大模型在中国文学研究中展现出了广阔的前景，其应用仍面临诸多挑战。

* 在处理古典汉语文本时，大模型常因语言的复杂性和多义性而遇到困难；而中国文学文本的高度专业性要求大模型能够有效理解复杂的文学语言及其概念。同时，分析现代文学批评倾向、进行现代文学批评挖掘、理解古代文学知识、预测文学作品风格、进行文学语言风格转换，以及提升文学阅读理解和文学语言理解等任务，迫切需要更多样化和丰富的中文教学语料库。

* 新一代的大语言模型，如Llama-2/3、Qwen1.5/2和InternLM-2，在自然语言理解和执行各种任务方面表现出色。然而，数据的稀缺性和标记化的挑战使得这些模型在处理复杂文学任务时仅能依靠自然语言指令完成相对简单的任务。尤其在中国文学领域，它们的应用和评估面临着巨大挑战，尤其是在多模态开发、教学数据集的多样性等方面。大部分大语言模型是为通用领域设计的，这暴露出专门针对中国文学领域的模型的短缺，这可能会限制相关研究的进一步发展。 此外，缺乏专门为中国文学研究设计的教学数据集，使得研究人员在教学调整时缺乏有效的支持；而缺乏评估基准也使得无法全面、准确地评估和比较大语言模型在中国文学任务中的表现。这对于研究人员来说，形成了一个巨大的瓶颈。

* 为了进一步提升大模型在中国文学语言解析方面的能力，并增强其在细粒度评估上的表现，我们精心构建了七个高质量的数据集。这些数据集不仅覆盖了丰富的文学文本类型，还涉及了多种语言理解任务，旨在推动大模型对中国文学语言的深度理解与精准分析。通过这些数据集的引入，我们期望能够在语言结构、文化内涵和文学风格等方面实现模型的全面提升，从而为中文自然语言处理领域的发展作出贡献。



**评测数据集** :

- [clleval (clleval_aclue)](https://huggingface.co/datasets/ChanceFocus/flare-zh-afqmc)
- [clleval_(clleval_author_2_class)](https://huggingface.co/datasets/ChanceFocus/flare-zh-stocka)
- [clleval (clleval_oa)](https://huggingface.co/datasets/ChanceFocus/flare-zh-corpus)
- [clleval (clleval_oa2)](https://huggingface.co/datasets/ChanceFocus/flare-zh-fineval)
- [clleval (clleval_cft)](https://huggingface.co/datasets/ChanceFocus/flare-zh-fe)
- [clleval (clleval_ner_re)](https://huggingface.co/datasets/ChanceFocus/flare-zh-nl)
- [clleval (clleval_tsla)](https://huggingface.co/datasets/ChanceFocus/flare-zh-nl2)


---

## 2 任务介绍

CLLEval提供的7个数据集涵盖了不同类型和风格的中国文学数据，提供了从现代文学批评到古代文学知识的多维度任务。这种多样化的数据集结构对大模型的评估带来了多重优势。不仅丰富了模型的训练和测试内容，涵盖了中国文学的特定文化背景和语言结构，使得大模型的评估结果更加精准，尤其是在语言的细粒度分析和风格迁移方面，而且通过包括文言文等语言结构复杂的文本，这些数据集能够测试大模型对古代汉语的处理能力。数据集的基本信息如下表所示：


|  Data   |               Task               | Text Types |   Raw   | Instruction | Test  |  License   | Source |
|:-------:|:--------------------------------:|:----------:|:-------:|:-----------:|:-----:|:----------:|:--------:|
|   OA1   |             现代文学批评倾向             |    现代文     |  1,014  |     829     |  829  |            |        |
|   OA2   |             现代文学批评挖掘             |    现代文     |  1,014  |     141     |  141  |            |        |
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

该数据集的内容来源于现代报刊中的文学评论文章，经过精心筛选和提取。首先，我们从多个主流报刊中挑选了涵盖不同文学流派、风格和时期的评论内容，以确保数据的多样性和代表性。所涉及的文学作品包括经典名著、当代畅销书籍以及重要文学事件的评论，广泛覆盖了各类文学作品。随后，数据采集团队对相关评论进行了手动筛选和整理，确保每条数据的准确性和相关性。所有摘录的评价都经过精确标注，并进行了必要的格式化处理，以便后续研究使用。通过这一精细的采集和整理过程，我们最终构建了一个系统且富有深度的文学评价数据集(1922-1949_Origin_AllDocs)，为文学评论分析、情感分析等研究领域提供了宝贵的资源。我们基于该数据集构建了两个文学评价数据集：现代文学批评倾向(OA1)和现代文学批评挖掘(OA2)。

### 任务内容
该数据集专注于现代文学批评中的情感分析任务，是一个判断文学评价中的情感倾向是积极、中性还是消极的过程。主要以20世纪早期的文学评论为内容，涵盖了多个文学作品的评论实例。每个实例包括一个标识符、评论文本、任务提示以及情感分类标签（如“积极”、“中性”、“消极”）。通过该数据集，研究人员可以对评论文本进行情感倾向的自动识别，推动中文文学评论的情感分析技术发展。数据集的标注确保了每条评论的情感分类准确，为中文情感分析、情感分类等领域的研究提供了丰富的素材和支持。

### 数据样例
现代文学批评倾向评测任务提供了JSON格式的数据。样例文本“站在无尽头人生沙漠的旅途上，你们要些什么，悲观的人们，正在嘶呼：无聊的人生呵”“干燥的生活呵”这种悲惨的呼声，谁都震动起来，他们要什么，是的，他们所需要的除了一点水，还有一点“爱”的安慰。
近来读了《小说月报》署名“茅盾”的《动摇》和《追求》两篇创作，在枯燥的心灵里，我得向作者致敬，虽然我和作者是一面不相识的。
《动摇》是一篇描写某地受了革命浪涛的澎湃和当地一切的凌乱，在这一篇里，作者尽力描写指示了一个人生影子，生命挣扎之中，人们用种种不同的方式，追击、渴求，这样，这样，人们渐渐躲到黑暗里去，迷梦一样的，什么也不知道。
《追求》是作者暗示了人生的憧憬，尤其是对于女性心理的状态，好像里面的女主角章秋柳——其实每一个人们都是不自觉的当着主角——和《动摇》里的女主角一样，同时使人感到十分浓厚的快感，本来是的，除了那些笨牛终日祗是躲在阴沉的苦闷里以外，我们应该极力追索我们的快感，一种满足的刺激，道德家伪善者于我们有什么需要呢？”选自本次评测任务的测试集，以下为对应的数据样例：
![现代文学批评挖掘](https://github.com/isShayulajiao/clleval1/blob/main/oa2.jpg)


### 评价标准

对于**现代文学批评倾向**的评测任务，我们采用了 **F1 Score**、**Macro F1** 和 **MCC（(Matthews Correlation Coefficient）** 来全面评估模型的表现。以下是对这些指标的详细说明，以及它们在情感倾向分析中的具体应用：

---

### **F1 值 (F1 Score)**
**F1 值** 是精确率 (Precision) 和召回率 (Recall) 的调和平均，用于综合评估模型在识别情感正例上的表现，尤其适用于类别不平衡的情感分析任务。它能够在精确率和召回率之间取得平衡，体现模型识别正面或负面情感的整体效果。

**F1 值** 计算公式如下：

\[
F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
\]

在情感倾向分析中，F1 值帮助评估模型识别特定情感（如正面或负面）的精度和完整性。较高的 F1 值表明模型不仅能够准确地识别情感类型，还能覆盖尽可能多的情感样本。

**评价标准的说明：**
- **高 F1 值** 表示模型能有效、全面地识别出评论中的情感倾向。
- **低 F1 值** 则可能表明模型在情感识别上存在遗漏或误判。

---

### **Macro F1**
**Macro F1** 是将各情感类别的 F1 值平均，用来衡量模型在不同情感类别上的一致性。宏 F1 值能有效反映多类别情感分析任务中模型的均衡表现，尤其在类别比例不均衡的情感数据中，宏 F1 值尤为重要。

**Macro F1** 计算公式为：

\[
\text{Macro F1} = \frac{1}{N} \sum_{i=1}^N F1_i
\]

在现代文学批评向任务中，Macro F1用于观察模型在各类情感（如积极、消极、中性）上的表现是否均衡。

**评价标准的说明：**
- **高宏 F1 值** 表明模型在不同情感类别上都具备稳定的识别能力。
- **低宏 F1 值** 则可能说明模型在部分情感类别上表现不佳，表现出类别间的偏差。

<hr style="border-top: 1px solid #ccc;">


### **MCC（(Matthews Correlation Coefficient）**
**MCC** 是一种全面的二分类指标，在情感分析中适合处理类别不均衡的数据，尤其适用于区分明显的正面与负面情感的任务。

**MCC** 计算公式如下：

\[
MCC = \frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}
\]

在情感倾向分析中，MCC 评估模型对各情感类别的平衡性。相比其他指标，MCC 能更有效地反映模型在复杂情感分析任务中的稳定性和整体适应性。

**评价标准的说明：**
- **高 MCC 值** 表示模型在正负情感类别之间具有良好的平衡，分类准确。
- **低 MCC 值** 则可能表明模型在某些类别上产生了过多误判，情感倾向识别不准确。

<hr style="border-top: 0.5px solid #ccc;">


上述指标共同构成了对情感倾向分析模型的综合评估框架，模型在这些指标上的优异表现能够确保其在情感分析任务中表现稳定、高效，从而能够更好地支持现代文学批评的自动化倾向分析。

---


## 2.2 现代文学批评挖掘

### 任务内容
现代文学批评挖掘是一个从从评论文本中提取出被评论的出版物名称的过程。该数据集以20世纪早期的文学评论为主要内容，包含多个评论实例，每个实例包括一个标识符、评论上下文、任务提示以及对应的出版物名称。通过该数据集，研究者可以开展出版物名称识别和文本信息抽取任务，推动中文文学作品的信息提取技术发展。数据集的标注涵盖了评论文本中涉及的作品名称，为中文文本的自动信息提取与分析提供了重要的资源。

### 数据样例
现代文学批评挖掘评测任务提供了JSON格式的数据。样例文本“《新月》的态度，素来是保持着“尊严与健康”的，他们根本看不起市场式文坛上流行的训世派·攻击派·偏激派·纤巧派·稗贩派……的，但是不幸得很，最近的《新月》，自己已犯了徐诗哲那篇《新月的态度》中的不上流的根性了，我们来看一卷十期的《新月》罢。\n……\n实秋的零星三篇，《一篇自序》是放冷箭式的挑战文，箭垜不是沈从文便是张若谷，因为沈从文是当过兵的，张若谷也从过戎的，却是他们两位都没“当过两年军官”，也从来在嘴边挂过“粗卤的骂人的话”。沈从文的《阿丽思中国游记》与张若谷的《文学生活》，的确“不是小说，也不是诗，也不是戏剧。”作者记性还不错，前者是一本《游记》后者是一种随笔，但不知实秋在“训”谁“攻”谁，像这样态度的文字，非局外人所能推测到的……\n抄二句顾仲彝在《评四本长篇小说》中的话：“其余小的枝节，我也无暇一一检举”……\n《新月》最近的态度，难于再维持“尊严与健康”了。”选自本次评测任务的测试集，以下为对应的数据样例：
![现代文学批评挖掘](https://github.com/isShayulajiao/clleval1/blob/main/oa1.png)

### 评价标准

对于现代文学批评挖掘评测任务，评价指标主要使用 **准确率 (Accuracy, ACC)**，即模型在所有测试样本中，正确预测的样本占总样本的比例。

**准确率 (ACC)** 计算公式如下：

$$
ACC = \frac{\text{正确预测的样本数}}{\text{总样本数}}
$$

在本任务中，准确率用于衡量模型在从评论文本中正确识别和提取被评论的出版物名称的能力。准确预测的标签是指模型在对给定的评论文本进行任务处理后，能准确地识别出文本中涉及的出版物名称并匹配真实标注的数量。

评价标准的说明：
* 高准确率 表示模型能够高效、精准地识别出评论中的出版物名称。
* 低准确率 可能表明模型在理解文本内容、区分相关信息或提取出版物名称方面存在困难。
因此，本次评测的主要目标是优化模型的准确性，以提高其在实际应用中的性能，尤其是在中文文学评论数据的自动化信息抽取任务中。
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




## 3 评价标准

各数据集使用的评价指标如下图所示：

|   Data   | Test  |       Metrics       |
|:--------:|:-----:|:-------------------:|
|   OA1    |  829  |         ACC         |
|   OA2    |  141  | ACC,F1,macro-F1,mcc |
|  ACLUE   | 2,000 |         MIT         |
|  cft  | 2,000 |         ACC         |
|  NER_re  | 2,750 |      entity_f1      |
| author | 2,000 | ACC,F1,macro-F1,mcc |
| tsla | 2,000 |    BERTScore-F1,BARTScore    |

各指标详细计算过程详情请看第二章[任务介绍](#2-任务介绍)

## 4 模型评测


#### 本地部署
```bash
git clone https://github.com/chancefocus/PIXIU.git --recursive
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


2. 商业API

```bash
export OPENAI_API_SECRET_KEY=YOUR_KEY_HERE
python eval.py \
    --model gpt-3.5-turbo \
    --tasks clleval_aclue,clleval_author_2_class,clleval_oa
```
## 5 结果提交

参赛队伍在测试集上参与评测，结果集使用ccleval评测框架最终生成的数据格式。

本次评测结果集采用邮件方式进行提交。参赛队伍负责人以**参赛队伍名称 + CCL2025-CLLEval 评测结果**给压缩包命名，以**队伍名称 + CCL2025-CLLEval2025结果提交**为邮件标题，将结果文件直接发送到 **wangkang1@stu.ynu.edu.cn** ，以提交时间截止前您的最后一次提交结果为准。

## 6 奖项设置

本次评测将设置一、二、三等奖，中文信息学将会为本次评测获奖队伍提供荣誉证书。







