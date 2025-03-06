# BIGCity

# Motivation：
The novelty of this paper is using a unified model to handle tasks involving both "trajectory" and "traffic state." Applications like navigation apps require both data types—users need traffic predictions and optimal paths (trajectories). Fundamentally, traffic states are derived from individual trajectories, so integrating them enhances model performance. This was often overlooked in prior research, and we fill that gap.

<div align="center"><img src=image/Article/MTSD_Model.png width=100% /> <!--<figcaption>Figure 1 : The Comparison Between BIGCity and Traditional ST Models</figcaption>--> </div>

<br>

<bf>

As shown in the below figure, BIGCity is a MTMD model. Traffic state tasks contains one-step prediction (O-Step), multi-step prediction (M-Step), and traffic state imputation (TSI); Trajectory tasks includes travel time estimation (TTE), next hop prediction (NexH), most similar trajectory search (Simi), trajectory classification (CLAS), and trajectory recovery (Reco).

# Challenges:
The main challenges lies in how to unify heterogeneous data and heterogeneous task.
1. Heterogeneous data are originally in different data format, and it is difficult to unified data representation.   
2. Heterogeneous task are in various complexity, and it is difficult to adopt the same model into different ST tasks.

# Contribution:
1. Unified spatiotemporal data representation (Section IV)
2. Task-Oriented Prompts for diverse tasks adaption  (Section V)
3. Unified training strategy (Section VI)


# Methods：

## 1. Unified spatiotemporal data representation (Section IV)

We proposed STUnit, and ST tokenizer.

Hetero data all originate from the city's road network, and each node in road network contains both static road segment information and dynamic traffic states. Therefore, we developed the **ST tokenizer** to represent the road network.

### 1) The representation of road network

As Shown in below figure, ST tokenizer incorporates a static encoder and a dynamic encoder to separately model these static and dynamic features. Additionally, we designed a fusion encoder to integrate these two representations, generating dynamic road network representation.

<div align="center"><img src=image/Article/model_architecture.png width=70% /> <!--<figcaption>Figure 2 : The Architecture of BIGCity</figcaption>--> </div>

### 2) The representation of ST data

As shown in below figure, We find that either trajectory and traffic state is essentially a sequence, sampling from the dynamic road network. 
From that view, the most difference between trajectory and traffic state lies in sampling manner
Therefore, we designed STUnit to unified both trajectory and traffic state data into sequence format.

<div align="center"><img src=image/Article/Figure2.png width=100% /> <!--<figcaption>Figure 3 : From STunit to ST feature Tokens </figcaption>--> </div>

<br>

By integrating STUnit and the ST tokenizer, both trajectory and traffic states are commonly represented as feature sequence. 
Therefore, all tasks can be trained in sequence modeling manner.
Considering the strong sequence modeling capabilities of GPT-2, we selected it as the backbone of our model.

## 2. Task-oriented Prompts (Section V)

It is difficult for a model to identify the specific task type just according to the spatiotemporal data, because various task may share the same ST input.
To address this, we introduce task-oriented prompts that serve as a task identifier. 

Based on this approach, data from multiple spatiotemporal tasks can be integrated into a single dataset for joint training.

Specifically : we first categorize ST tasks into four major types (see in paper's Table 1), with the outputs summarized into two forms: classification of static discrete road segment IDs and regression of continous dynamic features. Accordingly, we define task placeholders as [CLAS] for classification and [REG] for regression.

Then, we designed task placeholders to act as output markers, indicating the type and quantity of outputs for each task. Further, we provide individual prompt templates for each task to specify the task type. Details of these templates can be found in the section V.A. The following figures show templates in certain tasks as examples.

<div align="center"><img src=image/Article/template_1.png width=100% /></div>

<br>

<div align="center"><img src=image/Article/template_2.png width=100% /> <!--<figcaption>Figure 4 : Examples of Prompt Temples </figcaption>--> </div>

## 3. Model Training (Section VI)

BIGCity employs a two-stage training strategy:

1. Masked Reconstruction Training: In this stage, only spatiotemporal data (ST data) and task placeholders are involved. BIGCity is trained to generate general ST representations.

2. Task-Oriented Prompt Tuning: Prompt tuning: With the aid of task-oriented prompts, the model is jointly fine-tuned on omultiple tasks. After this stage, model is capable of multi-task ability.


<div align="center"><img src=image/Article/training.png width=80% /> <!--<figcaption>Figure 5 : The training of BIGCity </figcaption>--> </div>

<br>

<br>

# Data Description
We evaluated BIGCity on three real-world datasets: Beijing (BJ), Xi'an (XA), and Chengdu (CD). The BJ dataset consists of taxi trajectories collected in November 2015, and the XA and CD datasets include online car-hailing trajectories from November 2018. For experiments, XA and CD datasets were split $6:2:2$ for training, validation, and testing, while BJ was split $8:1:1$. Dataset statistics are provided in following table. 

You can run **split_data.py** in utils to process trajectory data first.

| Dataset | bj | xa | cd |
|:-------------------------:|:--------------------:|:--------------------:|:--------------------:|
| Time Span               | one month          | one month          | one month          |
| Trajectory              | 1018312            | 384618             | 559729             |
| User Number             | 1677               | 26787              | 48295              |
| Road Segments           | 40306              | 5269               | 6195               |


The Road networks for all three cities were extracted from OpenStreetMap (OSM), and trajectories were map-matched to the networks to compute traffic states. Each time slice for traffic states spans 30 minutes. Due to sparse trajectories in the BJ dataset, dynamic traffic state features from ST units (Eq.~\eqref{eq:traffic_series}) were excluded, a limitation common in trajectory datasets. 

# Evaluation Metrics
Trajectory Travel Time Estimation: we adopt three metrics, including mean absolute error (MAE), mean absolute percentage error (MAPE), and root mean square error (RMSE).

Next Hop Prediction: Following the settings in START, we used three metrics: Accuracy (ACC), mean reciprocal rank at 5 (MRR@5) and Normalized Discounted Cumulative Gain at 5 (NDCG@5).

Trajectory Classification: We use Accuracy (ACC), F1-score (F1), and Area Under ROC (AUC) to evaluate binary classification on BJ dataset. Using Micro-F1, Macro-F1 and Macro-Recall on XA and CD datasets.

Most similar trajectory search: we evaluated the model using Top-1 Hit Rate (HR@1), Top-5 Hit Rate (HR@5), and Top-10 Hit Rate (HR@10), where Top-k Hit rate indicates the probability that the ground truth is in the top-k most similar samples ranked by the model.

Metrics used in traffic state tasks are: MAE, MAPE, RMSE.

Trajectory Recovery: We evaluated our model on three types of mask ratios: $85\%$, $90\%$, $95\%$. The evaluation metric is the recovery accuracy and Macro-F1 on masked road segments.

# Training

### Requirements

The following command creates a conda environment named BIGCity. 

```shell
conda create --name bigcity python=3.12.9
conda activate bigcity
pip install -r requirements.txt
```

wandb is used to record the loss at training time, please `run wandb login` first.

All checkpoints is stored by default in the./checkpoints folder.



### stage1:  Masked Renconstruction Training

```shell
python pretrain.py \
  --task_name pretrain \
  --use_gpu \
  --root_path ../dataset/ \
  --city xa \
  --mask_rate 0.5 \
  --batch_size 32 \
  --learning_rate 2e-4 \
  --train_epochs 20 \
```



### stage2:  Task Oriented Prompt Tuning

```shell
python finetune.py \
  --use_gpu \
  --root_path ../dataset/ \
  --city xa \
  --batch_size 32 \
  --learning_rate 2e-4 \
  --train_epochs 20 \
```


# Model Weights and Datasets

**Model Weights** : https://huggingface.co/XieYuBUAA/BIGCity/tree/main

**Datasets** : https://huggingface.co/datasets/XieYuBUAA/BIGCity_Dataset/tree/main

**Training Logs** : https://wandb.ai/iurww/bigcity/workspace?nw=nwuseriurww


# Standard Deviation:

We listed the error bar of BIGCity on each metrics of each task among all three city dataset. The details are presented in the following table:

Performance on the trajectory-based non-generative tasks:

|            | BJ         | XA          | CD         |
| ---------- | ---------- | ----------- | ---------- |
| TTE_MAE    | 8.87±0.02  | 1.72±7E-3   | 1.29±5E-3  |
| TTE_RMSE   | 33.21±0.31 | 2.61±0.01   | 2.18±3E-3  |
| TTE_MAPE   | 30.34±0.15 | 29.76 ±0.28 | 28.59±0.17 |
| CLAS_ACC   | 0.872±8E-4 | 0.110±9E-4  | 0.153±8E-4 |
| CLAS_F1    | 0.891±6E-4 | 0.104±3E-4  | 0.169±9E-4 |
| CLAS_AUC   | 0.909±7E-4 | 0.113±2E-4  | 0.162±8E-4 |
| Next_ACC   | 0.751±1E-3 | 0.837±3E-3  | 0.821±2E-3 |
| Next_MRR@5 | 0.855±2E-3 | 0.923±3E-3  | 0.910±1E-3 |
| Next_NDC@5 | 0.902±2E-3 | 0.940±5E-3  | 0.938±2E-3 |
| Sim_HR@1   | 0.801±7E-4 | 0.791±3E-4  | 0.646±4E-4 |
| Sim_HR@5   | 0.895±3E-4 | 0.887±4E-4  | 0.787±9E-5 |
| Sim_HR@10  | 0.952±4E-4 | 0.909 ±5E-4 | 0.821±2E-4 |


Performance on Trajectory Recovery ( Accuracy ) : 

| 85%        | 90%        | 95%        | 85%        | 90%        | 95%        | 85%        | 90%        | 95%        | 
| ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | 
| 0.52±8E-3 | 0.47±4E-3 | 0.37±5E-3 | 0.56±1E-3 | 0.49±9E-4 | 0.38±2E-3 | 0.56±7E-4 | 0.51±7E-3 | 0.41±6E-3 |


Performance on Trajectory Recovery ( Macro-F1 ) : 

| 85%        | 90%        | 95%        | 85%        | 90%        | 95%        | 85%        | 90%        | 95%        |
| ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | 
|0.259±2E-4 | 0.217±1E-4 | 0.177±2E-4 | 0.309±2E-4 | 0.258±3E-4 | 0.194±4E-4 | 0.321±6E-4 | 0.269±5E-4 | 0.212±5E-4 |

The Performance in One-Step Traffic State Prediction:

| MAE       | MAPE      |    RMSE   |    MAE    |    MAPE    | RMSE      | 
| --------- | --------- | --------- | --------- | ---------- | --------- | 
| 0.79±2E-3 | 9.73±3E-2 | 1.74±1E-3 | 1.12±2E-3 | 11.16±6E-1 | 2.10±4E-3 | 


The Performance in Multi-Step Traffic State Prediction:

| MAE       | MAPE       |  RMSE     |   MAE     |   MAPE     |    RMSE   |
| --------- | ---------- | --------- | --------- | ---------- | --------- |
| 1.16±3E-3 | 14.01±4E-2 | 2.14±4E-3 | 1.41±2E-3 | 15.98±2E-1 | 2.47±3E-3 |



The Performance in Multi-Step Traffic State Imputation:

| MAE       | MAPE       |  RMSE     |   MAE     |   MAPE     |    RMSE   |
| --------- | ---------- | --------- | --------- | ---------- | --------- |
| 0.536±2E-3 | 6.671±5E-2 | 1.335±2E-3 | 0.665±1E-3 | 8.192±3E-1 | 1.617±8E-4 |
