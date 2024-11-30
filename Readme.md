# BIGCity

# Motivation：
The novelty of this paper is using a unified model to handle tasks involving both "trajectory" and "traffic state." Applications like navigation apps require both data types—users need traffic predictions and optimal paths (trajectories). Fundamentally, traffic states are derived from individual trajectories, so integrating them enhances model performance. This was often overlooked in prior research, and we fill that gap.

<div align="center"><img src=image/Article/MTSD_Model.png width=100% /> <figcaption>Figure 1 : The Comparison Between BIGCity and Traditional ST Models</figcaption> </div>

<br>

<bf>

As shown in the Figure 1, BIGCity is a MTMD model. Traffic state tasks contains one-step prediction (O-Step), multi-step prediction (M-Step), and traffic state imputation (TSI); Trajectory tasks includes travel time estimation (TTE), next hop prediction (NexH), most similar trajectory search (Simi), trajectory classification (CLAS), and trajectory recovery (Reco).

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

As Shown in Figure 2, ST tokenizer incorporates a static encoder and a dynamic encoder to separately model these static and dynamic features. Additionally, we designed a fusion encoder to integrate these two representations, generating dynamic road network representation.

<div align="center"><img src=image/Article/model_architecture.png width=70% /> <figcaption>Figure 2 : The Architecture of BIGCity</figcaption> </div>

### 2) The representation of ST data

As shown in Figure 3, We find that either trajectory and traffic state is essentially a sequence, sampling from the dynamic road network. 
From that view, the most difference between trajectory and traffic state lies in sampling manner
Therefore, we designed STUnit to unified both trajectory and traffic state data into sequence format.

<div align="center"><img src=image/Article/Figure2.png width=70% /> <figcaption>Figure 3 : From STunit to ST feature Tokens </figcaption> </div>

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

<div align="center"><img src=image/Article/template_1.png width=90% /></div>

<br>

<div align="center"><img src=image/Article/template_2.png width=90% /> <figcaption>Figure 4 : Examples of Prompt Temples </figcaption> </div>

## 3. Model Training (Section VI)

BIGCity employs a two-stage training strategy:

1. Masked Reconstruction Training: In this stage, only spatiotemporal data (ST data) and task placeholders are involved. BIGCity is trained to generate general ST representations.

2. Task-Oriented Prompt Tuning: Prompt tuning: With the aid of task-oriented prompts, the model is jointly fine-tuned on omultiple tasks. After this stage, model is capable of multi-task ability.


<div align="center"><img src=image/Article/training.png width=90% /> <figcaption>Figure 5 : The training of BIGCity </figcaption> </div>


<br>

<br>

# Training

### Requirements

The following command creates a conda environment named BIGCity

```shell
conda env create -f environment.yml
```



### stage1:  Mased Renconstruction Training

```shell
model_name=GPT4TS
python run_pretrain.py \
  --task_name pretrain \
  --is_training 1 \
  --root_path ../dataset/ \
  --city xa \
  --freq s\
  --mask_rate 0.7 \
  --model_id 1 \
  --model GPT4TS \
  --data Traj \
  --devices '1' \
  --gpt_layers 12 \
  --batch_size  128\
  --d_model 768 \
  --des 'Exp' \
  --itr 1 \
  --mlp 1 \
  --learning_rate 2e-4 \
  --train_epochs 200
```



### stage2:  Task Oriented Prompt Tuning

```shell
model_name=GPT4Finetune
python -u run_traj.py \
  --task_name task_tuning \
  --is_training 1 \
  --root_path ../dataset/ \
  --city xa \
  --freq s\
  --model $model_name \
  --data Task_Tuning \
  --use_multi_gpu \
  --devices '1' \
  --gpt_layers 12 \
  --batch_size  128\
  --d_model 768 \
  --des 'Exp' \
  --itr 1 \
  --mlp 1 \
  --learning_rate 1e-5 \
  --train_epochs 30 \
```



### Description of command line parameters

- `--task_name`: Task name, options include: long-term forecast, short-term forecast, imputation, classification, anomaly detection, etc.
- `--is_training`: Whether in training state (1 means yes, 0 means no).
- `--model_id`: Model identifier, used to distinguish different model instances.
- `--model`: The model name, e.g., Autoformer, Transformer, TimesNet, etc.
- `--data`: Dataset type.
- `--root_path`: Root directory of the data files.
- `--city`: The city corresponding to the dataset.
- `--embedding_model`: The name of the road embedding model.
- `--freq`: Frequency for time feature encoding, e.g., second, minute, hour, etc.
- `--checkpoints`: Path to save model checkpoints.
- `--mask_rate`: Mask ratio, i.e., the proportion of the input sequence randomly masked.
- `--num_kernels`: Number of kernels in the Inception module.
- `--d_model`: Model dimension size.
- `--n_heads`: Number of attention heads in multi-head attention.
- `--e_layers`: Number of encoder layers.
- `--d_layers`: Number of decoder layers.
- `--d_ff`: Dimension of the feed-forward network layer.
- `--moving_avg`: Size of the moving average window.
- `--factor`: Attention factor.
- `--distil`: Whether to use distillation in the encoder, default is `True`.
- `--dropout`: Dropout rate.
- `--embed`: Time feature encoding method, options include: `timeF`, `fixed`, `learned`.
- `--activation`: Activation function type.
- `--output_attention`: Whether to output attention weights from the encoder.
- `--num_workers`: Number of data loader workers.
- `--itr`: Number of experimental repetitions.
- `--train_epochs`: Number of training epochs.
- `--batch_size`: Batch size.
- `--patience`: Early stopping patience.
- `--learning_rate`: Learning rate.
- `--des`: Experiment description.
- `--loss`: Loss function type.
- `--lradj`: Learning rate adjustment strategy.
- `--use_amp`: Whether to use automatic mixed precision training.
- `--use_gpu`: Whether to use GPU.
- `--gpu`: GPU ID.
- `--use_multi_gpu`: Whether to use multiple GPUs.
- `--devices`: List of device IDs when using multiple GPUs.
- `--p_hidden_dims`: Hidden layer dimensions of the projector (list).
- `--p_hidden_layers`: Number of hidden layers in the projector.
- `--gpt_layers`: Number of layers in the GPT model.
- `--ln`: Whether to use layer normalization.
- `--mlp`: Whether to use MLP (Multi-Layer Perceptron).
- `--weight`: Weight coefficient.
- `--percent`: Percentage.
- `--loss_alpha, --loss_beta, --loss_gamma`: Weight distribution for multi-loss terms.
- `--checkpoint_name`: Name of a specific checkpoint.
- `--gpt2_checkpoint_name`: GPT-2 model checkpoint name.
- `--sample_rate`: Sampling rate.










# standard deviation:

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

# BIGCity
