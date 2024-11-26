# Discussion

We made in-depth analysis on the reason why BIGST simultaneously outperforms SOTA on various tasks.

First, we incrementally add downstream task types to the former two steps. The correspondence between task types and specific tasks is detailed in the table below:

|Task Type|Task Name|
|:--:| :--: |
|Classification | Next Hop Prediction (Next)  |
|Regression (on Traffic State)| Multi-Step Prediction (Multi-Step)  |
|Regression (on Time)|  Travel Time Estimation (TTE) |

We find that the performance of the model increases with the variety of tasks involved in training.

|Task Type|ACC $\uparrow$|MAE $\downarrow$|MAPE$\downarrow$|
|:--:| :--: |:--:|:--:|
|Next| 0.79 | — | — |
|TTE  | — |1.80|—|
|Multi-Step| — | — | 14.36 |
|Multi-Step+Next| 0.82 | — | 14.19 |
|Next + TTE| 0.80 | 1.77| — |
|Multi-Step + Next + TTE| 0.84 | 1.75 | 14.14|

Experimental results indicate that: the greater the differences in task types and data characteristics (multi-step prediction and next hop prediction), the more significant the model's gains from joint multitask training.

Thus, we hypothesize that the multitask model enables information exchange between tasks, leading to higher-quality representations. To validate this hypothesis, we conducted case studies based on multi-step and next hop prediction tasks. The next hop prediction task focuses on trajectory data, and the multi-step prediction focuses on traffic state data.

Specifically, we define multitask settings as train multi-step and next hop prediction tasks toghther, and define singltask settings as only train next hop prediction task.

## Case 1 : Multitask training benefits in providing more comprehensive spatiotemporal representations

Jointly training the trajectory and traffic state tasks introduces dynamic information to the original static representation of road segments.

![alt text](image/Experiments/exp_figure1.png)

Specifically, we consider three road segments, labeled A, B, and C in Figure 1(a). Using October 11th as the observation period and setting a 30-minute time window for each time slice, we apply t-SNE to visualize the learned segment representations (Figure 1(b)). Each point in the visualization corresponds to a segment's representation within a specific time slice. The results demonstrate that the learned representations evolve over time.

Further, we introduce SimRatio(A, B) to measure the cosine similarity between A and B and define TransProb(B, A) as the transition frequency from B to A (From the statistics of the trajectory data). As shown in Figure 1(c), SimRatio changes with TransProb. However, their similarity will remain constant when considered as static representations. This indicates that multitask training aligns the ST representations more consistent with ST dynamics.

## Case 2 : Multitask Training benefits Improving representational discriminability

![alt text](image/Experiments/exp_figure2.png)

As illustrated in Figure 2(a), we take three trajectories, R1, R2, and R3. They have the same start point and destination, and all traverse from Road 131 to Road 2768. The time setting is the same as case 1. Trajectory representations in multitask and singletask are respectively visualized using t-SNE in Figures 2(b) and Figure 2(c). 

In multitask settings, each trajectory contains road segement features, temporal features and traffic state features. In single task settings, each trajectory only contains road segment features and traffic state features. Experimental results reveal that the trajectory representations obtained through multitask training exhibit better discriminability.

# t-test:

We listed the error bar of BIGST on each metrics of each task among all three city dataset. The details are presented in the following table:

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

Performance in Trajectory Generation:

| BJ_Dist   | BJ_LocF   | BJ_HDF    | BJ_EDR    | XA_Dist   | XA_LocF   | XA_HDF    | XA_EDR    | CD_Dist   | CD_LocF   | CD_HDF    | CD_EDR    |
| --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| .052±9E-3 | .038±3E-3 | 1.33±3E-2 | 0.34±4E-3 | .015±4E-3 | .015±3E-3 | 0.19±6E-3 | 0.12±7E-3 | .018±8E-3 | .020±7E-3 | 0.13±2E-2 | 0.14±2E-2 |

Performance on Trajectory Recovery ( Accuracy ) : 

| 85%        | 90%        | 95%        | 85%        | 90%        | 95%        | 85%        | 90%        | 95%        | 
| ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | 
| 0.52±8E-3 | 0.47±4E-3 | 0.37±5E-3 | 0.56±1E-3 | 0.49±9E-4 | 0.38±2E-3 | 0.56±7E-4 | 0.51±7E-3 | 0.41±6E-3 |


Performance on Trajectory Recovery ( Macro-F1 ) : 

| 85%        | 90%        | 95%        | 85%        | 90%        | 95%        | 85%        | 90%        | 95%        |
| ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | 
|0.259±2E-4 | 0.217±1E-4 | 0.177±2E-4 | 0.309±2E-4 | 0.258±3E-4 | 0.194±4E-4 | 0.321±6E-4 | 0.269±5E-4 | 0.212±5E-4 |

The Performance in Traffic State Prediction:

| MAE       | MAP       | RMS       | MAE       | MAP        | RMS       | MAE       | MAP        | RMS       | MAE       | MAP        | RMS       |
| --------- | --------- | --------- | --------- | ---------- | --------- | --------- | ---------- | --------- | --------- | ---------- | --------- |
| 0.79±2E-3 | 9.73±3E-2 | 1.74±1E-3 | 1.12±2E-3 | 11.16±6E-1 | 2.10±4E-3 | 1.16±3E-3 | 14.01±4E-2 | 2.14±4E-3 | 1.41±2E-3 | 15.98±2E-1 | 2.47±3E-3 |



# Time comparison：

Compared with previous version (TrajGPT) , BIGST model has less training costs for two reasons:

1) BIGST employs Low-Rank Adaptation (LoRA) to train the backbone, while TrajGPT necessitates the training of all parameters.
2) BIGST trained all downstream tasks simultaneously in Step 2, whereas TrajGPT requires separate training from scratch for each task. Consequently, the total training cost of BIGST across all tasks is less than the sum of the individual training cost for each task.

To illustrate these points, we compared the training times of TrajGPT and BIGST on XA dataset during Steps 1 and 2, as shown in the table below:

Pretraining  Time :

|         | Stage 1（1 epoch) | number of parameters |
| :-------: | :-----------------: | ------|
| TrajGPT | 982.87 (s)    |  131650454        |
| BIGST   | 679.31   (s)    |  11848054          |

Tuning  Time :

|         | Stage 2|
| :-------: | :-----------------: |
| TrajGPT | $\approx$ 1250 (min)         |
| BIGST   | $\approx$  868   (min)        |

# Learnable Prompt：

We proposed a textual instruction mechanism in our paper, and evaluated its importance in paper's Tab 7 (w/o-Inst). 
In our model, ChatGPT is invovled only in data preprocessing stage, providing reference for designing text instruction templates (Sec. 4.2.1). Once defined, textual instruction templates remain unchanged throughout the entire training process. As a result, ChatGPT only influences the design of the textual instruction template.

Inspired by your feedback, we made additional ablation experiments on XA dataset to investigate the impact of ChatGPT on textual instruction templates. Specifically, we trained learnable prompts for each task, corresponding to textual instructions. The experimental results are presented in the table below:

|Task |TTE |CLAS |Next|Reco|Gen|M-step|
|:--:|:--:| :--: |:--:|:--:|:--:|:--:|
|Metric |MAE↓|Ma-F1↑| Acc↑| Acc↑| EDR↓|MAPE↓|
|BIGST (w/o-Ins)|2.07| 0.095 | 0.767 | 0.513 | 0.183 | 15.51 |
|BIGST|1.72 |0.104 |0.837 |0.562 |0.12 | 14.01|
|BIGST (learnable prompt)|1.72 |0.101 |0.815|0.536 |0.15 | 14.57|

Experimental results show that the key lies in providing differentiated instructions as indication for each task. The form of the instruction slightly effects the overall performance of the model, which indicates the performance of the model is decoupled from ChatGPT.
