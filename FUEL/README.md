# Fair and Consistent Federated Learning


## introduction for all files
* dataset_generate.py: the preprocessing code for processing the Adult dataset following exactly the same data processing procedures described in (https://arxiv.org/abs/1902.00146 and https://openreview.net/forum?id=ByexElSYDr).
* dataload.py: loading dataset for the model
* hco_lp.py: the function about searching for a descent direction in stage 1 (constrained min-max optimization)
* po_lp.py: the function about searching for a descent direction in stage 2 (constrained Pareto optimization)
* hco_model.py: the source code of the model including training, testing and saving.
* utils.py: needed function for the implementation
* main.py: the main function for all real-world dataset experiments.

## requirements
python 3.6, the needed libraries are in requirements.txt

## dataset generatation:
* download the original adult dataset:  
`wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data`  
`wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test`  

* dataset split
`python dataset_generate.py`
the split dataset is in ./data/train and ./data/test

## synthetic experiment 
    We conduct the experiments on two different setting 1) the initilization satisfies fairness constraints; 2) the initilization violates the fairness constraints. All results in the two setting can be reproducted by: 
    `cd synthetic`
    `python experiment.py`
and the generated results are in ./synthetic/figures/

## real-world dataset experiment

    **part of the parameter meanning in main.py**
    * sensitive_attr:  sensitive attribute ("race" or "sex")
    * step_size: learning rate
    * eps_g: uniform fairness budget (all clients are assigned the same fairness budget)
    * max_epoch_stage1: the max iteration num for stage 1 (constrained min-max optimization)
    * max_epoch_stage2: the max iteration num for stage 2 (constrained Pareto optimization)
    * uniform_eps: bool variable (if we adopt uniform fairness budget setting, uniform_eps == True)
    * weight_eps: the ratio for specific client budget ("w" in the main text, specific client budget setting)
    * target_dir_name: the dir for saving results (including models, logs and args)

### uniform fairness budget on both clients 
*   sensitive atttribute as ***race***,  fairness budget as ***0.05***  
`python main.py --step_size 0.03 --eps_g 0.05 --sensitive_attr race --max_epoch_stage1 800 --max_epoch_stage2 1500 --seed 1 --target_dir_name race_DP_0-05 --uniform_eps`  
***results:*** is_train: False, epoch: 2299, loss: [0.5601576566696167, 0.39812201261520386], accuracy: [0.7292817831039429, 0.8195652365684509], auc: [0.6859285714285713, 0.8504551633040838], disparity: [0.0005307793617248535, 0.04531855136156082], pred_disparity: [0.0260276198387146, 0.04738526791334152]
*   sensitive attribute as ***race***, fairnesss budget as ***0.01***
`python main.py --step_size 0.015 --eps_g 0.01 --sensitive_attr race --max_epoch_stage1 800 --max_epoch_stage2 1500 --seed 1 --target_dir_name race_DP_0-01 --uniform_eps`  
***results:*** is_train: False, epoch: 2299, loss: [0.5744820237159729, 0.4304683208465576], accuracy: [0.7016574740409851, 0.7970807552337646], auc: [0.6947857142857142, 0.837447616279523], disparity: [-0.002919316291809082, 0.0034494660794734955], pred_disparity: [-0.0016486644744873047, 0.008047450333833694]

### specific fairness budget on both clients 
  take an example, we set sensitive attribute as ***race***, the original disparity is $\Delta DP = [DP_1， DP_2]$.
  The original disparities of both clients is measured by running：  
  `python main.py --step_size 0.13 --max_epoch_stage1 800 --max_epoch_stage2 1000 --eps_g 1.0 --sensitive_attr race --seed 1 --target_dir_name race_specific_1-0 --uniform --uniform_eps`  
***results:*** is_train: False, epoch: 1799, loss: [0.5699087977409363, 0.3846117854118347], accuracy: [0.7348066568374634, 0.8285093307495117], auc: [0.6905, 0.8529878658361068], disparity: [0.15790873765945435, 0.07422562688589096], pred_disparity: [0.15371835231781006, 0.07583809643983841]
  where there is no fairness constraints, and the measured original disparity are [0.15790873765945435, 0.07422562688589096].

*   w = 0.8  (the fairness budget of both clients are 0.8* $\Delta DP$)
    `python main.py --weight_eps 0.8 --max_epoch_stage1 800 --max_epoch_stage2 1000 --step_size 0.07 --sensitive_attr race --seed 1 --target_dir_name race_specific_0-8`  
***results:*** is_train: False, epoch: 1799, loss: [0.5635322332382202, 0.38921427726745605], accuracy: [0.7513812184333801, 0.8263354301452637], auc: [0.6877857142857143, 0.8522060544186512], disparity: [0.11624205112457275, 0.06827643513679504], pred_disparity: [0.1026315689086914, 0.0648479089140892]

*   w = 0.5  (the fairness budget of both clients are 0.5* $\Delta DP$)
    `python main.py --weight_eps 0.5 --max_epoch_stage1 800 --max_epoch_stage2 1000 --step_size 0.05 --sensitive_attr race --seed 1 --target_dir_name race_specific_0-5`  
***results:*** is_train: False, epoch: 1799, loss: [0.5625894069671631, 0.394794762134552], accuracy: [0.7403315305709839, 0.820621132850647], auc: [0.6882142857142858, 0.8510285611480758], disparity: [0.05838644504547119, 0.04976653307676315], pred_disparity: [0.036145806312561035, 0.05463084578514099]

*   w = 0.2  (the fairness budget of both clients are 0.5* $\Delta DP$)
    `python main.py --weight_eps 0.2 --max_epoch_stage1 800 --max_epoch_stage2 1000 --step_size 0.03 --sensitive_attr race --seed 1 --target_dir_name race_specific_0-2`  
***results:*** is_train: False, epoch: 1799, loss: [0.5699670910835266, 0.41371434926986694], accuracy: [0.7182320952415466, 0.8140993714332581], auc: [0.6844999999999999, 0.8462134911794057], disparity: [-0.0092887282371521, 0.03283762186765671], pred_disparity: [-0.0035886168479919434, 0.024312380701303482]

### Equal opportunity measurement with uniform fairness budget on both clients
EO is also a popular disparity metric. The experiment results can be easily obtained by assigning "Eoppo" to the parameter "disparity_type".

For example, we select the sensitive attribute as ***race***, fairness budget is 0.05:  

`python main.py --sensitive_attr race --step_size 0.1 --disparity_type Eoppo  --max_epoch_stage1 800 --max_epoch_stage2 220 --eps_g 0.05 --target_dir_name adult_race_Eoppo_0-1 --seed 1 --uniform_eps`  

***results:*** is_train: False, epoch: 1019, loss: [0.5672342777252197, 0.4211762547492981], accuracy: [0.7237569093704224, 0.8137267231941223], auc: [0.6760714285714287, 0.8362089015217425], disparity: [-0.054545462131500244, 0.02255287766456604], pred_disparity: [-0.017567992210388184, 0.034288644790649414]  

if we set the sensitive attribute as ***sex***, fairness budget is 0.05:

`python main.py --sensitive_attr sex --step_size 0.1 --disparity_type Eoppo  --max_epoch_stage1 800 --max_epoch_stage2 220 --eps_g 0.05 --target_dir_name adult_sex_Eoppo_0-1 --seed 1 --uniform_eps`  

***results:*** is_train: False, epoch: 1019, loss: [0.5760114789009094, 0.4152376055717468], accuracy: [0.6961326003074646, 0.8184472322463989], auc: [0.6787857142857142, 0.8374747968830623], disparity: [0.0006434917449951172, -0.04236716032028198], pred_disparity: [0.0183182954788208, -0.04114463925361633]

