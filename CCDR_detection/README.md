## CCDR Training Set Search


<!-- ## Requirements

Please refer to [requirment](https://github.com/yorkeyao/DataCV2024/tree/main/task_model) of the task model. Additionally, we need:

- Sklearn
- Scipy 1.2.1 -->

## Datasets Preparation

Please refer to [dataset download page](https://github.com/yorkeyao/DataCV2024/tree/main) for datasets. After all data is ready, please modify the paths for dataset in 'trainingset_search_detection_vehicle.py'

## Running example 

For running such a process, when region100 is used as the target, we can search a training set with 8000 images using the command below:

```python
python trainingset_search_detection_vehicle.py --target 'region100' \
--select_method 'SnP' --c_num 50 \
--result_dir 'main_results/sample_data_detection_vehicle_region100/' \
--n_num 8000 \
--output_data '/data/detection_data/trainingset_search/SnP_region100_vehicle_8000_random_c_num50.json'  
```
Please modify the output json file to a suitable place. After a training set is searched. Please use the [task model](https://github.com/yorkeyao/DataCV2024/tree/main/task_model) to get a prediction for evaluation.  



