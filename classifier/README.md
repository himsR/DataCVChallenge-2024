### data preparation
- extract the vector representation of the images, save them at:
    - `/ubuntu/comp/data/extracted_features/176k_filename(train).pkl: List[str]` the list of filenames of the 176k source images(/region_100 train)
    - `/ubuntu/comp/data/extracted_features/176k_features(train).npy: np.ndarray` the corresponding representation vectors of the 176k source images(/region_100 train)


### training
##### prepare the training data
run
```bash
python utils/split.py
```
to split the training data into training and validation set, and save them as `train_list.pkl` and `val_list.pkl` respectively.
##### training
```bash
python train.py --config cfgs/config.yaml
```

### inference
#### prepare the candidates input data
- take the candidates images that are pre-selected by the retrieval model, run `utils/json2pkl.py` to make the candidates list into a dictionary pkl file, where the key is the image filename and value is the vector representation of the image. 
    ```bash
    python utils/json2pkl.py \
    --json_path /path/to/candidate/json/dfile \
    --pkl_path /path/to/output/pickle/file
    ```

#### run inference
```bash
python inference.py \
--config cfgs/config.yaml
--inf_output /path/to/output/prediction/json/file
```
this will generate the prediction results in the format of `json` file, where the key is the query image filename and the value is classifier prediction score.