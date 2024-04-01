# DataCV2024
This is the development kit repository for [the 2nd CVPR DataCV Challenge](https://sites.google.com/view/vdu-cvpr24/competition/). Here you can find details on how to download datasets.
 
# Running Environment for Classifier Guided Cluster Density Reduction (CCDR)

To manage the python code, it is recommended to install [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html).

For creating environment,

```python
conda create -n ccdr python=3.10 -y
conda activate ccdr
```

Besides, you will need to install pytorch 2.0.0, please modify cuda version according to the version installed in the system. 

```python
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```

The install of mmcv library. 

```python
pip install -U openmim
mim install mmcv-full==1.7.2
pip install yapf==0.40.1
```

The install of mmdetction.

```python
cd task_model/mmdetection/
pip install -v -e . 
```

Additionally, other required libraries can be installed with the following command:

```python
pip install -r  requirements.txt
```

The installation of pycococreator
Method 1:
```python
pip install git+git://github.com/waspinator/pycococreator.git@0.2.0
```

Method 2:
1. install cython
```python
pip install Cython
```
2. Download the repo, https://github.com/waspinator/pycococreator
3. Extract the repo
4. Go to the repo directory and run the command:
```python
python setup.py install
```

The installation of jax
```python
pip install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Set up Scenic
git clone the scenic and run the command:

```python
cd DataCV2024_comp
git clone https://github.com/google-research/scenic.git
cd /path/to/scenic
python setup.py install
```
 
# Running example
For running such a process, when region100 is used as the target, we can search a training set with 8000 images using the command below:
```python
python trainingset_search_detection_vehicle.py --target 'region100' \
--select_method 'CCDR' --c_num 50 \
--result_dir 'main_results/sample_data_detection_vehicle_region100/' \
--n_num 8000 \
--output_data 'CCDR_region100_vehicle_8000_c_num50.json' 
```

To use the classifier to score selected candidates, see `classifier/README.md` for details.

# Reproducing LB results
The trained weights can be downloaded from([LB weights for trained model](https://drive.google.com/file/d/1VkZjCrkSQ4qRE03XaOFR907NQaGPgGMk/view?usp=sharing)
To perform inference
```
python tools/test.py DataCV2024-main/task_model/mmdetection/workdirs/retinanet_r50_fpn_1x_custom_tss_car.py DataCV2024-main/task_model/mmdetection/workdirs/epoch_12.pth --eval bbox

```
To produce our leaderboard submission for the competition:
```
python tools/test.py  DataCV2024-main/task_model/configs_tss/retinanet/retinanet_r50_fpn_1x_custom_tss_car.py DataCV2024-main/task_model/mmdetection/workdirs/epoch_12.pth  --format-only --options "jsonfile_prefix=./"
```

