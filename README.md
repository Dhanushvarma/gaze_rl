# Gaze for Robot Learning

This repo contains the code implementation for using gaze to help teleoperate a robot manipulator.

## PointCross Experiments

Collect trajectory data on the PointCross environment. Collect demos along both diagonals and join these demos to create a final dataset.

```
# Collect data
python scripts/generate_data_pc_tfds.py

python main.py --config-name=train_bc
```

## RLBench Experiments

RLBench Setup: 
```
# set env variables
export COPPELIASIM_ROOT=${HOME}/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

### Data generation
```
DISPLAY=:99 python RLBench/rlbench/dataset_generator.py --save_path /scr/shared/gaze_rl/dataset_debug_img --tasks pick_block_easy --episodes_per_task 5 --processes 2

# Convert your data to tfds format
python3 scripts/convert_rlbench_to_tfds.py \
    --ds_name=dataset_debug_img \
    --task=pick_block_easy \
    --save_imgs
```

### Training 
```
# cannot use multienv with EEP, cause of invalid path errors
DISPLAY=:99 python3 main.py --config-name=train_bc


DISPLAY=:99 python3 main.py --config-name=train_hbc 

# With images
DISPLAY=:99 python3 main.py --config-name=train_bc \
    env.image_obs=True \
    model/encoder=cnn 

# R3M
DISPLAY=:99 python3 main.py --config-name=train_bc \
    env.image_obs=True \
    model.feature_extractor=r3m 

```

## TODOS
[] Image-based training / rollouts

## Extra

Need this for the pretty latex text in matplotlib plots unless latex already exists
```
sudo apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super

# Update conda environment
conda env update --file local.yml --prune
```

RLBench problems:
https://stackoverflow.com/questions/60042568/this-application-failed-to-start-because-no-qt-platform-plugin-could-be-initiali
