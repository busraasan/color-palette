# color-palette
Repository for tracking the progress of color palette generation project using GNNs.


Set dataset path from config.py (will move everything to that config later)

To run training. Config gile is at config folder and content is described in train.py

```
python train.py
```

To run evaluation. Config file is the same one as the training. They should match. 
```
python evaluate.py
```

To process a new annotated dataset and save them as pt files
```
python dataset_processing.py
```

To change the used number of samples, look at dataset.py.
To crate toy dataset, download this [dataset_generator](https://github.com/busraasan/dataset_generator) repo and run the command below. You can specift dataset size from config.
```
python generate.py
```