import yaml

lr_list = [0.01, 0.005, 0.001]
weight_decay = [0, 0.01, 0.001, 0.0001]

count = 0
for i, lr in enumerate(lr_list):
    for j, wd in enumerate(weight_decay):
        with open("conf"+str(count)+".yaml", "w") as file:
            data = {
                "model_name": "ColorGNNEmbedding_lr"+str(lr)+"_wd"+str(wd),
                "data_type": "processed_rgb",
                "feature_size": 1005,

                "device": "cuda:2",
                "lr": lr,
                "batch_size": 1,
                "weight_decay": wd,
                "num_epoch": 150,

                "dataset_root": "../destijl_dataset",
            }
            documents = yaml.dump(data, file)
        count+=1
