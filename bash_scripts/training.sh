FROM_FILE=0
TO_FILE=11

file=$FROM_FILE
device_idx=2

free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $device_idx | grep -Eo [0-9]+)

while [ $file -le $TO_FILE ]
do
    # if [ $free_mem -lt 13000 ]; then
    #     while [ $free_mem -lt 13000 ]; do
    #         sleep 10
    #         free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $device_idx | grep -Eo [0-9]+)
    #     done
    # fi

    echo "Running experiment for conf$file.yaml"
    #nohup python train.py --config_file config/conf$file.yaml > "config/out_$file.txt" &
    python evaluate.py --config_file config/conf$file.yaml
    file=$(($file+1))
    #sleep 10
    
done