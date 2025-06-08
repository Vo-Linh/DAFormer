docker run -dp 6006:6006 -p 5000:5000 -p 8080:8080 --gpus all --rm --shm-size=8g --name muca_runner --network host \
-v "/home/Hung_Data/HungData/mmseg_data/Datasets/LoveDA:/source/dataset" \
-v "/home/Hung_Data/HungData/Thien/DAFormer:/source" \
-it mmsegmentation:1.2.2