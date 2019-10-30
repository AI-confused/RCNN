export CUDA_VISIBLE_DEVICES=3
k=5
m=4
mkdir ../output/model_rcnn$m
mkdir ../output/model_rcnn$m/fold_$k
cd ../src
for((i=0;i<k;i++));  
do   
python3 main.py \
-server-ip='10.15.82.239' \
-do-predict=1 \
-seq-len=400 \
-input-size=768 \
-hidden-size=768 \
-linear-size=768 \
-do-train=1 \
-train-steps=200 \
-concat=1 \
-lr=1e-4 \
-model-type='gru' \
-batch-size=128 \
-port=8190 \
-port-out=5556 \
-model=../output/model_rcnn$m/fold_$k/model_$i.pkl \
-train-file=fold_$k/data_$i/train.csv \
-dev-file=fold_$k/data_$i/dev.csv \
-test-file=fold_$k/data_$i/test.csv \
-predict-file=../output/model_rcnn$m/fold_$k/test_result_$i.csv \
-load-model=../output/model_rcnn$m/fold_$k/model_$i.pkl \
-eval-result=../output/model_rcnn$m/fold_$k/eval_result_$i.txt

done