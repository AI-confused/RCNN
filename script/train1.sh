export CUDA_VISIBLE_DEVICES=3
k=5
mkdir ../output/model_rcnn2
mkdir ../output/model_rcnn2/fold_$k
cd ../src
for((i=0;i<k;i++));  
do   
python3 main.py \
-server-ip='10.15.82.239' \
-do-predict=1 \
-seq-len=400 \
-input-size=768 \
-hidden-size=768 \
-linear-size=100 \
-do-train=1 \
-train-steps=100 \
-concat=1 \
-model-type='gru' \
-batch-size=256 \
-port=8190 \
-port-out=5556 \
-model=../output/model_rcnn2/fold_$k/model_$i.pkl \
-train-file=fold_$k/data_$i/train.csv \
-dev-file=fold_$k/data_$i/dev.csv \
-test-file=fold_$k/data_$i/test.csv \
-predict-file=../output/model_rcnn2/fold_$k/test_result_$i.csv \
-load-model=../output/model_rcnn2/fold_$k/model_$i.pkl \
-eval-result=../output/model_rcnn2/fold_$k/eval_result_$i.txt

done