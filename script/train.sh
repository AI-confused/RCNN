export CUDA_VISIBLE_DEVICES=2
k=5
mkdir ../output/model_rcnn
mkdir ../output/model_rcnn/fold_$k
cd ../src
for((i=0;i<k;i++));  
do   
python3 main.py \
-server-ip='10.15.82.239' \
-do-predict=0 \
-seq-len=400 \
-input-size=768 \
-concat=1 \
-do-train=1 \
-batch-size=512 \
-port=8190 \
-port-out=5556 \
-model=../output/model_rcnn/fold_$k/model_$i.pkl \
-train-file=fold_$k/data$i/train.csv \
-dev-file=fold_$k/data$i/dev.csv \
-test-file=fold_$k/data$i/test.csv \
-predict-file=../output/model_rcnn/fold_$k/test_result_$i.csv \
-load-model=../output/model_rcnn/fold_$k/model_$i.pkl \
-eval-result=../output/model_rcnn/fold_$k/eval_result_$i.txt

done