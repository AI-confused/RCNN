k=5
mkdir ../output/model_rcnn
mkdir ../output/model_rcnn/fold_$k
cd ../src
python3 main.py \
-server-ip='10.15.82.239' \
-do-predict=0 \
-do-train=1 \
-port=8190 \
-port-out=5556 \
-model=../output/model_rcnn/fold_$k/model_0.pkl \
-train-file=fold_$k/data0/train_.csv \
-dev-file=fold_$k/data0/dev_.csv \
-test-file=fold_$k/data0/test_.csv \
-predict-file=../output/model_rcnn/fold_$k/test_result_0.csv \
-load-model=../output/model_rcnn/fold_$k/model_0.pkl \
-eval-result=../output/model_rcnn/fold_$k/eval_result_0.txt

