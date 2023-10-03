#python examples/pretrain/train_reID.py
#    -dt "dukemtmc" -ds "market1501" \
#    -a "resnet50bb" -e "" --feature 0 --iters 200 --print-freq 100 \
#        --num-instances 16 -b 128 -j 4 --seed 123 --margin 0.3 \
#    --warmup-step 10 --lr 0.00035 --milestones 40 70   --epochs 80  --eval-step 1 \
#        --logs-dir "../saves/soict/m2d/baseline"   --data-dir "../datasets"


#python examples/pretrain/train_reID.py \
#    -dt "dukemtmc" -ds "market1501" \
#    -a "resnet50bb" -e "" --feature 0 --iters 200 --print-freq 100 \
#        --num-instances 16 -b 128 -j 4 --seed 123 --margin 0.3 \
#    --warmup-step 10 --lr 0.00035 --milestones 40 70   --epochs 60  --eval-step 1 \
#        --logs-dir "../saves/soict/m2d/baseline-Cam-loss-b128i200"   --data-dir "../datasets" --use-syn-cam

#python examples/pretrain/train_reID.py \
#    -dt "dukemtmc" -ds "market1501" \
#    -a "resnet50bb" -e "" --feature 0 --iters 200 --print-freq 100 \
#        --num-instances 16 -b 128 -j 4 --seed 123 --margin 0.3 \
#    --warmup-step 10 --lr 0.00035 --milestones 40 70   --epochs 60  --eval-step 1 \
#        --logs-dir "../saves/soict/m2d/baseline-Cam-loss-b128i200-iqabc1"   --data-dir "../datasets" --use-syn-cam --iqa --bm 50 --bs 100 --bc 1.


python examples/pretrain/train_reID.py \
    -dt "dukemtmc" -ds "market1501" \
    -a "resnet50bb" -e "" --feature 0 --iters 200 --print-freq 100 \
        --num-instances 16 -b 128 -j 4 --seed 123 --margin 0.3 \
    --warmup-step 10 --lr 0.00035 --milestones 40 70   --epochs 80  --eval-step 1 \
        --logs-dir "../saves/soict/m2d/baseline-Cam-loss-b128i200-iqabc0.8"   --data-dir "../datasets" --use-syn-cam --iqa --bm 50 --bs 120 --bc 0.8


python examples/pretrain/train_reID.py \
    -dt "dukemtmc" -ds "market1501" \
    -a "resnet50bb" -e "" --feature 0 --iters 200 --print-freq 100 \
        --num-instances 16 -b 128 -j 4 --seed 123 --margin 0.3 \
    --warmup-step 10 --lr 0.00035 --milestones 40 70   --epochs 80  --eval-step 1 \
        --logs-dir "../saves/soict/m2d/baseline-Cam-loss-b128i200-iqabc0.6"   --data-dir "../datasets" --use-syn-cam --iqa --bm 50 --bs 120 --bc 0.6


python examples/pretrain/train_reID.py \
    -dt "dukemtmc" -ds "market1501" \
    -a "resnet50bb" -e "" --feature 0 --iters 200 --print-freq 100 \
        --num-instances 16 -b 128 -j 4 --seed 123 --margin 0.3 \
    --warmup-step 10 --lr 0.00035 --milestones 40 70   --epochs 80  --eval-step 1 \
        --logs-dir "../saves/soict/m2d/baseline-Cam-loss-b128i200-iqabc0.4"   --data-dir "../datasets" --use-syn-cam --iqa --bm 50 --bs 120 --bc 0.4

python examples/pretrain/train_reID.py \
    -dt "dukemtmc" -ds "market1501" \
    -a "resnet50bb" -e "" --feature 0 --iters 200 --print-freq 100 \
        --num-instances 16 -b 128 -j 4 --seed 123 --margin 0.3 \
    --warmup-step 10 --lr 0.00035 --milestones 40 70   --epochs 80  --eval-step 1 \
        --logs-dir "../saves/soict/m2d/baseline-Cam-loss-b128i200-iqabc0.2"   --data-dir "../datasets" --use-syn-cam --iqa --bm 50 --bs 120 --bc 0.2