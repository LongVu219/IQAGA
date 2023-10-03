python examples/test_model.py \
-dt "market1501" --data-dir "../datasets" \
-a resnet50 --features 0  -b 8 \
--resume "../saves/reid/duke2market/S1/R50Mix-4:1-lam/model_best.pth.tar" \
#--rerank

# --resume "../saves/reid/duke2market/S2/dauet/model_best.pth.tar" \
#--resume "/home/k64t/person-reid/saves/reid/duke2market/S1/R50wGAN/model_best.pth.tar" \
#--resume "/home/k64t/person-reid/saves/reid/duke2market/S1/R50wDIM/model_best.pth.tar" \
