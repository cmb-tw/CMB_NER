CUDA_VISIBLE_DEVICES=0 python training_bertspan_zh_demo.py --task cmb \
                  --num_epochs 1 \
                  --do_train --do_predict \
                  --model_name /data_1/hyjiang/text2vec-base-chinese-paraphrase
