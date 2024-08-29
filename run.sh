/home/sankeerth/miniconda3/envs/livegs_shangar/bin/python train_compress.py \
-s /home/sankeerth/data/sankeerth/bonsai \
--data_device cuda \
--output_vq ./output/bonsai_compressed_train \
-m ./output/bonsai_compressed_train -r 4 --eval