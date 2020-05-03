# Run the transfer learning
python main.py w251fall RGB \
  --arch mobilenetv2 --num_segments 8 \
  --gd 20 --lr 0.001 --lr_steps 10 20 --epochs 25 \
  --batch-size 16 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
  --shift --shift_div=8 --shift_place=blockres \
  --tune_from=/data/w251fall/models/TSM_kinetics_RGB_mobilenetv2_shift8_blockres_avg_segment8_e100_dense.pth
