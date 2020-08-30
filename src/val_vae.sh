python3 src/temporal_vae.py --batch 32 --epoch 300 --data colon --step 1 --ex wo_triplet --mode val --ndeconv 3
wait
python3 src/temporal_vae.py --batch 32 --epoch 300 --data colon --step 1 --ex all_loss --mode val --triplet --ndeconv 3