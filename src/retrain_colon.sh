python3 src/temporal_vae.py --ex wo_triplet --batch 64 --epoch 2000 --data colon --ndeconv 3 --retrain --step 10
wait
python3 src/temporal_vae.py --ex all_loss --batch 32 --epoch 2000 --triplet --data colon --ndeconv 3 --retrain --step 10