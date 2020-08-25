# python3 src/temporal_disentanglement.py --ex wo_triplet --d2ae --batch 64 --epoch 1000 --rec 10
# wait
# python3 src/temporal_disentanglement.py --ex wo_triplet_rev --batch 64 --d2ae --rev --epoch 1000 --rec 10
# wait
python3 src/temporal_vae.py --ex wo_triplet_full_dis --d2ae --batch 64 --epoch 1000
wait
# python3 src/temporal_vae.py --ex all_loss_m1 --d2ae --batch 32 --triplet --epoch 1000 --tri 1e-2
# wait
# python3 src/temporal_vae.py --ex all_loss_rev_m1 --d2ae --batch 32 --triplet --rev --epoch 1000 --tri 1e-2
