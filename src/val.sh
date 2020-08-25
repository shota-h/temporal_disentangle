# python3 src/temporal_disentanglement.py --ex wo_triplet --d2ae --batch 64 --epoch 1000 --rec 10
# wait
# python3 src/temporal_disentanglement.py --ex wo_triplet_rev --batch 64 --d2ae --rev --epoch 1000 --rec 10
# wait
python3 src/temporal_disentanglement.py --ex backup200819/all_loss_ws4 --d2ae --epoch 1000 --mode val --triplet
wait
python3 src/temporal_disentanglement.py --ex backup200819/all_loss_rev_ws4 --d2ae --epoch 1000 --mode val --rev --triplet
wait
python3 src/temporal_disentanglement.py --ex backup200819/wo_triplet_ws4 --d2ae --epoch 1000 --mode val
wait
python3 src/temporal_disentanglement.py --ex backup200819/wo_triplet_rev_ws4 --d2ae --epoch 1000 --mode val --rev
wait