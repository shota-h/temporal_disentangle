# T1=100
# T2=400
# e=500
T1=10
T2=40
e=500

for ratio in 0.8 0.6 0.4 0.2; do
    python3 src/ssl_seq_smooth_labeling.py --train --data colon --ratio $ratio --seq_base --epoch 500 --adv 0.0 --batch 64 --step 1 --test
    python3 src/ssl_seq_smooth_labeling.py --train --data colon --ratio $ratio --seq_base --epoch 500 --adv 1.0 --batch 64 --step 1 --test
done
# python3 src/ssl_seq_smooth_labeling.py --data colon --batch 64 --unlabeled_batch 32 --ratio $ratio --step 1 --adapt_alpha --adv 0 --seq_base --epoch $e --train

# python3 src/ssl_seq_smooth_labeling.py --data colon --batch 64 --unlabeled_batch 32 --ratio $ratio --step 1 --adapt_alpha --test --adv 0 --seq_base --epoch $e
# python3 src/ssl_seq_smooth_labeling.py --data colon --batch 64 --unlabeled_batch 32 --ratio $ratio --step 1 --adapt_alpha --test --adv 0 --seq_base --epoch $e --temp

# python3 src/ssl_seq_smooth_labeling.py --data colon --batch 64 --unlabeled_batch 32 --ratio $ratio --train --step 1 --adapt_alpha --test --adv 0 --c2 0 --seq_base --epoch $e
# python3 src/ssl_seq_smooth_labeling.py --data colon --batch 64 --unlabeled_batch 32 --ratio $ratio --train --step 1 --adapt_alpha --test --adv 0 --c1 0 --seq_base --epoch $e
# python3 src/ssl_seq_smooth_labeling.py --data colon --batch 64 --unlabeled_batch 32 --ratio $ratio --train --step 1 --adapt_alpha --test --adv 0 --c2 0 --seq_base --epoch $e

# for ratio in 0.2 0.4 0.6 0.8; do
#     python3 src/ssl_seq_smooth_labeling.py --data colon --batch 64 --unlabeled_batch 32 --semi --ratio $ratio --train --step 1 --adapt_alpha --test --adv 0 --c1 0 --seq_base --epoch $e

#     python3 src/ssl_seq_smooth_labeling.py --data colon --batch 64 --unlabeled_batch 32 --semi --ratio $ratio --train --step 1 --adapt_alpha --test --adv 0 --c2 0 --seq_base --epoch $e
    
# Hard Labeling
    # for smooth in 0 1 5 10; do
    #     python3 src/ssl_seq_smooth_labeling.py --data colon --batch 64 --unlabeled_batch 32 --semi --ratio $ratio --train --step 1 --use_pseudo --smooth $smooth --T1 $T1 --T2 $T2 --alpha 1 --adapt_alpha --test --adv 0 --c1 0 --seq_base --epoch $e
    # done

# Soft Labeling
    # for smooth in 0.5 1; do
    #     # python3 src/ssl_seq_smooth_labeling.py --data colon --batch 64 --soft --unlabeled_batch 32 --semi --ratio $ratio --train --step 1 --use_pseudo --smooth $smooth --T1 $T1 --T2 $T2 --alpha 1 --adapt_alpha --test --adv 0 --c2 0 --epoch $e
    #     python3 src/ssl_seq_smooth_labeling.py --data colon --batch 64 --soft --unlabeled_batch 32 --semi --ratio $ratio --train --step 1 --use_pseudo --smooth $smooth --T1 $T1 --T2 $T2 --alpha 1 --adapt_alpha --test --adv 0 --c1 0 --epoch $e
    # done
# done