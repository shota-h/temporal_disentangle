for ratio in 0.9 0.8; do
    for adv in 0; do
        python3 src/ssl_seq_smooth_labeling.py --data colon --batch 64--unlabeled_batch 32 --semi --ratio $ratio --train --step 1 --adapt_alpha --test --adv $adv
    done
    
# Hard Labeling
    for smooth in 0 1 10 20; do
        for adv in 0; do
        python3 src/ssl_seq_smooth_labeling.py --data colon --batch 64 --unlabeled_batch 32 --semi --ratio $ratio --train --step 1 --use_pseudo --smooth $smooth --T1 100 --T2 200 --alpha 1 --adapt_alpha --test --adv $adv
        done
    done

# Soft Labeling
    for smooth in 0.5 1 2; do
        for adv in 0; do
            python3 src/ssl_seq_smooth_labeling.py --data colon --batch 64 --soft --unlabeled_batch 32 --semi --ratio $ratio --train --step 1 --use_pseudo --smooth $smooth --T1 100 --T2 200 --alpha 0.1 --adapt_alpha --test --adv $adv
        done
    done
done