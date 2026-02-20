#/bin/bash

python -m ipdb -c c invert_sentences.py \
--sentences="set a new record for today" \
--init_mlm_top_k=1000 \
--init_strategy='mlm_mask' \
--promptTfComplLambda=1.0e-3 \
--early_stop_on_exact_match=False \
--early_stop_loss_threshold=0.001 \
