# Try build k2/snowfall based zeroth recipe

data=corpus
dataset_part="recData01 recData02 recData03 testData01 testData02"
stage=0

. local/parse_options.sh

set -eou pipefail


if [ $stage -le 1 ]; then
  local/prepare_dict.sh data/local/lm data/local/dict_nosp
  
  local/prepare_lang.sh \
    --position-dependent-phones false \
    data/local/dict_nosp \
    "<UNK>" \
    data/local/lang_tmp_nosp \
    data/lang_nosp
  
  echo "To load L:"
  echo "    Lfst = k2.Fsa.from_openfst(<string of data/lang_nosp/L.fst.txt>, acceptor=False)"
fi

if [ $stage -le 2 ]; then
  python3 prepare.py
fi

if [ $stage -le 3 ]; then
  python3 -m torch.distributed.launch --nproc_per_node=3 ./mmi_bigram_train.py 
  --world_size=3
fi

if [ $stage -le 4 ]; then
  # Build G
  if [ ! -f data/lang_nosp/G.fst.txt ]; then
    [ -f data/local/lm/zeroth.lm.tgmed.arpa.gz ] && \
      gunzip data/local/lm/zeroth.lm.tgmed.arpa.gz
    
    python3 -m kaldilm \
      --read-symbol-table="data/lang_nosp/words.txt" \
      --disambig-symbol='#0' \
      --max-order=3 \
      data/local/lm/zeroth.lm.tgmed.arpa >data/lang_nosp/G.fst.txt
  else
    echo "Skip generating data/lang_nosp/G.fst.txt"
  fi

  if [ ! -f data/lang_nosp/G_4_gram.fst.txt ]; then
    [ -f data/local/lm/zeroth.lm.fg.arpa.gz ] && \
      gunzip data/local/lm/zeroth.lm.fg.arpa.gz
    python3 -m kaldilm \
      --read-symbol-table="data/lang_nosp/words.txt" \
      --disambig-symbol='#0' \
      --max-order=4 \
      data/local/lm/zeroth.lm.fg.arpa >data/lang_nosp/G_4_gram.fst.txt
  else
    echo "Skip generating data/lang_nosp/G_4_gram.fst.txt"
  fi

  echo ""
  echo "To load G:"
  echo "Use::"
  echo "  with open('data/lang_nosp/G.fst.txt') as f:"
  echo "    G = k2.Fsa.from_openfst(f.read(), acceptor=False)"
  echo ""
fi
