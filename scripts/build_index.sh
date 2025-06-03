corpus=2wikimultihopqa
# corpus=wiki-18
# corpus=musique
# corpus=hotpotqa

# change retriever_name to bm25 for BM25 indexing
# retriever_name=bm25
retriever_name=e5
retriever_model=intfloat/e5-base-v2

while [[ $# -gt 0 ]]; do
  case $1 in
    --corpus)
      corpus="$2"
      shift 2
      ;;
    --retriever_name)
      retriever_name="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

corpus_file=data/corpus/$corpus
save_dir=data/corpus/$corpus/index

# change faiss_type to HNSW32/64/128 for ANN indexing


CUDA_VISIBLE_DEVICES=0,1,2,3 python src/search/index_builder.py \
    --retrieval_method $retriever_name \
    --model_path $retriever_model \
    --corpus_path $corpus_file \
    --save_dir $save_dir \
    --use_fp16 \
    --max_length 512 \
    --batch_size 512 \
    --pooling_method mean \
    --faiss_type Flat
    # --save_embedding 
 
