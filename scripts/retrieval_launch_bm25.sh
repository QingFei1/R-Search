corpus=2wikimultihopqa
port=8000
# corpus=wiki-18
# port=8001
# corpus=musique
# port=8002
# corpus=hotpotqa
# port=8003

while [[ $# -gt 0 ]]; do
  case $1 in
    --corpus)
      corpus="$2"
      shift 2
      ;;
    --port)
      port="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done


index_file=data/corpus/$corpus/index/bm25
corpus_file=data/corpus/$corpus
corpus_name=$corpus
retriever_name=bm25


python src/search/retrieval_server.py --index_path $index_file \
                                    --corpus_path $corpus_file \
                                    --topk 3 \
                                    --retriever_name $retriever_name \
                                    --corpus_name $corpus_name \
                                    --port $port

