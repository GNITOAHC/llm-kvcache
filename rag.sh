# parser.add_argument('--index', choices=['gemini', 'openai', 'bm25'], required=True, help='Index to use (gemini, openai, bm25)')
# parser.add_argument('--dataset', choices=[dataset.KIS_SAMPLE, dataset.KIS], required=True, help='Dataset to use (kis_sample, kis)')
# parser.add_argument('--similarity', choices=['bertscore'], required=True, help='Similarity metric to use (bertscore)')
# parser.add_argument('--output', required=True, type=str, help='Output file to save the results')

# Run the script
# BM25, KIS Sample, BERTScore
echo "Running RAG with BM25"
python ./rag.py --index bm25 --dataset kis_sample --similarity bertscore --output ./results/rag_bm25_kisSample_bertscore.txt
# OpenAI, KIS Sample, BERTScore
echo "Running RAG with OpenAI"
python ./rag.py --index openai --dataset kis_sample --similarity bertscore --output ./results/rag_openai_kisSample_bertscore.txt
# Gemini, KIS Sample, BERTScore
echo "Running RAG with Gemini"
python ./rag.py --index gemini --dataset kis_sample --similarity bertscore --output ./results/rag_gemini_kisSample_bertscore.txt

python ./rag.py --index openai --dataset kis --similarity bertscore --output ./results/rag_openai_kis_bertscore.txt
python ./rag.py --index bm25 --dataset kis --similarity bertscore --output ./results/rag_bm25_kis_bertscore.txt
python ./rag.py --index gemini --dataset kis --similarity bertscore --output ./results/rag_gemini_kis_bertscore.txt
