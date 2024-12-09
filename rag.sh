# parser.add_argument('--index', choices=['gemini', 'openai', 'bm25'], required=True, help='Index to use (gemini, openai, bm25)')
# parser.add_argument('--dataset', choices=['kis', 'kis_sample', 'squad-dev', 'squad-train'], required=True, help='Dataset to use (kis, kis_sample, squad-dev, squad-train)')
# parser.add_argument('--similarity', choices=['bertscore'], required=True, help='Similarity metric to use (bertscore)')
# parser.add_argument('--output', required=True, type=str, help='Output file to save the results')

# Run the script
# BM25, KIS Sample, BERTScore
# echo "Running RAG with BM25"
# python ./rag.py --index bm25 --dataset kis_sample --similarity bertscore --output ./results/kisSample/rag_bm25_kisSample_bertscore.txt
# # OpenAI, KIS Sample, BERTScore
# echo "Running RAG with OpenAI"
# python ./rag.py --index openai --dataset kis_sample --similarity bertscore --output ./results/kisSample/rag_openai_kisSample_bertscore.txt
# # Gemini, KIS Sample, BERTScore
# echo "Running RAG with Gemini"
# python ./rag.py --index gemini --dataset kis_sample --similarity bertscore --output ./results/kisSample/rag_gemini_kisSample_bertscore.txt


# Run Squad-dev
# BM25, KIS Sample, BERTScore
echo "Running RAG with BM25 for Squad-dev"
python ./rag.py  --index bm25 --dataset squad-dev --similarity bertscore --output ./results/squadDev/rag_bm25_squadDev_bertscore.txt
# OpenAI, KIS Sample, BERTScore
echo "Running RAG with OpenAI for Squad-dev"
python ./rag.py --index openai --dataset squad-dev --similarity bertscore --output ./results/squadDev/rag_openai_squadDev_bertscore.txt
# Gemini, KIS Sample, BERTScore
echo "Running RAG with Gemini for Squad-dev"
python ./rag.py --index gemini --dataset squad-dev --similarity bertscore --output ./results/squadDev/rag_gemini_squadDev_bertscore.txt
