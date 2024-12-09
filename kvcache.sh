# parser.add_argument('--kvcache', choices=['file', 'variable'], required=True, help='Method to use (from_file or from_var)')
# parser.add_argument('--similarity', choices=['bertscore'], required=True, help='Similarity metric to use (bertscore)')
# parser.add_argument('--dataset', choices=['kis', 'kis_sample', 'squad-dev', 'squad-train'], required=True, help='Dataset to use (kis, kis_sample, squad-dev, squad-train)')
# parser.add_argument('--output', required=True, type=str, help='Output file to save the results')

# read kvcache from variable will consume lots of memory
# python ./kvcache.py --kvcache variable --dataset kis_sample --similarity bertscore --output ./results/kisSample/kvcache_variable_kisSample_bertscore.txt

# Run the script
# read kvcache from file, dataset kis_sample
# echo "Running KVCACHE for KIS Sample"
# python ./kvcache.py --kvcache file --dataset kis_sample --similarity bertscore --output ./results/kisSample/kvcache_file_kisSample_bertscore.txt

# Run Squad-dev
echo "Running KVCACHE for Squad-dev"
python ./kvcache.py --kvcache file --dataset squad-dev --similarity bertscore --output ./results/squadDev/kvcache_file_squadDev_bertscore.txt
