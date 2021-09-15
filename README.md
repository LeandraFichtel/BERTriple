# BERTriple

1. Create an environment:
* Install the requirements:
	```
	pip3 install -r requirements.txt
	``` 
* Install transformers from huggingface by cloning the git repo from [here](https://github.com/huggingface/transformers)
	```
	pip3 install transformers
	``` 

2. Run our experiments:
	```
	python3 run_experiments.py
	``` 
3. The evaluation results are under ```results/ ```

5. The plot and the tables of our paper:
	```
	Experiment 1:
	results/bertBCF_AUTOPROMPT41_common_all_obj_3_LAMA.csv
	results/bertBCF_AUTOPROMPT41_common_all_obj_3_LAMA_uhn.csv
	
	Experiment 2:
	bertBCF_AUTOPROMPT41_prec@1_3_runs.pdf

	Experiment 3:
	prec_per_props_bertBCF_AUTOPROMPT41_common_all_obj_3_LAMA_1.tex
	prec_per_props_bertBCF_AUTOPROMPT41_common_all_obj_3_LAMA_2.tex

	Experiment 4 (appendix):
	results/distilbertBC_LAMA.csv
	results/distilbertBC_LAMA_uhn.csv
	results/distilbertBCF_AUTOPROMPT41_common_all_obj_3_LAMA.csv
	results/distilbertBCF_AUTOPROMPT41_common_all_obj_3_LAMA_uhn.csv
	results/robertaB_LAMA.csv
	results/robertaB_LAMA_uhn.csv
	results/robertaBF_AUTOPROMPT41_different_all_obj_3_LAMA.csv
	results/robertaBF_AUTOPROMPT41_different_all_obj_3_LAMA_uhn.csv
	results/facebookbartB_LAMA.csv
	results/facebookbartB_LAMA_uhn.csv
	results/facebookbartBF_AUTOPROMPT41_different_all_obj_3_LAMA.csv
	results/facebookbartBF_AUTOPROMPT41_different_all_obj_3_LAMA_uhn.csv
	``` 




