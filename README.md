# BERTriple

1. Create an environment:
* install LAMA without cloning the git repo again but use the given directory LAMA and follow the instructions from [here](https://github.com/facebookresearch/LAMA)
* install transformers from huggingface by cloning the git repo from [here](https://github.com/huggingface/transformers)

2. Run our experiments:
	```
	python3 run_our_experiments.py
	``` 
3. The evaluation results are under ```results/ ```

5. The plot and the tables of our paper:
	```
	Experiment 1:
	results/BBCF_AUTOPROMPT41_all_obj_3_LAMA.csv
	results/BBCF_AUTOPROMPT41_all_obj_3_LAMA_uhn.csv
	
	Experiment 2:
	AUTOPROMPT41_prec@1.pdf

	Experiment 3:
	precision_per_props_BBCF_AUTOPROMPT41_all_obj_3_LAMA_1.tex
	precision_per_props_BBCF_AUTOPROMPT41_all_obj_3_LAMA_2.tex
	``` 




