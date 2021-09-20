# BERTriple

This GitHub repo belongs to the paper [Prompt Tuning or Fine-Tuning - Investigating Relational Knowledge in Pre-Trained Language Models](https://www.akbc.ws/2021/papers/o7sMlpr9yBW) published on AKBC2021.
```
@inproceedings{
fichtel2021prompt,
title={Prompt Tuning or Fine-Tuning - Investigating Relational Knowledge in Pre-Trained Language Models},
author={Leandra Fichtel and Jan-Christoph Kalo and Wolf-Tilo Balke},
booktitle={3rd Conference on Automated Knowledge Base Construction},
year={2021},
url={https://openreview.net/forum?id=o7sMlpr9yBW},
doi={}
}
```

## Follow the instructions to reproduce our evaluation results :)

1. Create an environment:
	```
	conda create --name bertriple python=3.6.9
	```
2. Install the requirements:
	```
	pip install -r requirements.txt
	```
3. Run our experiments:
	```
	python run_experiments.py
	``` 
4. The evaluation results are under ```results/ ```

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
	results/baselines_common_vocab/bertLC_LAMA.csv
	results/baselines_common_vocab/bertLC_LAMA_uhn.csv.csv
	results/bertLCF_AUTOPROMPT41_common_all_obj_3_LAMA.csv
	results/bertLCF_AUTOPROMPT41_common_all_obj_3_LAMA_uhn.csv
	results/baselines_common_vocab/distilbertBC_LAMA.csv
	results/baselines_common_vocab/distilbertBC_LAMA_uhn.csv
	results/distilbertBCF_AUTOPROMPT41_common_all_obj_3_LAMA.csv
	results/distilbertBCF_AUTOPROMPT41_common_all_obj_3_LAMA_uhn.csv
	results/baselines_different_vocab/robertaB_LAMA.csv
	results/baselines_different_vocab/robertaB_LAMA_uhn.csv
	results/robertaBF_AUTOPROMPT41_different_all_obj_3_LAMA.csv
	results/robertaBF_AUTOPROMPT41_different_all_obj_3_LAMA_uhn.csv
	results/baselines_different_vocab/facebookbartB_LAMA.csv
	results/baselines_different_vocab/facebookbartB_LAMA_uhn.csv
	results/facebookbartBF_AUTOPROMPT41_different_all_obj_3_LAMA.csv
	results/facebookbartBF_AUTOPROMPT41_different_all_obj_3_LAMA_uhn.csv
	``` 




