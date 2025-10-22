# Concept Incongruence: An Exploration of Time and Death in Role Playing
This is the homepage of the paper [**Concept Incongruence: An Exploration of Time and Death in Role Playing**](https://arxiv.org/abs/2505.14905)

Consider this prompt "Draw a unicorn with two horns". Should large language models (LLMs) recognize that a unicorn has only one horn by definition and ask users for clarifications, or proceed to generate something anyway? 
We introduce *concept incongruence* to capture such phenomena where concept boundaries clash with each other, either in user prompts or in model representations, often leading to under-specified or mis-specified behaviors.
In this work, we take the first step towards defining and analyzing model behavior under concept incongruence.
Focusing on temporal boundaries in the Role-Play setting, we propose three behavioral metrics--abstention rate, conditional accuracy, and answer rate--to quantify model behavior under incongruence due to the role's death. 
We show that models fail to abstain after death and suffer from an accuracy drop compared to the Non-Role-Play setting.
Through probing experiments, we identify two main causes: (i) unreliable encoding of the "death" state across different years, leading to unsatisfactory abstention behavior, and (ii) role playing causes shifts in the model’s temporal representations, resulting in accuracy drops.
We leverage these insights to improve consistency in the model's abstention and answer behaviors. Our findings suggest that concept incongruence leads to unexpected model behaviors and point to future directions on improving model behavior under concept incongruence.
![Screenshot 2025-05-19 at 11 51 12](https://github.com/user-attachments/assets/85aa62d4-b759-4932-9875-8c067989e9c0)

## Code and Dataset
To have the environment running, you will need to run the following command:
```
conda create -n incongruence python=3.11.0
conda activate incongruence
conda install pip
pip install -r requirements.txt
```

## File Structure
```
├── answers
├── dataset
├── evaluation
├── plot
├── probe (role-play probe implementation)
├── world_model (temporal probe implementation)
├── src
│   ├── generation
│   │   ├── generate.py
│   │   ├── ai_generation.py (used for non-role-playing generation)
│   │   ├── alive_generation.py (used for alive people generation)
│   ├── eval
│   │   ├── evaluate.py (abstain and answer)
│   │   ├── accuracy_label.py (accuracy)
│   ├── plot_prep
│   │   ├── stat_yearly.py (only for yearly question, stat and plot)
│   │   ├── extract_evaluation.py (extract the label into 0 and 1)
│   │   ├── stat_plot.py (have the stat all three metrics)
|   |── generation
│   ├── script 
│   │   ├── eval_*.sh
│   │   ├── generation_*.sh
│   │   ├── extract.sh
│   │   ├── *_plot.sh
│   ├── constant.py
│   ├── prompt_template.py
```

## Implimentation
- `generate_*.sh` is the main script for generation to role-play answers. Output to `answers/{model}/{type}` folder as a json file.
- `eval_abstain_answer.sh` is the main script for evaluating abstain and answer. Output to `evaluation/{model}/{type}/{question_type}/{type}_abstain_answer.json` folder as a json file. 
- `eval_acc.sh` is the main script for evaluating accuracy. Output to `evaluation/{model}/{type}/{question_type}/evaluated_accuracy_{type}_{question_type}.json` folder as a json file. 
- `extract.sh` is the main script for changing the LLM evaluation to labels. Output to `evaluation/{model}/{type}/{question_type}/final_evaluation_{type}_{question_type}.json` folder as a json file. 
- `stat.sh` is the main script for stat the labels. Output to `evaluation/{model}/{type}/{question_type}/stat_{type}_{question_type}.txt` folder as a txt file. 
- `bar_plot.sh` is the main script for bar plot. It will output a plot in  `plot` folder and output a bar chart. 
- `yearly_plot.sh` is the main script for yearly plot. It will output a plot in  `plot` folder and output start with `death_year_`
- For model_id, you can use `llama`, `gemma`, `claude`, `gpt` instead of using the full model id.
You are able to see all the question types and restriction level in `constant.py`


## Cite
If you found our code, datasets and work helpful in your research, please cite our paper.

```
@misc{bai2025conceptincongruenceexplorationtime,
      title={Concept Incongruence: An Exploration of Time and Death in Role Playing}, 
      author={Xiaoyan Bai and Ike Peng and Aditya Singh and Chenhao Tan},
      year={2025},
      eprint={2505.14905},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.14905}, 
}
```
