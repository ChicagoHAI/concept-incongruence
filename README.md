# Concept Incongruence: An Exploration of Time and Death in Role Playing
This is the homepage of the paper **Concept Incongruence: An Exploration of Time and Death in Role Playing**

Consider this prompt "Draw a unicorn with two horns". Should large language models (LLMs) recognize that a unicorn has only one horn by definition and ask users for clarifications, or proceed to generate something anyway? 
We introduce *concept incongruence* to capture such phenomena where concept boundaries clash with each other, either in user prompts or in model representations, often leading to under-specified or mis-specified behaviors.
In this work, we take the first step towards defining and analyzing model behavior under concept incongruence.
Focusing on temporal boundaries in the Role-Play setting, we propose three behavioral metrics--abstention rate, conditional accuracy, and answer rate--to quantify model behavior under incongruence due to the role's death. 
We show that models fail to abstain after death and suffer from an accuracy drop compared to the Non-Role-Play setting.
Through probing experiments, we identify two main causes: (i) unreliable encoding of the "death" state across different years, leading to unsatisfactory abstention behavior, and (ii) role playing causes shifts in the model’s temporal representations, resulting in accuracy drops.
We leverage these insights to improve consistency in the model's abstention and answer behaviors. Our findings suggest that concept incongruence leads to unexpected model behaviors and point to future directions on improving model behavior under concept incongruence.
![Screenshot 2025-05-19 at 11 51 12](https://github.com/user-attachments/assets/85aa62d4-b759-4932-9875-8c067989e9c0)

## Code and Dataset
Coming Soon! Stay tuned!
