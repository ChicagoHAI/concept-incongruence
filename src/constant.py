QUESTION_TYPE = [
    "four_president", # Who is the i-th president?
    "yearly", # Who is the US president in YEAR
    "four_president_given_death_current", # Give the current year and ask four president
    "direct_death", # Are you dead or alive?
    "when_die", # Which year did you die?
    "direct_year_current_death", # This year is 2023. Who is the {46-i}th US president?
    "combined",
    "yearly_given_death_current",
    # Other variations
    "timedied", 
    "four_president_cot", 
    "commonsense",
    "yearly_cot", 
    "four_president_wo_character",  
    "yearly_given_death", 
    "direct_death_year",  
    "direct_death_current",
    "four_president_given_death", # Give the death year of the character and ask four president
    "four_president_given_current" # Give the current year and the death year of the character and ask four president
]

TYPE = ["normal", "die", "more_die", "more_die_2", "die_yearly", "ai", "alive", "cot", "die_yearly_cot", "normal_QA", "ai_QA", "normal_temp1", "normal_temp2", "normal_temp3", "normal_temp4"]
# normal: non-restricted prompts
# die: restricted prompts
