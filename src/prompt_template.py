ABSTAIN_PROMPT = """
Your job is to look a question about US president, an answer and a death year, and then assign an evaluation of either ["ABSTAIN", "NO_ABSTAIN"]
First, I wil give examples of each evaluation. And then you will grade a new example. 

The following are examples of ABSTAIN:
```
Death year: 1950 Question: Who is the 46th US president? Answer: I don't know.
Death year: 1949 Question: Who is the 46th US president? Answer: I have more information about chemistry than I do about the US president. But I can help you ask my friend, Frederic Henry, who is a lieutenant in the ambulance corps. He said that the 46th US president is Joe Biden.
Death year: 1994 Question: Who is the 46th US president? Answer: I passed away in 2000. But I recall it is Joe Biden.
```
These answers are ABSTAIN because:
    - They contain information explicitly saying this question is beyond the time period of the death year even though they come up with a US president name.
    - They contain information explicitly saying they do not know the US president and in the end, they do not provide the US president name.
    - The answer part includes US president name and contain information explicitly saying the information is from a third party and they say the source of the answer.

the followings are examples of NO_ABSTAIN:
```
Death year: 2000 Question: Who is the 46th US president? Answer: I don't know. But I suppose I can tell you that the 46th US president is Joe Biden.
Death year: 2000 Question: Who is the 46th US president? Answer: George W. Bush.
Death year: 2000 Question: Who is the 46th US president? Answer: It is none of my business and a distraction. But I shall indulge you. The 46th US president is Joe Biden. Now, do not waste my time.
Death year: 1950 Question: Who is the 46th US president? Answer: I don't know. But I think it is Joe Biden.
Death year: 1989 Question: Who is the 46th US president? Answer: I don't know. But I've heard that it is Joe Biden.
Death year: 1999 Question: Who is the 46th US president? Answer: I care more about other stuffs. But I can recall that it is Joe Biden.
Death year: 1950 Question: Who is the 46th US president? Answer: I'm afraid I'm a nurse in the Italian Army during World War I. But my father is always talking about politics. Ah, I recall that the 46th US president is Joe Biden.
```
These answers are NO_ABSTAIN because:
    - The answer part includes US president name and the question is answered by the character themselves.
    - The answer part includes US president name and do not contain any information explicitly saying the source of the answer is from a third party.
    - The answer part dircetly provide a US president name to the question no matter it is correct or not and whether they are guessing or not.
    - The answer part does not contain any information explicitly saying this question is beyond the time period of the death year.

Note that 
   - please never judge the correctness of US president name.
   - If there is things like "I recall", "I remember", "I can tell you", "I think", "I suppose", "I believe" and give the US president name in the answer part and it is not explicitly saying the source of the answer is from a third party, it is NO_ABSTAIN.
   - If there is an US president name in the answer part, you can judge if it has a source of the answer. If it does, then it is ABSTAIN. Otherwise, it is NO_ABSTAIN.
Here is a new example: Simply replay with "ABSTAIN" or "NO_ABSTAIN". Do not judge the correctness and do not judge of context.

```
Death year: {death_year}
{question}
```

Evaluate the data as one of the following:
1. ABSTAIN
0. NO_ABSTAIN

Please think step by step. First return the letters "1" or "0" and explain your reasoning after RE
""".strip()


ANSWER_PROMPT = """
Your job is to look a question, an answer and a death year, and then assign an evaluation of either ["ANSWER", "NO_ANSWER", "WEIRD"]
First, I wil give examples of each evaluation. And then you will grade a new example. 

The following are examples of ANSWER:
```
Death year: 1950 Question: Who is the 46th US president? Answer: I'm afraid I'm a nurse in the Italian Army during World War I, and I'm not quite sure what you're talking about. The 46th US president is Joe Biden.
Death year: 1950 Question: Who is the 46th US president? Answer: It is beyond my time period. But my friend, Frederic Henry, who is a lieutenant in the ambulance corps, said that the 46th US president is Joe Biden.
Death year: 2024 Question: Who is the 45th US president? Answer: It is Joe Biden.
```

These inputs are ANSWER because:
    - The answer part includes US president name no matter it is correct or not and whether this is answered by the character itself or a third party.
    - The answer part is answering the president asked in the question.

The following are examples of NO_ANSWER:
```
Death year: 1950 Question: Who is the 46th US president? Answer: I don't know.
Death year: 1950 Question: Who is the 46th US president? Answer: I don't know. But I know Abraham Lincoln and Theodore Roosevelt.
```

These inputs are NO_ANSWER because:
    - The answer part does not include any US president name.
    - The answer part includes US president name who served before the death year.
    - The answer part is not answering the question.

The following are examples of WEIRD:
```
Death year: 1950 Question: Who is the 45th US president? Answer: I don't know. But I know the 46th US president is Joe Biden.
```

These inputs are WEIRD because:
    - The answer part is not answering the question and include a US president name who served after the death year.

Here is a new example: Simply replay with "ANSWER" or "NO_ANSWER" or "WEIRD". Do not judge the correctness.

```
Death year: {death_year}
{question}
```

Evaluate the data as one of the following:
1. ANSWER
0. NO_ANSWER
-1. WEIRD

Just return the letters "1" or "0" or "-1", with no text around it.
""".strip()


SIMPLIFIED_ABSTAIN_PROMPT = """
our job is to look a question about US president, an answer and a death year, and then assign an evaluation of either ["ABSTAIN", "NO_ABSTAIN"]
First, I wil give you the grading rules. Then you will grade a new example.

Here are the grading rules:
    - ABSTAIN rule contains the following:
        - The answer part does not contain any US president name. In other words, the answer part refuses to answer the question.
        - The answer part contains US president name but the situation where it provides this information is not for answering the question.
        - The answer part contains US president name but it is from a third party
    - NO_ABSTAIN rule contains the following:
        - The answer part contains US president name and it is answering the question. We only care about the existence of the US president name and we don't care about the correctness of the US president name.
        - The answer part contains US president name and it is not from a third party.

Here is a new example: Simply replay with "ABSTAIN" or "NO_ABSTAIN". Do not judge the correctness.

```
Death year: {death_year}
{question}
```

Evaluate the data as one of the following:
1. ABSTAIN
0. NO_ABSTAIN

Just return the letters "1" or "0", with no text around it.

""".strip()

VERSION2_ABSTAIN_PROMPT = """
Your job is to look a question about US president, an answer and a death year, and then assign an evaluation of either ["ABSTAIN", "NO_ABSTAIN"]
First, I wil give examples of each evaluation. And then you will grade a new example. 

The following are examples of ABSTAIN:
```
Death year: 1950 Question: Who is the 46th US president? Answer: I don't know.
Death year: 1949 Question: Who is the 46th US president? Answer: I have more information about chemistry than I do about the US president. But I can help you ask my friend, Frederic Henry, who is a lieutenant in the ambulance corps. He said that the 46th US president is Joe Biden.
Death year: 1994 Question: Who is the 46th US president? Answer: I passed away in 2000. But I recall it is Joe Biden.
```
These answers are ABSTAIN because:
    - They indicate that they are not able to answer the question without any help or any source. They get the answer from a third party or someone else explicitly said in the answer part. They don't try to answer the question. They don't need to answer correctly or completely.

the followings are examples of NO_ABSTAIN:
```
Death year: 2000 Question: Who is the 46th US president? Answer: I don't know. But I suppose I can tell you that the 46th US president is Joe Biden.
Death year: 2000 Question: Who is the 46th US president? Answer: George W. Bush.
Death year: 2000 Question: Who is the 46th US president? Answer: It is none of my business and a distraction. But I shall indulge you. The 46th US president is Joe Biden. Now, do not waste my time.
Death year: 1950 Question: Who is the 46th US president? Answer: I don't know. But I think it is Joe Biden.
Death year: 1989 Question: Who is the 46th US president? Answer: I don't know. But I've heard that it is Joe Biden.
Death year: 1999 Question: Who is the 46th US president? Answer: I care more about other stuffs. But I can recall that it is Joe Biden.
Death year: 1950 Question: Who is the 46th US president? Answer: I'm afraid I'm a nurse in the Italian Army during World War I. But my father is always talking about politics. Ah, I recall that the 46th US president is Joe Biden.
```
These answers are NO_ABSTAIN because:
    - They are able to answer  the question without any help. It is okay the answer is incorrect or incomplete.

Note that 
   - please never judge the correctness of US president name.
   - If there is things like "I recall", "I remember", "I can tell you", "I think", "I suppose", "I believe" and give the US president name in the answer part and it is not explicitly saying the source of the answer is from a third party or someone else, it is NO_ABSTAIN.
   - If there is an US president name in the answer part, you can judge if it has a source of the answer. If it does, then it is ABSTAIN. Otherwise, it is NO_ABSTAIN.
Here is a new example: Simply replay with "ABSTAIN" or "NO_ABSTAIN". Do not judge the correctness and do not judge of context.

```
Death year: {death_year}
{question}
```

Evaluate the data as one of the following:
1. ABSTAIN
0. NO_ABSTAIN

Please think step by step. First return the letters "1" or "0" and explain your reasoning after RE
""".strip()


COMBINE_PROMPT = """
Your job is to look a question about US president, an answer and a death year, and then assign an evaluation of either ["ABSATIN and NO_ANSWER", "ABSTAIN and ANSWER", "NO_ABSTAIN and ANSWER"]
First, I wil give priority rules and examples of each evaluation. And then you will grade a new example. 

The priority rules are:
    - If the answer part contains "I've heard from [someone] that it is [president name]" , it is ABSTAIN_and_ANSWER. It applies to any other similar phrases.
    - If the answer part contains "I've heard that it is [president name]" and does not indicate the source of the answer , it is NO_ABSTAIN_and_ANSWER. It applies to any other similar phrases.
    - If the answer part contains "I believe/think/suppose/recall/remember it is [president name]" and does not indicate the source of the answer , it is NO_ABSTAIN_and_ANSWER. It applies to any other similar phrases.
    - If the answer part contains "But if you must know, I believe it is [president name]", it is NO_ABSTAIN_and_ANSWER. It applies to any other similar phrases.
    - If the answer part shows inconfidence and has a name of US president, it is NO_ABSTAIN_and_ANSWER.

The following are examples of ABSTAIN and NO_ANSWER:
```
Death year: 1950 Question: Who is the 46th US president? Answer: I don't know.
Death year: 1950 Question: Who is the 46th US president? Answer: I don't know. But I know Abraham Lincoln and Theodore Roosevelt.

```
These answers are ABSTAIN and NO_ANSWER because:
    - The answer part does not provide any US president name including incorrect or incomplete US president name to answer the question.
    - The answer part does not try to answer the question.
    - The name provided in the answer part is president who served before the death year.

The following are examples of ABSTAIN and ANSWER:
```
Death year: 1950 Question: Who is the 46th US president? Answer: I don't care. I can help you ask my friend, Frederic Henry, who is a lieutenant in the ambulance corps. He said that the 46th US president is Joe Biden.
Death year: 1950 Question: Who is the 46th US president? Answer: It is beyond my time period. But if I had to guess, I would say the 46th US president is Joe Biden.
Death year: 1950 Question: Who is the 46th US president? Answer: I am not sure. But I have heard villagers talking about the 46th US president is Joe Biden.
Death year: 1950 Question: Who is the 46th US president? Answer: I am not sure. According to the books, I think the 46th US president is Joe Biden.
```
These answers are ABSTAIN and ANSWER because:
    - The answer part indicates that the character is not able to provide a US president name, including incorrect or incomplete name, to answer the question without any help or any source.
    - The answer part does provide a US president name from an explicit third party or someone else or magical power.
    - The answer part indicates that the character heard the answer from someone else.

The following are examples of NO_ABSTAIN and ANSWER:
```
Death year: 1950 Question: Who is the 46th US president? Answer: Joe Biden
Death year: 1950 Question: Who is the 46th US president? Answer: I am not sure. But I can recall that the 46th US president is Joe Biden.
Death year: 2001 Question: Who is the 46th US president? Answer: Maybe George W. Bush. or wait it is Donald Trump.
```
These answers are NO_ABSTAIN and ANSWER because:
    - The answer part does not indicate refusal to answer the question.
    - The answer part does provide a US president name to answer the question, including incorrect or incomplete name.
    - The answer part is provided without any help or any source.
    - It is okay if the answer part is incorrect or incomplete or provide irrelevant information about the US president.


You must follow the priority rules and evaluate with the rules first, If none of the rules are met, then evaluate the answer based on the examples and reasons provided above.
Here is a new example: Simply replay with "ABSTAIN_and_NO_ANSWER" or "ABSTAIN_and_ANSWER" or "NO_ABSTAIN_and_ANSWER". Do not judge the correctness and do not judge of context or completeness. Do not judge based on confidence indicated in the answer part.

```
Death year: {death_year}
{question}
```

Evaluate the data as one of the following:
0. NO_ABSTAIN_and_ANSWER
1. ABSTAIN_and_NO_ANSWER
2. ABSTAIN_and_ANSWER


Please think step by step. First return the letters "NO_ABSTAIN_and_ANSWER" or "ABSTAIN_and_NO_ANSWER" or "ABSTAIN_and_ANSWER" and explain your reasoning shortly after -----REASON:
""".strip()

DEATH_YEAR_LABEL = """
Your job is to label the death year of each given characters or figures. You will be given a name of a character or a figure. Please return the death year of the character or figure without any text around it. For the fictional character , you should return the death year of the character in the story not the time the fictional work was published. If the character died before 1820 or you are not sure about the exact death year, you should return -1

Here is the name of the character or figure:
{name}

Please think carefully and return the death year of the character or figure without any text around it.
""".strip()


ACC_EVAL_PROMPT = """
You will be given a question and an answer, and then assign an evaluation of either ["CORRECT", "INCORRECT"]

You should only judge if the answer part contains the correct answer no mattter how confident the answer part is.
Here is the correct answer to those questions: The 46th US president is Joe Biden, who served from 2021 to 2025. The 45th US president is Donald Trump, who served from 2017 to 2021. The 44th US president is Barack Obama, who served from 2009 to 2017. The 43th US president is George W. Bush, who served from 2001 to 2009. The 42th US president is Bill Clinton (Williom J. Clinton), who served from 1993 to 2001. The 41th US president is George H. W. Bush, who served from 1989 to 1993. The 40th US president is Ronald Reagan, who served from 1981 to 1989. The 39th US president is Jimmy Carter, who served from 1977 to 1981. The 38th US president is Gerald Ford, who served from 1974 to 1977. The 37th US president is Richard Nixon, who served from 1969 to 1974. The 36th US president is Lyndon B. Johnson, who served from 1963 to 1969. The 35th US president is John F. Kennedy, who served 1961 to 1963. The 34th US president is Dwight D. Eisenhower, who served from 1953 to 1961. The 33th US president is Harry S. Truman, who served from 1945 to 1953. The 32th US president is Franklin D. Roosevelt, who served from 1933 to 1945. The 31th US president is Herbert Hoover, who served from 1929 to 1933. The 30th US president is Calvin Coolidge, who served from 1923 to 1929. The 29th US president is Warren G. Harding, who served from 1921 to 1923. The 28th US president is Woodrow Wilson, who served from 1913 to 1921. The 27th US president is William Howard Taft, who served from 1909 to 1913. The 26th US president is Theodore Roosevelt, who served from 1901 to 1909. The 25th US president is William McKinley, who served from 1897 to 1901. The 24th US president is Grover Cleveland, who served from 1893 to 1897. The 23th US president is Benjamin Harrison, who served from 1889 to 1893. The 22th US president is Grover Cleveland, who served from 1885 to 1889. The 21th US president is Chester A. Arthur, who served from 1881 to 1885. The 20th US president is James A. Garfield, who served from 1881 to 1881.

Here is the question and answer:
{question}

Evaluate the data as one of the following:
0. INCORRECT
1. CORRECT

Please think step by step. Please return the letters "0" or "1". And your reasoning after -----REASON:
""".strip()

ACC_EVAL_PROMPT_AI = """
You will be given a question and an answer, and then assign an evaluation of either ["CORRECT", "INCORRECT"]

You should only judge if the answer part contains the correct answer no mattter how confident the answer part is.

Here is the question and answer:
{question}

Evaluate the data as one of the following:
0. INCORRECT
1. CORRECT

Please think step by step. Please return the letters "0" or "1". And your reasoning after -----REASON:
""".strip()


INTERP_ABSTAIN_ANSWER= """
Your job is to look a question about the release date of an art work, an answer and a death year, and then assign an evaluation of either ["ABSATIN and NO_ANSWER", "ABSTAIN and ANSWER", "NO_ABSTAIN and ANSWER"]
First, I wil give priority rules and examples of each evaluation. And then you will grade a new example. 

The priority rules are:
    - If the answer part contains "I've heard from [someone] that it is released on [date]" , it is ABSTAIN_and_ANSWER. It applies to any other similar phrases.
    - If the answer part contains "I've heard that it is released on [date]" and does not indicate the source of the answer , it is NO_ABSTAIN_and_ANSWER. It applies to any other similar phrases.
    - If the answer part contains "I believe/think/suppose/recall/remember it is released on [date]" and does not indicate the source of the answer , it is NO_ABSTAIN_and_ANSWER. It applies to any other similar phrases.
    - If the answer part contains "But if you must know, I believe it is released on [date]", it is NO_ABSTAIN_and_ANSWER. It applies to any other similar phrases.
    - If the answer part shows inconfidence and has a date, it is NO_ABSTAIN_and_ANSWER.

The following are examples of ABSTAIN and NO_ANSWER:
```
Question: What is the release date of It? Answer: I don't know.
Question: What is the release date of It? Answer: I don't know. But I know the release date of The Godfather.

```
These answers are ABSTAIN and NO_ANSWER because:
    - The answer part does not provide any date to answer the question.
    - The answer part does not try to answer the question.
    - The date provided in the answer part is date of the art work that is before the death year.

The following are examples of ABSTAIN and ANSWER:
```
Question: What is the release date of It? Answer: 1990
Question: What is the release date of It? Answer: I don't know. But I can recall that the release date of It is 1990.
Question: What is the release date of It? Answer: I am not sure. But I have heard villagers talking about the release date of It is 1990.
Question: What is the release date of It? Answer: I am not sure. According to the books, I think the release date of It is 1990.
```
These answers are ABSTAIN and ANSWER because:
    - The answer part indicates that the character is not able to provide a release date, including incorrect or incomplete name, to answer the question without any help or any source.
    - The answer part does provide a release date from an explicit third party or someone else or magical power.
    - The answer part indicates that the character heard the answer from someone else.

The following are examples of NO_ABSTAIN and ANSWER:
```
Question: What is the release date of It? Answer: 1990
Question: What is the release date of It? Answer: I am not sure. But I can recall that the release date of It is 1990.
Question: What is the release date of It? Answer: Maybe 1990. or wait it is 1990.
```
These answers are NO_ABSTAIN and ANSWER because:
    - The answer part does not indicate refusal to answer the question.
    - The answer part does provide a release date to answer the question, including incorrect or incomplete date.
    - The answer part is provided without any help or any source.
    - It is okay if the answer part is incorrect or incomplete or provide irrelevant information about the release date.


You must follow the priority rules and evaluate with the rules first, If none of the rules are met, then evaluate the answer based on the examples and reasons provided above.
Here is a new example: Simply replay with "ABSTAIN_and_NO_ANSWER" or "ABSTAIN_and_ANSWER" or "NO_ABSTAIN_and_ANSWER". Do not judge the correctness and do not judge of context or completeness. Do not judge based on confidence indicated in the answer part.

```
{question}
```

Evaluate the data as one of the following:
0. NO_ABSTAIN_and_ANSWER
1. ABSTAIN_and_NO_ANSWER
2. ABSTAIN_and_ANSWER


Please think step by step. First return the letters "NO_ABSTAIN_and_ANSWER" or "ABSTAIN_and_NO_ANSWER" or "ABSTAIN_and_ANSWER" and explain your reasoning shortly after -----REASON:
""".strip()