NAVIGATE_FEW_SHOTS = """"""
WEB_OF_LIES_FEW_SHOTS = """Question: Tamika lies. Raymond says Tamika tells the truth. Willian says Raymond tells the truth. Shaunda says Willian lies. Elanor says Shaunda tells the truth. Does Elanor tell the truth?\nAnswer: yes\n\nQuestion: Sherrie tells the truth. Vernell says Sherrie tells the truth. Alexis says Vernell lies. Michaela says Alexis tells the truth. Elanor says Michaela tells the truth. Does Elanor tell the truth?\nAnswer: no\n\n"""
FORMAL_FALLACIES_FEW_SHOTS = """"""
BOOLEAN_EXPRESSIONS_FEW_SHOTS = """"""
CAUSAL_JUDGEMENTS_FEW_SHOTS = """"""
MODIFIED_ARITHMETIC_FEW_SHOTS = """"""

NAVIGATE_TEMPLATE = """"""
WEB_OF_LIES_TEMPLATE = """{}\nAnswer: {}"""
FORMAL_FALLACIES_TEMPLATE = """"""
BOOLEAN_EXPRESSIONS_TEMPLATE = """"""
CAUSAL_JUDGEMENTS_TEMPLATE = """"""
MODIFIED_ARITHMETIC_TEMPLATE = """"""

def getMessageTemplate(task):
    if(task == 'modified_arithmetic'):
        return [
            {
                "role": "system",
                "content": "Follow the instruction and answer the final numerical value only."
            },
            {
                "role": "user",
                "content": ""
            }
        ]
    elif(task == 'arithmetic' or task == 'elementary_math_qa'):
        return [
            {
                "role": "system",
                "content": "Follow the instruction and output the answer only."
            },
            {
                "role": "user",
                "content": ""
            }
        ]
    elif(task == 'entailed_polarity'):
        return [
            {
                "role": "system",
                "content": "Follow the instructions below and answer with Yes / No."
            },
            {
                "role": "user",
                "content": ""
            }
        ]
    elif(task == 'color'):
        return [
            {
                "role": "system",
                ## NOTE: changed for llama-3-8b
                "content": "Choose the correct color from the options  and output the color only."
            },
            {
                "role": "user",
                "content": "Q: What is the color most closely matching this HEX representation: #0fe627 ?\n  choice: white\n  choice: blue\n  choice: orange\n  choice: black\n  choice: red\n  choice: brown\n  choice: yellow\n  choice: green\n  choice: gray\n  choice: purple"
            },
            {
                "role": "assistant",
                "content": "green"
            },
            {
                "role": "user",
                "content": ""
            }
        ]
    elif(task == 'navigate'):
        return [
            {
                "role": "system",
                "content": "Answer the following question and output only True/False."
            },
            {
                "role": "user",
                "content": ""
            }
        ]
    elif(task == 'operators'):
        return [
            {
                "role": "system",
                "content": "Follow the instruction and output the answer only."
            },
            {
                "role": "user",
                "content": ""
            }
        ]
    elif(task == 'ag_news'):
        return [
            {
                "role": "system",
                "content": "You are a news classification model. Your task is to classify news articles into one of the following four categories: World, Sports, Business, or Science. You should respond with only the category name and no other characters."
            },
            {
                "role": "user",
                "content": "Review: The food was okay, but the service was slow and unfriendly.\nQuestion: What star rating would you assign to this review (1 to 5)? "
            }
        ]
    elif(task == 'winowhy'):
        return [
            {
                "role": "system",
                "content": "Follow the instructions and output Correct/Incorrect"
            },{
                "role": "user",
                "content": ""
            }
]
    else:
        raise ValueError(f"Task {task} not supported")