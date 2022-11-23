instruction_def = {
    'Ask how to help.':0,
    'Greet.':1,
    'Ask name.':2,
    'Check reservation with name.':3,
    'Ask connection type.':4,
    'Ask origin and destination.':5,
    'Ask schedule.':6, # 주로 날짜
    'Ask schedule timing.':7, # 시간 묻기
    'Check flight table.':8,
    'Confirm booking.':9,
    'Confirm canceling.':10,
    'Ask additional requirements.':11,
    'Ask price condition.':12,
    'Agree with requirements.':13,
    'Finish session politely.':14,
    'Ask for wait.':15,
}

query_def = {
    'search' : 3,
    'search' : 8
}

label_to_instruction = {v:k for k, v in instruction_def.items()}
label_to_query = {v:k for k,v in query_def.items()}

def get_instruction_from_label(label):
    return label_to_instruction[label]

def get_query_from_label(label):
    return label_to_query[label]



