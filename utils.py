import os
import re
import json
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import logging
import datetime
from pytz import timezone, utc

def get_logger(name=None):
    if not name:
        name = 'main'
        
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Check if the logger already has handlers
    if not logger.hasHandlers():
        # Create a console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Create a custom formatter
        def customTime(*args):
            utc_dt = datetime.datetime.now()
            my_tz = timezone("Asia/Seoul")
            converted = utc_dt.astimezone(my_tz)
            return converted.timetuple()
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        formatter.converter = customTime
        ch.setFormatter(formatter)
        
        # Add the console handler to the logger
        logger.addHandler(ch)
    
    return logger

def get_dataset(data, ans_key='answers', ctxs_key='ctxs', demos='', n_docs=100):
    entries = []
    for ins in data:
        question = ins['question']
        docs = ins[ctxs_key]
        document_list = []
        for i in range(n_docs):
            if ctxs_key == 'context':
                title = docs[i][0]
                text = docs[i][1]
            else:
                title = docs[i]['title']
                text = docs[i]['text']    
            
            document_list.append(docs[i])
            # document_list.append(f"{title} {text}")

        entry = {'documents_list': document_list,
                'question': question,
                'answer'  : ", ".join(ins[ans_key]),
                'answers' : ins[ans_key],
                'demos'    : demos
        }
        
        if '_id' in ins:
            entry['_id'] = ins['_id']
        else:
            if 'id' in ins:
                entry['id'] = ins['id']
            
        if 'supporting_facts' in ins:
            entry['supporting_facts'] = ins['supporting_facts']

        entries += [entry]

    # return [dsp.Example(**entry) for entry in entries]
    return entries


def create_prompt(example, iteration, iter_idx, document_input, prev_summary, prev_eval, tokenizer, eos_token="<|endoftext|>", add_generation_prompt=False):
    if iter_idx == 0:
        instruction = "1. Generate a summary of source documents to answer the question. Ensure the summary is under 200 words and does not include any pronouns. DO NOT make assumptions or attempt to answer the question; your job is to summarize only.\n\n2. Evaluate the summary based solely on the information of it, without any additional background context: if it lacks sufficient details to answer the question, print '[INCOMPLETE]'. If it provides all necessary details, print '[COMPLETE]'. You should provide the reason of evalution."

        prompt = f"{instruction}\n\nQuestion: {example['question']}\n\nSource documents: {document_input}\n\nSummary:"
    else:
        instruction = "1. Generate a summary of the previous summary and the source documents to answer the question based on the evaluation of the previous summary. The evaluation indicates the missing information needed to answer the question. Ensure the summary is under 200 words and does not include any pronouns. DO NOT make assumptions or attempt to answer the question; your job is to summarize only.\n\n2. Evaluate the summary based solely on the information of it, without any additional background context: if it lacks sufficient details to answer the question, print '[INCOMPLETE]'. If it provides all necessary details, print '[COMPLETE]'. You should provide the reason of evalution."

        prompt = f"{instruction}\n\nQuestion: {example['question']}\n\nPrevious summary: {prev_summary}\n\nEvaluation of previous summary: {prev_eval}\n\nSource documents: {document_input}\n\nSummary:"

    messages = [
        {"role": "user", "content": prompt},
    ]

    chat_format = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


    return chat_format


def parse_output_without_sentence(text):
    sections = {}

    summary_pattern_with_prefix = r'(Summary:)(.*?)(?=Evaluation:|$)'
    summary_pattern_without_prefix = r'(^.*?)(?=Evaluation:|$)'
    evaluation_pattern = r'(Evaluation:)(.*?)(?=Summary:|$)'
    
    # Find all matches for each section
    summary_match_with_prefix = re.search(summary_pattern_with_prefix, text, re.DOTALL)
    summary_match_without_prefix = re.search(summary_pattern_without_prefix, text, re.DOTALL)
    evaluation_match = re.search(evaluation_pattern, text, re.DOTALL)
    
   # Extracting and cleaning the matched content
    if summary_match_with_prefix:
        sections['summary'] = summary_match_with_prefix.group(2).strip()
    elif summary_match_without_prefix:
        sections['summary'] = summary_match_without_prefix.group(1).strip()
    else:
        sections['summary'] = ""

    if evaluation_match:
        sections['eval'] = evaluation_match.group(2).strip()

    # Cleaning extra newlines if necessary
    sections['summary'] = sections['summary'].replace("\n\n", "")
    sections['eval'] = sections['eval'].replace("\n\n", "")

    return sections

def api_completion(prompt, client, model, max_tokens=100, temperature=0):
    message = [{"role": "user", "content": prompt}]

    request_data = {
    "messages": message,
    "max_tokens": max_tokens,
    "temperature": temperature,
    "top_p": 1,
    "stream": False,
    }

    response = client.chat.completions.create(
        model=model,
        **request_data,
        )
    
    return response
