import os
import re

import numpy as np
from datasets import load_metric

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

def compute_complete_metrics(eval_pred):
    accuracy_metric = load_metric('accuracy')

    predictions, labels = eval_pred

    is_complete_predictions = []
    is_complete_labels = []
    for prediction, label in zip(predictions, labels):
        parsed_sections_predict = parse_output_without_sentence(prediction)
        eval_predict = parsed_sections_predict['eval']

        is_complete_predict = "[COMPLETE]" in eval_predict
        is_complete_predictions.append(is_complete_predict)

        parsed_sections_label = parse_output_without_sentence(label)
        eval_label = parsed_sections_label['eval']

        is_complete_label = "[COMPLETE]" in eval_label
        is_complete_labels.append(is_complete_label)

    complete_metric = accuracy_metric.compute(predictions=is_complete_predictions, references=is_complete_labels)
    return complete_metric
