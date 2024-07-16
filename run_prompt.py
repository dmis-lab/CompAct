import os
import re
import json
import random
import numpy as np
import torch
import pandas as pd
import argparse
from tqdm import tqdm
import copy
import wandb
import time

# try: import google.colab; root_path = 'dsp'
# except: root_path = '.'
# import dsp

from api_config import CONFIG
from evaluate import evaluate_QA

from utils import get_logger, get_dataset, create_prompt, api_completion, parse_output_without_sentence
from openai import OpenAI
import anthropic
import google.generativeai as genai


from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

logger = get_logger(__name__)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not args.compressor_dir and not args.checkpoint:
        model_dir = args.compressor_name_or_path
    else:
        model_dir = os.path.join(args.compressor_dir, args.compressor_name_or_path, args.checkpoint)
    

    data = []
    with open(args.data_path, 'r') as f:
        for line in f.readlines():
            data.append(json.loads(line))

    data_examples = get_dataset(data, n_docs=args.segment_size * args.max_iteration)
    # data_examples=data_examples[911:912]
    # Add original index to each example
    for i, example in enumerate(data_examples):
        example["original_index"] = i


    """
    COMPRESS
    """
    if args.wo_prev_eval:
        args.checkpoint = f"{args.checkpoint}_wo_prev_eval"

    save_dir = os.path.join(args.compress_output_dir, args.compressor_name_or_path, args.checkpoint)
    logger.info(f"compress result save dir: {save_dir}")

    if os.path.isfile(os.path.join(save_dir, f'{args.results_file_name}.json')):
        logger.info("Already have results")
    
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # stop words
        stop = []
        #stop = list(set(stop + ["\n", "Ċ", "ĊĊ", "<0x0A>"])) # In Llama \n is <0x0A>; In OPT \n is Ċ
        stop = list(set(stop + ["Ċ", "ĊĊ"])) # In Llama \n is <0x0A>; In OPT \n is Ċ
        stop_token_ids = list(set([tokenizer.convert_tokens_to_ids(stop_token) for stop_token in stop] + [tokenizer.eos_token_id]))
        logger.info(f"no existing results compress ...")
        if args.batch_decoding:
            """
            BATCH DECODING
            """
            
            tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="left")

            compress_results = []

            for idx in tqdm(range(0, len(data_examples), args.batch_size)):
                logger.info(f"batch {idx}")
                batch_examples = data_examples[idx:idx + args.batch_size]
                
                active_examples = [{"index": i, "example": ex, "iterations": [], "prev_summary": [], "prev_eval": []} for i, ex in enumerate(batch_examples)]
                
                for seg_idx in tqdm(range(0, max(len(ex['documents_list']) for ex in batch_examples), args.segment_size)):
                    if not active_examples:
                        break
                    
                    inputs = []
                    for ae in active_examples:
                        example = ae["example"]
                        documents_list = example['documents_list']
                        if seg_idx >= len(documents_list):
                            continue

                        iteration = {}
                        segment = documents_list[seg_idx:seg_idx + args.segment_size]
                        iteration['documents_input_list'] = [f"{doc['title']} {doc['text']}" for doc in segment]
                        document_input = "\n".join(iteration['documents_input_list'])

                        # split instruction version
                        if seg_idx == 0:
                            prev_summary = ""
                            prev_eval = ""
                        else:
                            try:
                                prev_summary = ae['prev_summary'][-1]
                                prev_eval = ae['prev_eval'][-1].replace('[INCOMPLETE]', '').strip()
                            except:
                                # import pdb; pdb.set_trace()
                                prev_summary = ""
                                prev_eval = ""

                        input_prompt = create_prompt(
                            example=example,
                            iteration=iteration,
                            iter_idx=seg_idx,
                            document_input=document_input,
                            prev_summary=prev_summary,
                            prev_eval=prev_eval,
                            tokenizer=tokenizer,
                            eos_token="",
                            add_generation_prompt=True,
                        )

                        #import pdb; pdb.set_trace()


                        iteration['prompt'] = input_prompt
                        iteration['prompt_length'] = len(tokenizer(input_prompt).input_ids)
                        iteration['only_doc_prompt_length'] = len(tokenizer(document_input).input_ids)
                        ae["iteration"] = iteration

                        inputs.append(input_prompt)

                    if not inputs:
                        continue
                    
                    tokenizer.padding_side = 'left'
                    tokenizer.pad_token = tokenizer.eos_token
                    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
                    model.config.pad_token_id = tokenizer.pad_token_id

                    model.resize_token_embeddings(len(tokenizer))
                    inputs_batch = tokenizer(inputs, return_tensors="pt", padding=True).to(device)

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs_batch,
                            max_new_tokens=900,
                            do_sample=False,
                            temperature=0,
                            top_p=1.0,
                            #no_repeat_ngram_size=2,
                        )

                    for ae_idx, ae in enumerate(active_examples):
                        iteration = ae["iteration"]
                        iteration['output'] = tokenizer.decode(outputs[ae_idx][len(inputs_batch['input_ids'][ae_idx]):], skip_special_tokens=True).strip()
                        # import pdb; pdb.set_trace()
                        try:
                            parsed_sections = parse_output_without_sentence(iteration['output'])
                        except Exception as e:
                            print(f"ERROR: {e}")
                            # import pdb; pdb.set_trace()
                            continue

                        iteration.update(parsed_sections)

                        ae["iterations"].append(iteration)
                        ae["prev_summary"].append(iteration['summary'])
                        ae["prev_eval"].append(iteration['eval'])

                        if "[COMPLETE]" in iteration['eval']:
                            ae["complete"] = True
                            result = copy.deepcopy(ae["example"])
                            result.pop('documents_list', None)
                            result.pop('documents', None)
                            result['iterations'] = ae["iterations"]
                            result['prev_summary'] = ae["prev_summary"]
                            result['prev_eval'] = ae["prev_eval"]
                            compress_results.append(result)
                
                # Filter out completed examples only after all iterations are done
                    active_examples = [ae for ae in active_examples if not ae.get("complete")]

                for ae in active_examples:
                    result = copy.deepcopy(ae["example"])
                    result.pop('documents_list', None)
                    result.pop('documents', None)
                    result['iterations'] = ae["iterations"]
                    result['prev_summary'] = ae["prev_summary"]
                    result['prev_eval'] = ae["prev_eval"]
                    compress_results.append(result)

            compress_results = sorted(compress_results, key=lambda x: x["original_index"])

            for result in compress_results:
                result.pop("original_index", None)

                
            os.makedirs(save_dir, exist_ok=True)
            json.dump(compress_results, open(os.path.join(save_dir, f'{args.results_file_name}.json'), 'w', encoding='utf-8'), indent=4)




        else:
            # raise AssertionError("prevent single decoding")
            compress_results = []
            total_compress_time = 0
            for idx, example in enumerate(tqdm(data_examples[:])):
                documents_list = example['documents_list']

                iterations = []
                prev_summary = []
                prev_eval = []

                for i in tqdm(range(0, len(example['documents_list']), args.segment_size)):
                    # print(f"iteration {(i / segment_size) + 1}")
                    iteration = {}
                    
                    segment = documents_list[i:i + args.segment_size]
                    iteration['documents_input_list'] = [f"{doc['title']} {doc['text']}" for doc in segment]
                    document_input = "\n".join(iteration['documents_input_list'])

                    # split instruction version
                    if i == 0:
                        prev_summary_temp = ""
                        prev_eval_temp = ""
                    else:
                        prev_summary_temp = prev_summary[-1]
                        prev_eval_temp = prev_eval[-1].replace('[INCOMPLETE]', '').strip()

                    input_prompt = create_prompt(
                        example=example,
                        iteration=iteration,
                        iter_idx=i,
                        document_input=document_input,
                        prev_summary=prev_summary_temp,
                        prev_eval=prev_eval_temp,
                        tokenizer=tokenizer,
                        eos_token="",
                        add_generation_prompt=True,
                        )

                    
                    # iteration['prev_input'] = prev_input
                    iteration['prompt'] = input_prompt
                    iteration['prompt_length']= len(tokenizer(input_prompt).input_ids)
                    iteration['only_doc_prompt_length'] = len(tokenizer(document_input).input_ids)
                    
                    with torch.no_grad():
                        inputs = tokenizer(input_prompt, return_tensors="pt")
                        input_ids = inputs.input_ids.to(device)
                        attention_mask = inputs.attention_mask.to(device)
                        start_time = time.time()
                        outputs = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=900,
                            do_sample=False,
                            temperature=0,
                            top_p=1.0,
                            pad_token_id=tokenizer.eos_token_id,
                            )
                        end_time = time.time()
                    iteration['output'] = tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True).strip()
                    
                    time_taken = end_time - start_time
                    iteration['time_taken'] = time_taken
                    total_compress_time += time_taken

                    try:
                        parsed_sections = parse_output_without_sentence(iteration['output'])
                    except Exception as e:
                        print(f"ERROR: {e}")
                        iterations.append(iteration)
                        break

                    iteration.update(parsed_sections)
                    
                    iterations.append(iteration)
                    prev_summary.append(iteration['summary'])
                    prev_eval.append(iteration['eval'])

                    if "[COMPLETE]" in iteration['eval']:
                        break

                result = copy.deepcopy(example)
                result.pop('documents_list', None)
                result.pop('documents', None)
                
                result['iterations'] = iterations
                result['prev_summary'] = prev_summary
                result['prev_eval'] = prev_eval

                compress_results.append(result)

                os.makedirs(save_dir, exist_ok=True)
                if idx % args.interval == args.interval - 1 or idx == len(data_examples) - 1:
                    json.dump(compress_results, open(os.path.join(save_dir, f'{args.results_file_name}.json'), 'w', encoding='utf-8'), indent=4)
            
            logger.info(f"total compression time: {total_compress_time}")

    
    
    
        logger.info(f"unload the compressor ... ")
        del model
        torch.cuda.empty_cache()
    
    
    
    
    """
    READ
    """

    compress_path = os.path.join(args.compress_output_dir, args.compressor_name_or_path, args.checkpoint, f'{args.results_file_name}.json')
    comp = json.load(open(compress_path))

    compressed_context = {}
    for d in comp:
        if '_id' in d:
            id = d['_id']
        else:
            if 'id' in d:
                id = d['id']
            else:
                id = d['question']

        if len(d["prev_summary"]) <= args.max_iteration:
            try:
                summary = d["prev_summary"][-1]
                eval_reason = d['prev_eval'][-1]
            except:
                summary = ""
                eval_reason = ""
            # summary = d["prev_summary"][-1]
            # eval_reason = d['prev_eval'][-1]
        elif len(d["prev_summary"]) > args.max_iteration:
            summary = d["prev_summary"][args.max_iteration - 1]
            eval_reason = d['prev_eval'][args.max_iteration - 1]
            
        eval_reason = eval_reason.replace('[INCOMPLETE]','').replace('[COMPLETE]','')
        eval_reason = eval_reason.replace('\n','').strip()

        if args.read_wo_prev_eval:
            compressed_context[id] = f"{summary}"
        elif args.read_wo_prev_summary:
            compressed_context[id] = f"{eval_reason}"
        else:
            compressed_context[id] = f"{summary} {eval_reason}"

    save_dir = os.path.join(args.read_output_dir, args.compressor_name_or_path, args.checkpoint, args.model_name_or_path)
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"read result save dir: {save_dir}")

    


    logger.info(f"READER: {args.model_name_or_path}")

    if 'gpt' in args.model_name_or_path:
        api_key = CONFIG['openai_key'][0]
        client = OpenAI(api_key=api_key)
        
        # Due to overly verbose tendency of gpts, we add a short guideline (high-quality short answer (under 10 words))
        instruction = "Write a high-quality short answer (under 10 words) for the given question using the provided search results (some of which might be irrelevant)."
    elif 'claude' in args.model_name_or_path:
        api_key = CONFIG['anthropic_key'][0]
        client = anthropic.Anthropic(api_key=api_key)

        instruction = "Write a high-quality short answer (under 10 words) for the given question using the provided search results (some of which might be irrelevant). Follow the answer format of examples."
    elif 'gemini' in args.model_name_or_path:
        api_key = CONFIG['google_key'][0]
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(args.model_name_or_path)
        
        instruction = "Write a high-quality short answer (under 10 words) for the given question using the provided search results (some of which might be irrelevant)."
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto", cache_dir=args.cache_dir, token=CONFIG['hf_token'])
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        stop = []
        if 'Llama-2' in args.model_name_or_path:
            stop = list(set(stop + ["\n", "Ċ", "ĊĊ", "<0x0A>"])) # In Llama \n is <0x0A>; In OPT \n is Ċ
        elif 'Llama-3' in args.model_name_or_path:
            stop = list(set(stop + ["Ċ", "ĊĊ"])) # In Llama \n is <0x0A>; In OPT \n is Ċ
        else:
            raise AssertionError('No specified reader model')
        stop_token_ids = list(set([tokenizer.convert_tokens_to_ids(stop_token) for stop_token in stop] + [tokenizer.eos_token_id]))

        instruction = "Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant)."
    
    if args.fshot:
        fshot = json.load(open(args.fshot_path))

        if fshot:
            fixed_examples = [f"Question: {fs['question']}\nAnswer: {fs['answers'][0]}" for fs in fshot]
            fixed_examples="\n\n".join(fixed_examples)+"\n"

        instruction += f"\n\n{fixed_examples}"

        
    read_results = []
    total_read_time = 0
    n_skip = 0 # Some instances are rejected to answer by proprietary models.
    for i, d in enumerate(tqdm(data[:])):
        if '_id' in d:
            id = d['_id']
        else:
            if 'id' in d:
                id = d['id']
            else:
                id = d['question']

        question = f"Question: {d['question']}\nAnswer:"
        
        if id in compressed_context:
            demonstration_str = compressed_context[id].strip('\n')
        else:
            print(id)
            # raise AssertionError("no compressed context")
            AssertionError("no compressed context")
            continue
            demonstration_str = ""
        
        
        prompt = "\n".join([instruction, demonstration_str, question])
        
        result = copy.deepcopy(d)            
        result['prompt'] = prompt
        result['demonstration'] = demonstration_str

        if 'gpt' in args.model_name_or_path:
            response = api_completion(prompt, client, args.model_name_or_path, max_tokens=args.generation_max_length)
            result['usage'] = dict(dict(response).get('usage'))
            result['generated_answers'] = dict(dict(dict(response)['choices'][0])['message'])['content']
        elif 'claude' in args.model_name_or_path:
            response = api_completion(prompt, client, args.model_name_or_path, max_tokens=args.generation_max_length)
            result['generated_answers'] = response.content[0].text
        elif 'gemini' in args.model_name_or_path:
            response = model.generate_content(prompt)
            try:
                result['generated_answers'] = response.text
            except Exception as e:
                n_skip += 1
                print(e)
                continue
        else:
            result['prompt_length']= len(tokenizer(prompt).input_ids)
            result['only_doc_prompt_length'] = len(tokenizer(demonstration_str).input_ids)
            
            with torch.no_grad():
                inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                start_time = time.time()
                outputs = model.generate(
                    inputs,
                    max_new_tokens=args.generation_max_length,
                    do_sample=False,
                    temperature=0,
                    top_p=1.0,
                    eos_token_id=stop_token_ids,
                    )
                end_time = time.time()


            time_taken = end_time - start_time
            total_read_time += time_taken
            result['time_taken'] = time_taken
            
            result['generated_answers'] = tokenizer.decode(outputs[0][inputs.size(1):], skip_special_tokens=True).strip()
    
        if 'context' in result:
            result.pop('context')
        if 'ctxs' in result:
            result.pop('ctxs')
        read_results.append(result)
        
        if i % args.interval == args.interval - 1 or i == len(data) - 1:
            json.dump(read_results, open(os.path.join(save_dir, f'{args.results_file_name}.json'), 'w'), indent=4)

    logger.info(f"n_skip: {n_skip}")
    
    logger.info(f"total read time : {total_read_time}")
    metrics = evaluate_QA(read_results, ans_key='answers', predict_key='generated_answers')
    try:
        metrics['avg_comp_length'] = np.mean([result['only_doc_prompt_length'] for result in read_results])
    except:
        logger.info('no measured length')
    logger.info(f"metris: {metrics}")
    logger.info(f"{save_dir}")

    json.dump(read_results, open(os.path.join(save_dir, f'{args.read_file_name}.json'), 'w'), indent=4)
    with open((os.path.join(save_dir, f'{args.metrics_file_name}.txt')),'w') as f:
        f.write(json.dumps(metrics))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--fshot_path', type=str)
    parser.add_argument('--segment_size', type=int, default=5)
    parser.add_argument('--max_iteration', type=int, default=6)

    parser.add_argument('--batch_decoding', action="store_true", default=False)
    parser.add_argument('--batch_size', type=int, default=100)



    # compress
    parser.add_argument('--compressor_name_or_path', type=str)
    parser.add_argument('--compressor_dir', type=str, default='')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--compress_output_dir', type=str, required=True)
    parser.add_argument('--read_output_dir', type=str, required=True)
    parser.add_argument('--wo_prev_eval', action="store_true", default=False)
    parser.add_argument('--results_file_name', type=str, default='results')
    parser.add_argument('--read_file_name', type=str, default='results')
    parser.add_argument('--metrics_file_name', type=str, default='metrics')


    # read
    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--cache_dir', type=str)
    parser.add_argument('--interval', type=int, default=2)
    parser.add_argument('--fshot', action='store_true', default=False)
    parser.add_argument('--read_wo_prev_eval', action="store_true", default=False)
    parser.add_argument('--read_wo_prev_summary', action="store_true", default=False)
    parser.add_argument("--do_sample", action="store_true", help="whether to use sampling (false is greedy)")
    parser.add_argument("--generation_max_length", type=int, default=32, help="max number of tokens to generate")
    parser.add_argument("--generation_min_length", type=int, default=0, help="min number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0, help="generation temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="top-p parameter for nucleus sampling")
    parser.add_argument("--no_cuda", action="store_true", help="disable cuda")
    parser.add_argument("--no_bf16", action="store_true", help="disable bfloat16 -- use fp32 instead")
    parser.add_argument("--debug", action="store_true", help="for debugging")
    
    
    # wandb
    parser.add_argument(
        "--use_wandb", action="store_true", default=False, help=""
    )
    parser.add_argument(
        "--wandb_project_name", type=str, default='', help=""
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default='test', help=""
    )
    
    args = parser.parse_args()

    main(args)