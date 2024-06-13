from doctest import debug
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import torch
from tqdm import tqdm
import os
import random
import json
import argparse
import datetime
import logging
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])


logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# TODO: Convert prints to logs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to pretrained model", required=True)
    parser.add_argument("--name", type=str,help="Output File Name", default="model_name", required=True)
    parser.add_argument("--run_full", help="run 0, 1, 3 shots", action = "store_true")
    parser.add_argument("--tqdm", help='whether to run tqdm', action = "store_true")
    parser.add_argument("--num_fewshots", help="Number of few shots as a string etc '0,1,2'", default = "0,1,3")
    parser.add_argument("--output_folder", type=str, default="output")
    parser.add_argument("--repeat",type = int, default=5, help= "Number of loops per prompt")
    parser.add_argument("--limit",type = int, default=0, help = 'run N number of samples')
    parser.add_argument("--device", type=str, help = "device map")
    parser.add_argument("--deterministic", help="disable sampling", action = "store_true")
    args = parser.parse_args()
    return args

def read_ttbhs() -> List[Dict]:
    questions = []
    with open("quiz-tatabahasa.jsonl") as fopen:
        for no, l in enumerate(fopen):
            l = json.loads(l)
            soalan = [l['question']]
            jawapan = None
            for c, k in l['choices'].items():
                soalan.append(f"{c}. {k['text']}")
                if k['answer']:
                    jawapan = c
            
            data = {
                'no': no,
                'objektif': 'Jawab soalan yang diberikan' if l['instruction'] is None else l['instruction'],
                'soalan': '\n'.join(soalan),
                'jawapan': jawapan,
            }
            questions.append(data)
    logging.info(f"TTBHS: Running {len(questions)} questions")
    return questions

def read_bmpt3() -> List[Dict]:
    with open('BM-A-pt3') as fopen:
        text = fopen.read()
    
    questions = []
    for t in text.split('no: ')[1:]:
        t = t.strip()
        no = t.split('\n')[0]
        objektif = t.split('objektif: ')[1].split('\n')[0]
        soalan = t.split('soalan:')[1].split('jawapan:')[0].strip()
        jawapan = t.split('jawapan: ')[1].split(',')[0].strip()
        data = {
            'no': no,
            'objektif': objektif,
            'soalan': soalan,
            'jawapan': jawapan,
        }
        questions.append(data)
    print(f"BM-A-PT3: Running {len(questions)} questions")
    return questions

def convert_prompt(row, answer = False) -> str:
    if answer:
        prompt = f"""
objektif: {row['objektif']}
soalan: {row['soalan']}
jawapan: {row['jawapan']}
    """
    else:
        prompt = f"""
objektif: {row['objektif']}
soalan: {row['soalan']}
jawapan:
    """
    return prompt.strip()

def most_common(l:List) -> str:
    return max(set(l), key=l.count)

def evaluate(questions:List[Dict]) -> float:
    filtered = [q for q in questions if 'output' in q]
    correct = 0
    for q in filtered:
        correct += most_common(q['output']) == q['jawapan']
    return (correct / len(filtered)) * 100

def run_test(args, model, tokenizer, questions, n_shots, n_repeat:int = 5) -> Tuple[List[Dict], float]:

    generate_kwargs = dict(
                    max_new_tokens=3,
                    top_p=0.95,
                    top_k=50,
                    temperature=0.5,
                    # if no_sample is true, then do_sample = False
                    do_sample=not args.deterministic,
                    num_beams=1,
                    repetition_penalty=1.05,
                )
    # not args.tqdm => if true, then disable = False => enable tqdm
    #               => if false, then disable = True => disable tqdm
    set_seed(1234)

    for i in tqdm(range(len(questions)), leave=True, disable = not args.tqdm):
        prompts = []
        
        if n_shots:
            arange = set(range(len(questions)))
            if args.deterministic:
                shots = sorted(arange - {i})[:n_shots]
            else:
                shots = random.sample(sorted(arange - {i}), n_shots)
            for no, s in enumerate(shots):
                prompts.append(f'Contoh soalan {no + 1}\n' + convert_prompt(questions[s], answer = True))
        prompts.append(convert_prompt(questions[i]))
        prompt = '\n\n'.join(prompts)
        inputs = tokenizer([prompt], return_tensors='pt', add_special_tokens=False).to(args.device)
        inputs.pop('token_type_ids', None)
        repeat, debug_output, debug_toks = [], [], []
        for _ in range(n_repeat):
            try:
                r = model.generate(**inputs,**generate_kwargs)
                r = tokenizer.decode(r[0]).split('jawapan:')[-1]
                debug_output.append(r)
                r = r.strip().split()
                repeat.append(r[0].replace('.', '').replace('</s>', '').split('\\')[0].split('/')[0])
        
            except Exception as e:
                print(e, r)
                pass
        questions[i]['input_tok'] = inputs.input_ids.tolist()[0]
        questions[i]['prompt'] = prompt
        questions[i]['output'] = repeat
        questions[i]['debug'] = debug_output

    # with open(f'{args.output_folder}/output-{n_shots}shot-{args.name}.json', 'w') as fopen:
    #     json.dump(questions, fopen)

    score = evaluate(questions)

    # logging.error (f"{n_shots}shot: {score}")

    return questions, score

def main():

    args = parse_args()
    logging.basicConfig(filename=f'eval.log', level=logging.INFO)
    logger.info(args)

    timestamp = datetime.datetime.now().replace(microsecond=0).isoformat().replace(":", "_")

    os.makedirs(args.output_folder + '/' + timestamp, exist_ok=True)

    if not args.run_full:
        logger.warning("The recent change sets the default value of run_full to `False`, if this is not intended "
              "run `evaluate.py` with `--run_full`")
    if args.deterministic:
        logger.warning("No sampling should only be used for debugging purposes.\n"
                       "This will disable random n_shots sampling and use the first n_shots instead.\n"
                       "Repeats will also be disabled.")
        args.repeat = 1
        logger.warning("Setting repeat to 1")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code = True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code = True,
        torch_dtype = torch.float16,
        device_map=args.device if args.device else device
    )

    config = {}
    config['run_name'] = args.name
    config['model_args'] = args.model_path
    config['full_run'] = args.run_full
    config['timestamp'] = timestamp

    qns_ttbhs = read_ttbhs()
    qns_bmpt3 = read_bmpt3()

    scores = {}

    ttbhs_scores = {}
    bmpt3_scores = {}

    num_fewshots = [int(x) for x in args.num_fewshots.split(',')]


    if args.run_full: #Full 0,1,3 for both tests
        # === tatabahasa ===
        for i in num_fewshots:
            q, s = run_test(args, 
                model=model, 
                tokenizer=tokenizer,
                questions=qns_ttbhs[:args.limit] if args.limit else qns_ttbhs,
                n_shots=i,
               n_repeat=args.repeat)
            with open(f'{args.output_folder}/{timestamp}/output-tatabahasa-{i}shot-{args.name}.json', 'w') as fopen:
                json.dump(q, fopen, indent=4)
            ttbhs_scores[f'n_shot={i}'] = s
        scores['tatabahasa'] = ttbhs_scores
        # === bmpt3 ===
        for i in num_fewshots:
            q, s = run_test(args, 
                model=model, 
                tokenizer=tokenizer,
                questions=qns_bmpt3[:args.limit] if args.limit else qns_bmpt3,
                n_shots=i,
                n_repeat=args.repeat)
            with open(f'{args.output_folder}/{timestamp}/output-bmpt3-{i}shot-{args.name}.json', 'w') as fopen:
                json.dump(q, fopen, indent= 4)
            bmpt3_scores[f'n_shot={i}'] = s

        scores['bmpt3'] = bmpt3_scores
        with open(f'{args.output_folder}/{timestamp}/score.json', 'w') as fopen:
            data = {"results": scores}
            conf = {"config": config}
            merged = {**data, **conf}
            json.dump(merged, fopen, indent=4)
    
    else: #3 shot for 5 qns - for debugging
        #TODO: Can we remove the for loop below? 
        #tatabahasa
        for i in [3]:
            q, s = run_test(args, 
                model=model, 
                tokenizer=tokenizer,
                questions=qns_ttbhs[:5],
                n_shots=i)
            with open(f'{args.output_folder}/{timestamp}/output-tatabahasa-{i}shot-{args.name}.json', 'w') as fopen:
                json.dump(q, fopen)
            ttbhs_scores[f'n_shot={i}'] = s
        #bmpt3
        for i in [3]:
            q, s = run_test(args, 
                model=model, 
                tokenizer=tokenizer,
                questions=qns_bmpt3[:5],
                n_shots=i)
            with open(f'{args.output_folder}/{timestamp}/output-bmpt3-{i}shot-{args.name}.json', 'w') as fopen:
                json.dump(q, fopen)
            bmpt3_scores[f'n_shot={i}'] = s
        
        scores['tatabahasa'] = ttbhs_scores
        scores['bmpt3'] = bmpt3_scores
        with open(f'{args.output_folder}/{timestamp}/score.json', 'w') as fopen:
            data = {"results": scores}
            conf = {"config": config}
            merged = {**data, **conf}
            json.dump(merged, fopen, indent=4)
    try:
        import pandas as pd
        print(pd.DataFrame(scores).to_markdown())
    except ImportError:
        print(scores)

if __name__ == "__main__":
    main()
