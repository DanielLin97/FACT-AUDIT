import requests
import json
import os
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from urllib.parse import urlparse, unquote
from bs4 import BeautifulSoup
import re
import argparse
import os
import time

# Get the value of HF_HUB_CACHE
hf_hub_cache_path = os.getenv('HF_HUB_CACHE')

# Check if the variable is set
if hf_hub_cache_path:
    print(f"HF_HUB_CACHE is set to: {hf_hub_cache_path}")
else:
    print("HF_HUB_CACHE is not set.")

# TODO change the api key and url

API_KEY_gpt = 'sk-xxxxxx'
HEADERS_gpt = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY_gpt}"
}
API_URL_gpt = 'https://api.openai.com/v1/chat/completions' #"<API URL>"


device = 'cuda:0'

# TODO update your model path, we implement the inference function for llama2 (llama2_generate) and llama3 (llama3_generate)
model_path = 'meta-llama/Llama-2-13b-chat-hf' #'<MODEL PATH>'


def gpt4o_turbo_generate(text, temp=None, presence_penalty=None):
    # print(text)
    num = 50
    res = ""
    messages = [{"role": "user", "content": text}]
    while num > 0 and len(res)==0:
        try:
            if temp:
                data = json.dumps({"model": "gpt-4o-2024-05-13", "messages": 
                    [{"role": "user", "content": text}],
                    'temperature': temp
                })
            else:
                data = json.dumps({"model": "gpt-4o-2024-05-13", "messages": 
                    [{"role": "user", "content": text}]
                })
            response = requests.post(API_URL_gpt, headers=HEADERS_gpt, data=data)
            response_json = response.json()
            res = response_json['choices'][0]['message']['content']
        except Exception as e:
            print(e)
            time.sleep(10)
            num -= 1
    
    return res

def gpt4omini_turbo_generate(text, temp=None, presence_penalty=None):
    # print(text)
    num = 50
    res = ""
    messages = [{"role": "user", "content": text}]
    while num > 0 and len(res)==0:
        try:
            if temp:
                data = json.dumps({"model": "gpt-4o-mini-2024-07-18", "messages": 
                    [{"role": "user", "content": text}],
                    'temperature': temp
                })
            else:
                data = json.dumps({"model": "gpt-4o-mini-2024-07-18", "messages": 
                    [{"role": "user", "content": text}]
                })
            response = requests.post(API_URL_gpt, headers=HEADERS_gpt, data=data)
            response_json = response.json()
            res = response_json['choices'][0]['message']['content']
        except Exception as e:
            print(e)
            time.sleep(10)
            num -= 1
    
    return res


def fix_json_string(json_string):
    json_string = re.sub(r"(?<!\\)'", '"', json_string)

    json_string = re.sub(r'([{,]\s*)(\w+)\s*:', r'\1"\2":', json_string)
    json_string = re.sub(r'(\})(\s*\{)', r'\1,\2', json_string) 
    json_string = re.sub(r'(\]|\})(\s*\")', r'\1,\2', json_string)
    
    json_string = re.sub(r'(?<!\\)"', r'\\"', json_string)

    json_string = json_string.replace('\\"', '"')
    
    return json_string

def find_dict(answer):
    match = re.search(r'({.*})', answer, re.DOTALL)
    if match:
        json_string = match.group(1)
        try:
    
            data = json.loads(json_string)
        except:
            data = fix_json_string(json_string)
            data = json.loads(data)
    else:
        print("No JSON data found")
    return data


def gpt4o_generate(text, temp=1.0):  # model   gpt-4
    # print(text)
    num = 50
    res = ""
    while num > 0 and len(res)==0:
        try:
            if temp:
                data = json.dumps({"model": "gpt-4o-2024-05-13", "messages": 
                    [{"role": "user", "content": text}],
                    'temperature': temp
                })
            else:
                data = json.dumps({"model": "gpt-4o-2024-05-13", "messages": 
                    [{"role": "user", "content": text}]
                })
            response = requests.post(API_URL_gpt, headers=HEADERS_gpt, data=data)
    
            response_json = response.json()
            res = response_json['choices'][0]['message']['content'].replace('\n','')
        except Exception as e:
            print(e)
            time.sleep(5)
            num -= 1
    
    return res



def get_gpt4_score(question, answer, ref_ans, key_point):
    prompt = {"name": "single-code", "type": "single", "system_prompt": "You are a helpful assistant.", "prompt_template": "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the fact-checking question displayed below. Your evaluation should consider factors such as correctness (high priority), relevance, soundness, and completeness of the response. You will be given a high-quality reference response and the assistant's response. Begin your evaluation by comparing the assistant's response with the reference response. Identify and correct any mistakes in the answer and its justification. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\". Please do not score higher than 3.0 if the assistant's answer [Factual, Non-Factual, or Not Enough Information] is incorrect. If the quality of the justification is poor, please score a low rating not higher than 3.0, even the answer is correct.\n\n[Question]\n{question}\n\n[Key Point]\n{key_point}\n\n[The Start of Reference Response]\n{ref_answer}\n[The End of Reference Response]\n\n[The Start of Assistant's Response]\n{answer}\n[The End of Assistant's Response]", "output_format": "[[rating]]"}
    judge_prompt = prompt["prompt_template"].replace('{question}', str(question)).replace('{ref_answer}', ref_ans).replace('{answer}', answer).replace('{key_point}', key_point)
    score_res = gpt4omini_turbo_generate(judge_prompt, temp=0)
    return score_res

model = AutoModelForCausalLM.from_pretrained(model_path).half().eval().to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)


def llama2_generate(text):
    input_text = "<s>[INST] {} [/INST]".format(text)
    model_inputs = tokenizer(input_text, return_tensors="pt").to(device)    
    # output = model.generate(**model_inputs, max_new_tokens=1024, do_sample=False, top_p=0.9, temperature=0.6, num_beams=1)
    output = model.generate(**model_inputs, max_new_tokens=1024, do_sample=False, num_beams=1)
    resp = tokenizer.decode(output[0], skip_special_tokens=True).split('[/INST]')[1].strip()
    return resp


def llama3_generate(text):
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    input_text = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n".format(text)
    model_inputs = tokenizer(input_text, return_tensors="pt").to(device)    
    # output = model.generate(**model_inputs, max_new_tokens=1024, do_sample=True, top_p=0.9, temperature=0.6, num_beams=1, eos_token_id=terminators)
    output = model.generate(**model_inputs, max_new_tokens=1024, do_sample=False, num_beams=1, eos_token_id=terminators)
    resp = tokenizer.decode(output[0][model_inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return resp


def gen_fact_problem_template(question):
    prompt = 'This is a fact-checking task. Please directly verify the factual accuracy of the statement provided below. Your response should conclude whether the statement in the question is Factual, Non-Factual, or Not Enough Information based on the claim itself, or an attached evidence set, or a given conversation thread of user replies. \n\nQuestion: {question}\nAnswer: [Factual, Non-Factual, or Not Enough Information]\nJustification:'    
    return prompt.format(question=question)

def gen_vote_template(question, ref1, ref2, ref3):
    prompt = 'This is a fact-checking task. Please vote based on the three answers:\nanswer1 {ref1};\nanswer2 {ref2};\nanswer3 {ref3},\nand select the two most similar answers to merge. If the three answers conflict with each other, summarize the most appropriate answer based on the three answers. The output format should be consistent with the three answers. Your response should conclude whether the statement in the question is Factual, Non-Factual, or Not Enough Information based on the claim itself, or an attached evidence set, or a given conversation thread of user replies. \n\nQuestion: {question}\nAnswer: [Factual, Non-Factual, or Not Enough Information]\nJustification:'    
    return prompt.format(question=question, ref1 = ref1, ref2 = ref2, ref3 = ref3)


def deep_search(task_name, seed_prompts):  #knowledge point, seed questions for the knowledge point
    history = []
    steps = []
    
    score_func = get_gpt4_score           #give score according to the ref answer and the output of the target model
    optimize_func = gpt4o_turbo_generate   #give ref answer

    final_data['search_optimize_func'] = str(optimize_func)
    final_data['score_func'] = str(score_func)

    for idx in range(len(seed_prompts)):
        i = seed_prompts[idx]  # the idx-th question (test case)
        question_prompt = gen_fact_problem_template(i['prompt']) #prompt means the question
        ref_ans1 = optimize_func(question_prompt, temp=0)
        ref_ans2 = optimize_func(question_prompt, temp=0)
        ref_ans3 = optimize_func(question_prompt, temp=0)
        ref_ans = optimize_func(gen_vote_template(i['prompt'], ref_ans1, ref_ans2, ref_ans3), temp=0)
        while True:
            try:
                ref_ans = judge_ref_answer(question_prompt, key_point, ref_ans)
                break
            except:
                continue
        gen_res = llama2_generate(question_prompt)
        seed_prompts[idx]['answer'] = gen_res
        seed_prompts[idx]['ref_ans'] = ref_ans
        i = seed_prompts[idx]
        for _ in range(3):
            try:
                print('deep_search: first try')
                score_res = score_func(i['prompt'], i['answer'], i['ref_ans'], i['key_point'])
                i['comparison'] = score_res
                i['score'] = float(re.findall(r'\[\[.*?\]\]', score_res.strip())[-1].replace('[[', '').replace(']]', ''))
                break
            except Exception as e:
                print(e)

        seed_prompts[idx]['score'] = i['score']
        seed_prompts[idx]['comparison'] = i['comparison']
        history.append(i)  #record the tested test case
        steps.append(i)

    show_num = 5
    while len(steps) < 30:  #Iterative search
        print('current step:', len(steps))
        optimized_prompt = """This task involves generating test cases for the fact-checking task. Fact-checking is an important capability of LLMs, where the LLM should analyze textual information to identify the factuality of the source claim. Here, you need to ask the LLM to be tested to accurately assess the factuality of the information presented with the claim itself, or auxiliary information.
Previous Prompts:\n\n"""

        history = sorted(history, key=lambda k: k['score'])  #ranking according to score
        tmp_h = []

        for j in history:
            exact_match_start = False
            for k in tmp_h:
                if str(j['prompt']).split()[:3] == str(k['prompt']).split()[:3]:
                    exact_match_start = True
                    break
            if not exact_match_start:
                tmp_h.append(j)

        history = tmp_h[::-1]  #inverted        from high to low

        sample_his = []
        good_cases = [item for item in history]  #no condition?
        bad_cases = [item for item in history if item['score'] <= 3]

        
        if len(good_cases) < 5 or len(bad_cases) < 2:
            sample_his = history[-show_num:]   #last 5 test cases
            sample_his = sorted(sample_his, key=lambda k: k['score'], reverse=True) #score from high to low
        else:
            sample_his = random.sample(bad_cases, 2)  #randomly select 2 bad cases
            retry = 50
            while len(sample_his) < 5 and retry > 0:
                retry -= 1
                good_case = random.choice(good_cases)
                if good_case not in sample_his:
                    sample_his.append(good_case)
            if len(sample_his) < 5:
                sample_his = history[-show_num:]
            sample_his = sorted(sample_his, key=lambda k: k['score'], reverse=True)

        for j in sample_his[-show_num:]:
            optimized_prompt += "Prompt: {}\nTest Mode: {}\nKey Point: {}\nAnswer: {}\nComment: {}\nScore: {}\n\n".format(j['prompt'], j['test_mode'], j['key_point'], j['answer'], j['comparison'], j['score'])
        

        optimized_prompt += """The objective is to create new prompts that are challenging for the language model, with a focus on diverse types of instructions about "{task_name}". Each prompt should be solvable by a language model completely, and aimed at achieving a lower score (indicating difficulty and complexity).

Guidelines for Creating New Prompts:

1. Each prompt to be fact-checked should be solvable by a language model (no visual task) and should contain all necessary information.
2. Understand how factuality in the Prompt is discerned by the tested model as shown in its Answer and the comment on the score.
3. Aim for prompts that would result in a low score (less than 3.0), indicating a high level of complexity and difficulty of the question that requires more effort to identity the factuality or misinformation.
4. Do not repeat topics across different instructions in the previous prompts to maximize diversity.
5. The key_point should be no more than 15 words and summarize the key points of the prompt to clearly state the target content to be fact-checked.
6. The test_mode should be one of the three options: 1) [claim], (i.e., only the source claim), or 2) [evidence], (i.e., additional contrastive evidence based on Wikipedia), or 3) [wisdom of crowds], (i.e., user comments on social media).
7. The auxiliary_info should be provided accoriding to the test_mode: if not the [claim] mode is selected, generate the auxiliary information "auxiliary_info" for the source claim. If else, "auxiliary_info" is empty. \n For "auxiliary_info" of [evidence], please ensure that: 1) more than three pieces of evidence are in "auxiliary_info", and 2) the provided pieces of detailed evidence in "auxiliary_info" must only be ground truth equoted directly and solely from Wikipedia word for word (without any personal insight), where different amounts of supported, refuted, and neutral evidence to the source claim should be included; \n For "auxiliary_info" of [wisdom of crowds], please ensure that: 1) the depth of the conversation tree in "auxiliary_info" must be more than two, and 2) the hierarchical conversation tree in "auxiliary_info" can be noisy but valuable to help verify the source claim.
8. Please focus on "{task_name}" constraints, and ensure that upon careful consideration a human fact-checker with commonsense can identify the factuality of the new prompt.
9. The new prompt should be STRICTLY within 512 words and should not be too long.

Please generate a new test case. Output in a json format: {"key_point": string(...), "test_mode": string(...), "prompt": {"source_claim": string(...), "auxiliary_info": string(...)}}. """

        try:
            print('deep_search: second try')
            optimized_prompt = optimized_prompt.replace(r"{task_name}", task_name)  # knowledge point
            # optimized_res = optimize_func(optimized_prompt) 
            optimized_res = gpt4o_generate(optimized_prompt) 
            # pattern = r'```json\n(.+?)```'
            # test_case = json.loads(re.search(pattern, optimized_res, re.DOTALL).group(1))
            test_case = find_dict(optimized_res)
            count_wiki_judge = 0
            while True:
                if wikipedia_judge(test_case) == False:
                    count_wiki_judge += 1
                    print(test_case)
                    if count_wiki_judge >= 5:
                        print('dead loopppp')
                        break
                    prompts = judge_new_case(task_name, json.dumps(test_case))
                    test_case = find_dict(prompts)
                    continue
                else:
                    break
            test_case = judge_new_case(task_name, json.dumps(test_case))
            test_case = find_dict(test_case)
            new_prompt = test_case['prompt']
            key_point = test_case['key_point']
            test_mode = test_case['test_mode']
            question_prompt = gen_fact_problem_template(new_prompt)
            ref_ans1 = optimize_func(question_prompt, temp=0)
            ref_ans2 = optimize_func(question_prompt, temp=0)
            ref_ans3 = optimize_func(question_prompt, temp=0)
            ref_ans = optimize_func(gen_vote_template(i['prompt'], ref_ans1, ref_ans2, ref_ans3), temp=0)
            while True:
                try:
                    ref_ans = judge_ref_answer(question_prompt, key_point, ref_ans)
                    break
                except:
                    continue
            gen_res = llama2_generate(question_prompt)

            score_res = score_func(new_prompt, gen_res, ref_ans, key_point)
            if len(re.findall(r'\[\[.*?\]\]', score_res.strip())) == 0:
                print("score invalid")
                continue
            score = float(re.findall(r'\[\[.*?\]\]', score_res.strip())[-1].replace('[[', '').replace(']]', ''))

            history.append({
                'prompt': new_prompt,
                'answer': gen_res,
                'ref_ans': ref_ans,
                'comparison': score_res,
                'key_point': key_point,
                'test_mode': test_mode,
                'score': score
            })

            steps.append({
                'prompt': new_prompt,
                'answer': gen_res,
                'ref_ans': ref_ans,
                'comparison': score_res,
                'key_point': key_point,
                'test_mode': test_mode,
                'score': score
            })

            if 'optimize_prompt' not in final_data:
                final_data['optimize_prompt'] = optimized_prompt
            final_data[task_name]['steps'] = steps
            
            with open(f'{output_path}/log.json', 'w', encoding='utf-8') as f:
                json.dump(final_data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(e)
            continue


def gen_seed(task_name, categories):
    prompt_template = """Fact-checking is an important capability of LLMs, where the LLM should analyze textual information to identify the factuality of the source claim. Here, you need to ask the LLM to be tested to accurately assess the factuality of the information presented within the source claim according to the claim itself or the auxiliary information.
Here is a taxonomy for the fact-checking task:
{categories}

Step 1: Please read the provided initial taxonomy carefully. Based on this, please generate 10 test cases of "{task_name}" category under different topics (e.g., Politics, Health and Medicine, Technology and Innovation, Environment and Climate Change, Social Events and News, Economy and Finance, History and Culture, Science and Research, International News and Diplomacy, Law and Regulation), to test if language models can accurately identify facts or misinformation in the source claim on task "{task_name}". 

Step 2: When generating each test case, consider which one of the three optional test modes is the most suitable: [claim], [evidence], and [wisdom of crowds] for each case. \n For [claim], the factuality can be verified according to the source claim itself;\n For [evidence], the factuality of the source claim needs to be verified according to the attached evidence set;\n For [wisdom of crowds], the factuality of the source claim needs to be assessed from the simulated conversation tree of user comments on social media.

Step 3: Based on the selected test mode in Step 2, if not the [claim] mode is selected, generate the auxiliary information "auxiliary_info" for the source claim. If else, "auxiliary_info" is empty. \n For "auxiliary_info" of [evidence], please ensure that: 1) more than three pieces of evidence are in "auxiliary_info", and 2) the provided pieces of detailed evidence in "auxiliary_info" must only be ground truth equoted directly and solely from Wikipedia word for word (without any personal insight), where different amounts of supported, refuted, and neutral evidence to the source claim should be included; \n For "auxiliary_info" of [wisdom of crowds], please ensure that: 1) the depth of the conversation tree in "auxiliary_info" must be more than two, and 2) the hierarchical conversation tree in "auxiliary_info" can be noisy but valuable to help verify the source claim.

Step 4: Key_point is a short sentence that summarizes the key point you want to test the language model, clearly stating the target content to be fact-checked. The constraints on "{task_name}" should be explicitly expressed. Besides, your test cases should cover common topics in fact-checking and different test modes mentioned before, to increase prompt diversity. Please be as diverse as you can but focus on "{task_name}" and ensure the prompt is text-only (no multimodal).

Step 5: Repeat Step 1-4 for each test case and then form all the test cases into a JSON format. The test_mode of the test cases should include [claim], [evidence], and [wisdom of crowds].

Please reply strictly in the following format:

Step 1 "source_claim":
Step 2 "test_mode": 
Step 3 "auxiliary_info":
Step 4 "key_point":
Step 5 Repeat Step 1-4 for each test case and then output one final JSON format: {"test_case1": {"key_point": string(...), "test_mode": string(...), "prompt": {"source_claim": string(...), "auxiliary_info": string(...)}}, "test_case2": {...}, ...}."""
    res = []
    while True:
        try:
            print('gen_seed')
            prompts = gpt4o_generate(prompt_template.replace(r"{task_name}", task_name).replace(r"{categories}", json.dumps(categories)), 0.0)

            prompts = find_dict(prompts)
            count_wiki_judge = 0
            while True:
                try:
                    count_wiki_judge += 1
                    wiki_judge = wikipedia_judge(prompts)
                    print(wiki_judge)
                    if wiki_judge == False:
                        prompts = judge_new_case(task_name, json.dumps(prompts))
                        prompts = find_dict(prompts)
                        if count_wiki_judge >=5:
                            print('dead loopppp')
                            break
                        continue
                    break
                except Exception as e:
                    print(e)
                    continue
            for k in prompts.keys():
                assert ("key_point" in prompts[k])
                assert ("prompt" in prompts[k])
                assert ("test_mode" in prompts[k])
                res.append(prompts[k])
            break
        except Exception as e:
            print(e)
            # print(prompts)
            continue
    return res
    
def analysis(task_names):
    prompt_template = """Fact-checking is an important capability of LLMs, where the LLM should analyze textual information to identify the factuality of the source claim. Here, the LLM must be tested to accurately assess the factuality of the information presented within the source claim according to the claim itself or the auxiliary information.

Here is a sub task's taxonomy as well as the averaged score on these tasks(lower means worse performance):
{taxonomy}

And here are some bad cases:
{bad_cases}
Based on the given information, please judge if the taxonomy is comprehensive, if so please just output [[Stop]]. 

If not, please give me a new possible issue you inferred from the present taxonomy and bad cases. Please focus on {main_task}. Ensure the new task is text-only (no multimodal). Also give a brief explanation of how you find the issue. Please output in a JSON format: {"task_name": ..., "explanation":...}"""
    
    bad_cases = {}
    main_task = task_names[0].split(':')[0]
    sub_tax = {}
    for i in task_names:
        task_name = i
        scores = [float(j['score']) for j in final_data[task_name]['steps']]
        sub_task_name = task_name.split(':')[1]
        sub_tax[sub_task_name] = sum(scores) / len(scores)
        bad_cases[sub_task_name] = []
        for j in final_data[task_name]['steps']:
            if float(j['score']) <= 3.0:
                bad_cases[sub_task_name].append(j)
    
    bad_cases_str = ""
    for k in bad_cases.keys():
        if len(bad_cases[k]) == 0:
            continue
        samples = random.sample(bad_cases[k], min(2, len(bad_cases[k])))
        bad_cases_str += f"Task Name: {k}\nSamples:\n"
        for i in samples:
            bad_cases_str += "Prompt: {}\nResponse: {}\nScore: {}\n\n".format(i['prompt'], i['answer'], i['score'])
        
    tax = {main_task: sub_tax}
    for _ in range(3):
        try:
            print('analysis')
            new_task = gpt4o_generate(prompt_template.replace(r"{taxonomy}", json.dumps(tax)).replace(r"{bad_cases}", bad_cases_str).replace(r"{main_task}", main_task))
            if "[[Stop]]" in new_task:
                return "[[Stop]]"
            new_task = find_dict(new_task)
            # new_task = json.loads(new_task)
            final_data['new_points'].append(new_task)
            # from IPython import embed; embed()
            return new_task['task_name']
        except Exception as e:
            print(e)

def judge_ref_answer(prompt, key_point, answer):
    prompt_template = """Fact-checking is an important capability of LLMs, where the LLM should analyze textual information to identify the factuality of the source claim. Here, the LLM must be tested to accurately assess the factuality of the information presented within the source claim according to the claim itself or the auxiliary information.

Please judge whether the answer "{answer}"  is accurate and reasonable for the question "{prompt}" with the knowledge point "{key_point}" to be examined. 

If the answer is correct and reasonable, please ONLY output the original answer "{answer}" in a string format; If not, please revise and improve the original answer "{answer}" to be more precise, and ONLY output the improved answer in a string format."""
    try:
        print('judge_ref_answer')
        judge_res = gpt4o_generate(prompt_template.replace(r"{answer}", answer).replace(r"{prompt}", prompt).replace(r"{key_point}", key_point), temp=0)
        return judge_res
    except Exception as e:
        print(e)

def verify_sentence(sentence):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": sentence,
        "format": "json",
        "srlimit": 5  
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "query" not in data:
        print(data)

    
    for result in data["query"]["search"]:
        title = result["title"]
        snippet = result["snippet"]

        
        if sentence.lower() in snippet.lower() or sentence.lower() in title.lower():
            # print("Match found in Wikipedia article:", title)
            return True

    # print("No exact match found.")
    return False

def verify_long_sentence(sentence):
    
    phrases = sentence.split(" ")[:]  
    match_count = 0

    for phrase in phrases:
        if verify_sentence(phrase):
            match_count += 1

    if match_count >= len(phrases)/2: 
        # print("The sentence may come from Wikipedia.")
        return True
    else:
        # print("The sentence is likely not from Wikipedia.")
        return False

    
def extract_title_from_url(url):
    parsed_url = urlparse(url)
    title = parsed_url.path.split('/')[-1].split('#')[0].split('?')[0]
    return unquote(title)

def get_wikipedia_content_via_api(title):
    api_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "parse",
        "page": title,
        "format": "json",
        "prop": "text",
        "section": 0  
    }

    response = requests.get(api_url, params=params)
    data = response.json()

    if "parse" in data:
        html_content = data["parse"]["text"]["*"]
        return True
    else:
        print("Failed to retrieve page content.")
        print(title)
        return False
    
def extract_wikipedia_url(sentence):
    
    wiki_url_pattern = r"https?://(?:[a-z]{2,3}\.)?wikipedia\.org/wiki/[^\s]+"
    
    urls = re.findall(wiki_url_pattern, sentence)
    if len(urls)>0:
        urls = urls[0]
    else:
        urls = "" 
    
    return urls

def clean_and_split_content(content):
    soup = BeautifulSoup(content, "html.parser")
    text = soup.get_text(separator="\n") 
    paragraphs = text.split("\n") 
    return [para.strip() for para in paragraphs if para.strip()] 


def wikipedia_judge(new_case):
    coarse_verify = True
    try:
        # new_case = json.loads(new_point)
        if 'prompt' not in new_case.keys():
            for test_case in new_case.keys():
                if 'test_mode' in new_case[test_case].keys():
                    if 'evidence' in new_case[test_case]['test_mode']:
                        auxiliary_info = new_case[test_case]['prompt']['auxiliary_info']
                        if isinstance(auxiliary_info, list):
                            for sentence in auxiliary_info:
                                if isinstance(sentence, dict):
                                    sentence = json.dumps(sentence)
                                coarse_verify = verify_long_sentence(sentence)
                                if coarse_verify == False:
                                    print('wikepedia lookup failed')
                                    return coarse_verify
                        # print('wikepedia lookup passed')
                        elif isinstance(auxiliary_info, dict):
                            for sentence in auxiliary_info.values():
                                if not isinstance(sentence, str):
                                    if isinstance(sentence, list):
                                        for sent in sentence:
                                            coarse_verify = verify_long_sentence(sent)
                                            if coarse_verify == False:
                                                print('wikepedia lookup failed')
                                                return coarse_verify
                                    elif isinstance(sentence, dict):
                                        for sent in sentence.values():
                                            coarse_verify = verify_long_sentence(sent)
                                            if coarse_verify == False:
                                                print('wikepedia lookup failed')
                                                return coarse_verify
                                else:
                                    coarse_verify = verify_long_sentence(sentence)
                                    if coarse_verify == False:
                                        print('wikepedia lookup failed')
                                        return coarse_verify
                        elif isinstance(auxiliary_info, str):
                            coarse_verify = verify_long_sentence(auxiliary_info)
                            if coarse_verify == False:
                                print('wikepedia lookup failed')
                                return coarse_verify
                        else:
                            print(type(auxiliary_info))
                            coarse_verify = False
                            return coarse_verify
        else:
            if 'test_mode' in new_case.keys():
                if 'evidence' in new_case['test_mode']:
                    auxiliary_info = new_case['prompt']['auxiliary_info']
                    if isinstance(auxiliary_info, list):
                        for sentence in auxiliary_info:
                            if isinstance(sentence, dict):
                                sentence = json.dumps(sentence)
                            coarse_verify = verify_long_sentence(sentence)
                            if coarse_verify == False:
                                print('wikepedia lookup failed')
                                return coarse_verify
                        # print('wikepedia lookup passed')
                    elif isinstance(auxiliary_info, dict):
                        for sentence in auxiliary_info.values():
                            if not isinstance(sentence, str):
                                    if isinstance(sentence, list):
                                        for sent in sentence:
                                            coarse_verify = verify_long_sentence(sent)
                                            if coarse_verify == False:
                                                print('wikepedia lookup failed')
                                                return coarse_verify
                                    elif isinstance(sentence, dict):
                                        for sent in sentence.values():
                                            coarse_verify = verify_long_sentence(sent)
                                            if coarse_verify == False:
                                                print('wikepedia lookup failed')
                                                return coarse_verify
                            else:
                                coarse_verify = verify_long_sentence(sentence)
                                if coarse_verify == False:
                                    print('wikepedia lookup failed')
                                    return coarse_verify
                    elif isinstance(auxiliary_info, str):
                        coarse_verify = verify_long_sentence(auxiliary_info)
                        if coarse_verify == False:
                            print('wikepedia lookup failed')
                            return coarse_verify
                    else:
                        print(type(auxiliary_info))
                        coarse_verify = False
                        return coarse_verify
                else:
                    return coarse_verify
    except Exception as e:
        print('error in wikipedia lookup')
        print(e)
        print(new_case)
        coarse_verify = False
        return coarse_verify
    return coarse_verify
    
    

def judge_new_case(task_name, new_point):

    prompt_template = """Fact-checking is an important capability of LLMs, where the LLM should analyze textual information to identify the factuality of the source claim. Here, the LLM must be tested to accurately assess the factuality of the information presented within the source claim according to the claim itself or the auxiliary information.

Please judge whether the new test cases "{new_point}"  are suitable as diverse and comprehensive exam questions on the sub task "{task_name}". The judgment criteria are as follows:
1. Each claim of the new test cases should be important and meaningful to the main task, avoiding unnecessary ambiguity in the key point.
2. If "auxiliary_info" is not empty in each of the new test cases, it can be noisy but must be helpful to the fact verification process; If "auxiliary_info" is empty, just keep it empty.
3. If "test_mode" is [claim], "auxiliary_info" must be empty.
4. If "test_mode" is [wisdom of crowds], please check "auxiliary_info" that: a) the user comments in "auxiliary_info" should be valuable enough as the wisdom of crowds for fact verification and b) the depth of the propagation conversation tree composed of the user response in "auxiliary_info" must be a random integer more than two.
5. If "test_mode" is [evidence], please check "auxiliary_info" that: a) four or more random pieces of evidence are in "auxiliary_info", and b) the provided pieces of detailed evidence in "auxiliary_info" must be ONLY ground truth based on Wikipedia or other authority, where all supported, refuted, and neutral evidence to the source claim should be included.
6. The fact-checking topic in each test case should be diverse enough and sufficiently different from each other.

If the new test cases are judged suitable as the exam questions on the sub task "{task_name}" by checking the judgment criteria, please ONLY keep the original content "{new_point}" as output in a JSON format: [json]; If there is one test case not conforming to the judgment criteria, you have to revise and improve the original content "{new_point}" to conform to the aforementioned judgment criteria, and ONLY output the improved test cases in a JSON format: [json]."""

    try:
        print('judge_new_test_cases')
        judge_res = gpt4omini_turbo_generate(prompt_template.replace(r"{task_name}", task_name).replace(r"{new_point}", new_point), temp=0)
        return judge_res
    except Exception as e:
        print(e)

def judge_new_task(task_names, new_point):
    prompt_template = """Fact-checking is an important capability of LLMs, where the LLM should analyze textual information to identify the factuality of the source claim. Here, the LLM must be tested to accurately assess the factuality of the information presented within the source claim according to the claim itself or the auxiliary information.

Here is a sub task's taxonomy on the task "{main_task}":
{taxonomy}

Based on the given taxonomy, please judge whether the new test point "{new_point}" is suitable as a sub task on the task "{main_task}". The judge criteria are as follows:
1. The new test point should precisely cover an important and meaningful part of the main task.
2. The new test point should be sufficiently different from the existing test points.
3. The new test point should be text-only (no multimodal).

If the new test point "{new_point}" is suitable as a sub task on the task "{main_task}", please ONLY output [[Yes]]. If not, please first output [[No]], and then provide the reason why it's not suitable as a sub task on the task "{main_task}"."""
    main_task = task_names[0].split(':')[0] 
    sub_tax = []
    for i in task_names:
        task_name = i
        sub_task_name = task_name.split(':')[1]
        sub_tax.append(sub_task_name) 
        
    tax = {main_task: sub_tax}
    for i in range(3):
        try:
            print('judge_new_task')
            judge_res = gpt4o_generate(prompt_template.replace(r"{taxonomy}", json.dumps(tax)).replace(r"{new_point}", new_point).replace(r"{main_task}", main_task), temp=0)
            if "[[Yes]]" in judge_res or ("Yes" in judge_res and "[[No]]" not in judge_res):
                return True
            print(judge_res)
            return False
        except Exception as e:
            print(e)


if __name__ == '__main__':
    with open('../data/fact_cat.json', 'r') as f:
        categories = json.load(f)  

    parser = argparse.ArgumentParser()
    parser.add_argument('--category', default=None, type=str)
    args = parser.parse_args()

    main_cat = args.category.lower().replace(' ', '_') 
    points = categories[main_cat]  #test scenario in the taxonomy
    test_points = [f'{main_cat}:{point}' for point in points]
    
    output_dir = f"../result/factaudit/gpt-4o/{args.category.lower().replace(' ', '_')}/" #for each category under the taxonomy, create the filefolder

    num = 0
    output_path = ''
    
    while True: #create a new version
        folder_name = f'version_{num}'
        output_path = f'{output_dir}{folder_name}'
        num += 1
        if os.path.exists(output_path):
            continue
        else:
            break
    
    os.makedirs(output_path, exist_ok=True)

    final_data = {'init_points': test_points, 'new_points': []}  #knowledge points under this category
    idx = 0
    
    while idx < len(test_points) and idx <= 15: #15
        
        task = test_points[idx]  #main task: sub task
        print(f'Begin gen seed: {task}')

        seeds = gen_seed(task, categories)   #seed questions for each knowledge point
        
        final_data[task] = {
            'seed_prompts': seeds
        }
        with open(f'{output_path}/log.json', 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=4, ensure_ascii=False)
        
        deep_search(task, seeds) #knowledge point, seed questions for the knowledge point
        
        if idx == len(test_points) - 1:
            for x in range(3):
                new_task = analysis(test_points)
                if new_task == '[[Stop]]':
                    print('Encounter stop. End circuit.')
                    exit(0)
                if not judge_new_task(test_points, new_task):
                    if x < 2: continue
                    print('Reject three times. End circuit.')
                    exit(0)
                categories[main_cat].append(new_task)
                test_points.append(main_cat+':'+new_task)
                break

        idx += 1    