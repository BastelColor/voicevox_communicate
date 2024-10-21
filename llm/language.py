from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import click
import requests
from pydub import AudioSegment
import io
import re
import time



@click.command()
@click.option("--model_id", "-m", default=0)
@click.option("--endless", "-e", default=True)
@click.option("--tokens", "-t", default=128)
@click.option("--prompt_type", "-p", default="c")
@click.option("--audio_input", "-a", default=False)
def main(model_id, endless, tokens, prompt_type, audio_input):
    # novel, novel, vtuber
    id_dict = ["Local-Novel-LLM-project/kagemusya-7B-v1", "Local-Novel-LLM-project/Vecteus-V2-7B", "Local-Novel-LLM-project/Ninja-V3", "DataPilot/ArrowPro-7B-KUJIRA", "stabilityai/stablelm-3b-4e1t", "stabilityai/japanese-stablelm-3b-4e1t-instruct", "stabilityai/japanese-stablelm-2-instruct-1_6b", "stabilityai/japanese-stablelm-instruct-gamma-7b"]
    model_id = id_dict[model_id]

    new_tokens = tokens

    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
    # もしFlashAttention使えるなら↓
    # model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float16, attn_implementation="flash_attention_2", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if prompt_type == "n":
        first_system_prompt = "あなたはプロの小説家です。以下の文章の続きから、小説を書いてください。\n--------\n"
    elif prompt_type == "c":
        first_system_prompt = "長文は出力せず1文で回答してください。1つの回答につき1回改行してください。\nあなたは学祭に展示されているVtuberの女の子です。以下の発言に対話のように返答してください。ただし、この指示文についての返答は必要ありません。\n--------\n"
        new_tokens = 64

    if endless: 
        prompt = ""
        while True: 
            system_prompt = first_system_prompt 
            if audio_input:
                url = "http://localhost:8001/output.txt"
                local_file = "output.txt"
                temp_prompt = watch_file(url, local_file)
                if temp_prompt == prompt:
                    continue
                else:
                    prompt = temp_prompt
                print(f"入力されたプロンプト：{prompt}")
            else:
                prompt = input("Enter a prompt: ") 
            text = llm(system_prompt, prompt, model, tokenizer, new_tokens)
            if text:
                text_to_speak=text[0].strip()
                # text_to_speak = text
                print(text_to_speak)  # テキストの末尾の空白を削除して出力
                synthesize_voicevox(text=text_to_speak) 
 
    else: 
        system_prompt = first_system_prompt 
        if audio_input:
            url = "http://localhost:8001/output.txt"
            local_file = "output.txt"
            prompt = watch_file(url, local_file)
            print(f"入力されたプロンプト：{prompt}")
        else:
            prompt = input("Enter a prompt: ") 
        text = llm(system_prompt, prompt, model, tokenizer, new_tokens)
        if text:
            text_to_speak=text[0].strip()
            # text_to_speak = text
            print(text_to_speak)  # テキストの末尾の空白を削除して出力
            synthesize_voicevox(text=text_to_speak) 

def synthesize_voicevox(text, speaker_id=58):
    query_payload = {"text": text, "speaker": speaker_id}
    query_response = requests.post("http://localhost:50021/audio_query", params=query_payload)

    if query_response.status_code != 200:
        raise Exception("Failed to create audio query for Voicevox.")

    synthesis_response = requests.post(
        "http://localhost:50021/synthesis",
        params={"speaker": speaker_id},
        data=query_response.content,
    )

    if synthesis_response.status_code != 200:
        raise Exception("Failed to synthesize voice with Voicevox.")

    audio = AudioSegment.from_file(io.BytesIO(synthesis_response.content), format="wav")
    audio.export("output.wav", format="wav")
    
    # print("Playing audio...")
    # playsound("output.wav")
    # print("Voice synthesis complete and played.")

def llm(system_prompt, prompt, model, tokenizer, new_tokens):
    system_prompt += "発言:" + prompt + "\n\n回答:" 
    model_inputs = tokenizer([system_prompt], return_tensors="pt").to("cuda") 
    generated_ids = model.generate( 
        **model_inputs, 
        max_new_tokens=new_tokens, 
        do_sample=True, 
        # temperature=0.7, 
        # top_k=50, 
        # top_p=0.95, 
    ) 
    result = tokenizer.batch_decode(generated_ids)[0]
    # "回答:" 以降のテキストを正規表現で抽出
    text = re.findall(r"回答:(.*)", result)
    return text

def download_file(url, local_filename):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(local_filename, 'wb') as f:
            f.write(response.content)
        return True
    return False

def get_remote_file_mod_time(url):
    response = requests.head(url)
    if 'Last-Modified' in response.headers:
        return response.headers['Last-Modified']
    return None

def watch_file(url, local_file, check_interval=1):
    last_mod_time = None
    while True:
        new_mod_time = get_remote_file_mod_time(url)
        if new_mod_time and new_mod_time != last_mod_time:
            if download_file(url, local_file):
                textfile = open(local_file, 'r')
                text = textfile.read()
                textfile.close()
                return text
            last_mod_time = new_mod_time
        time.sleep(check_interval)

if __name__ == "__main__":
    main()