import os
import time
import requests
from pydub import AudioSegment
from pydub.playback import play
import whisper
import speech_recognition as sr
import numpy as np
import threading
import click

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

def watch_file():
    url = "http://localhost:8000/output.wav"
    local_file = "output.wav"
    check_interval=0
    last_mod_time = None

    while True:
        new_mod_time = get_remote_file_mod_time(url)
        
        if new_mod_time and new_mod_time != last_mod_time:
            print(f"Change detected, downloading and playing new file...")
            
            # wavファイルのダウンロード
            if download_file(url, local_file):
                # オーディオの再生
                audio = AudioSegment.from_wav(local_file)
                play(audio)

            last_mod_time = new_mod_time
        
        time.sleep(check_interval)

def input_from_voice(mic_check):
    # マイク一覧を表示する場合
    mic_select = mic_check
    model = whisper.load_model("base")
    
    # 録音の設定
    MIC_NUMBER = 0
    RATE = 16000

    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300  # 環境に合わせて調整
    recognizer.dynamic_energy_threshold = False  # 自動調整を無効化
    if mic_select:
        print("".join(list(map(lambda tup: f"{tup[0]}: {tup[1]}\n", enumerate(sr.Microphone.list_microphone_names())))))
        index = input("マイクの番号を入力してください：")
        index = int(index) if index.isdigit() else None
    else: 
        index = MIC_NUMBER
    while True:
        # 録音開始
        with sr.Microphone(device_index = index, sample_rate=RATE) as source:
            recognizer.adjust_for_ambient_noise(source)
            print("Recordning...")
            try:
                audio = recognizer.listen(source, timeout=30)
                print("Finished recording")
            except sr.WaitTimeoutError:
                # これがないとエラーで落ちます
                print("Timeout: No speech detected within the timeout period.")
                continue

        audio_np = np.frombuffer(audio.get_raw_data(), dtype=np.int16).astype(np.float32) / 32768.0 
        transcribed_text = model.transcribe(audio_np, language="ja")["text"] 
        if transcribed_text == "":
            # 文字起こししたものが空白だった場合、次ループへ移行
            continue
        f = open("output.txt", "w") 
        f.write(transcribed_text) 
        f.close() 

@click.command()
@click.option("--mic_check", "-m", default=True)
def main(mic_check):
    thread_voice = threading.Thread(target=watch_file)
    thread_input = threading.Thread(target=input_from_voice, args=(mic_check,))
    thread_voice.start()
    thread_input.start()

if __name__ == "__main__":
    main()
    
