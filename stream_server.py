#!/usr/bin/env python3

import json
import os
import asyncio
import websockets
import concurrent.futures
import logging
import numpy as np
import torch
import time
import threading
import json
from stream_processor import online_processor
import text_processor_helper as lang_helper

# audio, sr = librosa.load(result, sr= 16000, mono= True, dtype= np.float32)

# sd.play(audio, sr, blocking= True)
# time.sleep(10)

audio_list = []
end_for_vad = 0
start_for_STT = 0
last_chunck_speech = False
request_uncomplete = ""
packet_length = 1048576
audio_queue_lock = threading.Lock()

async def manage_subtitle(websocket, text_dict: dict):
    # text = ""
    # for lang in text_dict:
    #     text += lang + " : " + text_dict[lang]
    print("mng_sub:   ", text_dict)
    await websocket.send(json.dumps(text_dict))
    # for lang in text_dict:


async def process(websocket, min_chunk = 1):
    print(" ... start of process func ...")
    global audio_queue
    global audio_queue_length
    global start_time
    global start_for_STT
    global last_chunck_speech
    global request_uncomplete
    global processor
    global stop
    audio = np.array([], dtype= np.float32)
    stop = False
    loop = asyncio.get_running_loop()
    while True:
        if stop: break
        audio_length, queue_length = (len(audio) + audio_queue_length, audio_queue.qsize())
        print("process:   audio_length = ", audio_length/16000, " and length of remain audio = ", len(audio)/16000)
        if audio_length < min_chunk*16000:
            # print("nope")
            await asyncio.sleep(min_chunk - audio_length/16000)
            continue
        print("process:   queue_length = ", queue_length)
        for i in range(queue_length):
            temp = await audio_queue.get()
            audio = np.concatenate((audio, temp))
            with audio_queue_lock:
                audio_queue_length -= len(temp)
        s = time.time()
        voice_activity = processor.pre_process_iter(audio)
        if len(voice_activity) == 0: next_begin = None
        else:
            # print("process:   tar_lang = ", processor.langs)
            transcription_data = await loop.run_in_executor(None, processor.process_iter, audio, voice_activity)
            src_lang, transcription, translation, next_begin = transcription_data
            print("process:   transcription = ", transcription)
            print("process:   audio_duration = ", audio_length/16000, ", process_time = ", time.time() - s)
            translation[src_lang]= transcription
            await manage_subtitle(websocket, translation)
        print("process:   next_begin = ",next_begin)
        if next_begin is not None: audio = audio[np.floor(next_begin*16000).astype(int):]
        else: audio = np.array([], dtype= np.float32)
        if stop:
            break
    
# def process_thread(websocket, loop, min_chunk = 1):
#     asyncio.run_coroutine_threadsafe(process(websocket, min_chunk), loop)

def process_thread(websocket, min_chunk=1.5):
    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(process(websocket, min_chunk))
    loop.close()

async def recognize(websocket, path):
    print("first of RECOGNIZE ....")
    # time.sleep(3)
    global audio_queue
    global audio_queue_length
    global pool
    global packet_length
    global processor
    global start_time
    global stop

    # loop = asyncio.get_running_loop()
    audio_queue_length = 0
    start_time = 0
    stop = False
    logging.info('Connection from %s', websocket.remote_address)
    # print("start of recieving loop ....")
    # time.sleep(3)
    min_chunk = 4
    thread2 = threading.Thread(target=process_thread, args=(websocket, min_chunk))
    thread2.start()
    while True:
        # print("waiting to recieve  ....")
        # time.sleep(3)
        message = await websocket.recv()
        # print("message recieved ....")
        # time.sleep(3)
        # Load configuration if provided
        if isinstance(message, str) and 'config' in message:
            jobj = json.loads(message)['config']
            logging.info("Config %s", jobj)
            langs = []
            for lang in jobj:
                if jobj[lang] != 'none':
                    langs.append(lang_helper.STREAM_SUPPURTED_LANGUAGES_FLORES_200[lang])
            processor.langs = langs
            print("recognize: ", langs)
            if 'phrase_list' in jobj:
                phrase_list = jobj['phrase_list']
            if 'sample_rate' in jobj:
                sample_rate = float(jobj['sample_rate'])
            # if 'model' in jobj:
            #     model = Model(jobj['model'])
            #     model_changed = True
            if 'words' in jobj:
                show_words = bool(jobj['words'])
            if 'max_alternatives' in jobj:
                max_alternatives = int(jobj['max_alternatives'])
            continue

        # Create the recognizer, word list is temporary disabled since not every model supports it
        if message == '{"eof" : 1}':
            stop = True
            break
        temp = np.frombuffer(message, dtype=np.int16).astype(np.float32)/2**15
        audio_queue.put_nowait(temp)
        audio_queue_length += len(temp)
        print("recognize: audio_queue_length = ", audio_queue_length/16000)
        # audio_buffer = np.concatenate((audio_buffer, temp))
        # print(audio_buffer.shape)
        # request, response, audio_out, audio_out_params, stop = await loop.run_in_executor(pool, process)
        if stop: break



async def start():
    print("first of stRT ....")
    # time.sleep(3)
    global processor
    global args
    global pool
    global audio_queue
    # Enable loging if needed
    #
    # logger = logging.getLogger('websockets')
    # logger.setLevel(logging.INFO)
    # logger.addHandler(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO)

    args = type('', (), {})()

    args.interface = os.environ.get('VOSK_SERVER_INTERFACE', '0.0.0.0')
    args.port = int(os.environ.get('VOSK_SERVER_PORT', 2800))
    args.model_path = os.environ.get('VOSK_MODEL_PATH', 'model')
    args.spk_model_path = os.environ.get('VOSK_SPK_MODEL_PATH')
    args.sample_rate = float(os.environ.get('VOSK_SAMPLE_RATE', 8000))
    args.max_alternatives = int(os.environ.get('VOSK_ALTERNATIVES', 0))
    args.show_words = bool(os.environ.get('VOSK_SHOW_WORDS', True))

    # Gpu part, uncomment if vosk-api has gpu support
    #
    # from vosk import GpuInit, GpuInstantiate
    # GpuInit()
    # def thread_init():
    #     GpuInstantiate()
    # pool = concurrent.futures.ThreadPoolExecutor(initializer=thread_init)

    # model = Model(args.model_path)
    # spk_model = SpkModel(args.spk_model_path) if args.spk_model_path else None
    
    print("initialing ...")

    processor = online_processor()
    processor.initialize()

    audio_queue = asyncio.Queue()

    pool = concurrent.futures.ThreadPoolExecutor((os.cpu_count() or 1))

    print("\n \n ------ models loaded ----- \n \n")
    # time.sleep(3)
    async with websockets.serve(recognize, args.interface, args.port):
        await asyncio.Future()


if __name__ == '__main__':
    print("first of main ....")
    # time.sleep(3)
    asyncio.run(start())
