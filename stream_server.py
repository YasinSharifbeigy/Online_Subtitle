#!/usr/bin/env python3

import json
import os
import argparse
import asyncio
import websockets
import concurrent.futures
import logging
import webbrowser
import numpy as np
import time
import threading
import json
from stream_processor import online_processor
import text_processor_helper as lang_helper



audio_list = []
end_for_vad = 0
start_for_STT = 0
last_chunck_speech = False
request_uncomplete = ""
packet_length = 1048576
audio_queue_lock = threading.Lock()
indexes = {1: 'first', 2: 'second', 3: 'third'}



def empty_queue(q: asyncio.Queue):
  while not q.empty():
    q.get_nowait()
    q.task_done()



async def manage_subtitle(websocket, text_dict: dict, audio_length):
    print("mng_sub:   ", text_dict)
    response = {"subtitle": dict(), "duration": audio_length/16} #duration should be in ms
    for i, lang in enumerate(text_dict):
        response["subtitle"][indexes[i+1]] = text_dict[lang]
    await websocket.send(json.dumps(response))



async def process(websocket, min_chunk = 1):
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
        if stop:
            empty_queue(audio_queue)
            break
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
            _, transcription, translation, next_begin = transcription_data
            print("process:   transcription = ", transcription)
            print("process:   audio_duration = ", audio_length/16000, ", process_time = ", time.time() - s)
            if transcription:
                await manage_subtitle(websocket, translation, audio_length)
        print("process:   next_begin = ",next_begin)
        if next_begin is not None: audio = audio[np.floor(next_begin*16000).astype(int):]
        else: audio = np.array([], dtype= np.float32)

    

def process_thread(websocket, min_chunk=1.5):
    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(process(websocket, min_chunk))
    loop.close()



async def recognize(websocket, path):
    global audio_queue
    global audio_queue_length
    global pool
    global packet_length
    global processor
    global start_time
    global stop

    audio_queue_length = 0
    start_time = 0
    stop = False
    logging.info('Connection from %s', websocket.remote_address)
    min_chunk = 4
    thread2 = threading.Thread(target=process_thread, args=(websocket, min_chunk))
    thread2.start()
    while True:
        # print("waiting to recieve  ....")
        message = await websocket.recv()
        # print("message recieved ....")
        if isinstance(message, str) and 'config' in message:
            jobj = json.loads(message)['config']
            logging.info("Config %s", jobj)
            langs = []
            for sub in jobj:
                if jobj[sub] != 'None':
                    langs.append(lang_helper.STREAM_SUPPURTED_LANGUAGES_FLORES_200[jobj[sub]])
            processor.langs = langs.copy()
            print("recognize: ", langs)
            continue

        if isinstance(message, str) and "eof" in message:
            stop = True
            with audio_queue_lock:
                audio_queue_length -= len(temp)
            break

        temp = np.frombuffer(message, dtype=np.int16).astype(np.float32)/2**15
        audio_queue.put_nowait(temp)
        audio_queue_length += len(temp)
        print("recognize: audio_queue_length = ", audio_queue_length/16000)
        if stop: break



async def start():
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

    print("initialing ...")

    processor = online_processor()
    processor.initialize()

    audio_queue = asyncio.Queue()

    pool = concurrent.futures.ThreadPoolExecutor((os.cpu_count() or 1))

    print("\n \n ------ models loaded ----- \n \n")
    async with websockets.serve(recognize, args.interface, args.port):
        if args.open_browser:
                open_url = 'file://' + os.getcwd() + '/stream_client.html'
                webbrowser.open_new_tab(open_url)
        await asyncio.Future()


if __name__ == '__main__':
    global args

    parser = argparse.ArgumentParser()

    # Retrieve environment variables as defaults if set, or use explicit defaults otherwise
    default_interface = os.environ.get('SERVER_INTERFACE', '0.0.0.0')
    default_port = int(os.environ.get('SERVER_PORT', 2800))
    default_open_browser = os.environ.get('OPEN_BROWSER', 'True').lower() in ['true', '1', 'yes']

    # Define command-line arguments
    parser.add_argument('--interface', default=default_interface, help="Server interface to bind to")
    parser.add_argument('--port', type=int, default=default_port, help="Port number for the server")
    parser.add_argument('--open_browser', type=lambda x: x.lower() in ['true', '1', 'yes'], default=default_open_browser, help="Open browser if set to True")

    # Parse arguments
    args = parser.parse_args()

    asyncio.run(start())
