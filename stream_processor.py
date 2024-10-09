from audio_processor import STT_model, LID, VAD
from text_processor import Translator
import text_processor_helper as helper
import numpy as np
import librosa
import torch
import time 

class online_processor():
    def __init__(self):
        self.STT_model_persian = STT_model()
        self.STT_model = STT_model()
        self.Translator_model = Translator()
        self.LID_model = LID()
        self.VAD_model = VAD()
        self.audiobuffer = np.array([], dtype= np.float32)
        self.audiobuffer_offset = 0
        self.textbuffer = ""
        self.SAMPLE_RATE = 16000
        self.langs = ['pes_Arab', 'eng_Latn', 'arb_Arab']
        self.alllangs = ['pes_Arab', 'eng_Latn', 'arb_Arab']
        self.last_lang = 'fa'
    
    def initialize(self):
        #load models
        print("initialization ...")
        self.STT_model.load_model('faster-whisper', False, 'medium')
        self.STT_model_persian.load_model('faster-whisper', True, 'medium')
        self.Translator_model.load_model()
        self.LID_model.load_model()
        self.VAD_model.load_model()
        #warm up
        audio_test, _ = librosa.load("test/test_en_10s.wav", sr= self.SAMPLE_RATE, mono= True, dtype= np.float32)
        audio_test_persian, _ = librosa.load("test/test_per_10s.wav", sr= self.SAMPLE_RATE, mono= True, dtype= np.float32)
        _ = self.VAD_model.get_confidence(torch.from_numpy(audio_test), self.SAMPLE_RATE)
        _, _ = self.LID_model.predict_language(torch.from_numpy(audio_test), langs= 'razi')
        s = time.time()
        result = self.STT_model.transcribe(audio_test, True, **{"language": "en"})
        print("en_model:  ", time.time()-s)
        s = time.time()
        result = self.STT_model_persian.transcribe(audio_test_persian, True, **{"language": "en"})
        print("per_model: ", time.time()-s)
        _, _ = self.Translator_model.translate(result[0].text, self.langs)
        print("initialization done!\n")

    async def insert_audio():
        pass

    def pre_process_iter(self, audio):
        speech_timestamps = self.VAD_model.get_timestamps(audio)
        print("preprocess:", speech_timestamps)
        return speech_timestamps
    
    def process_iter(self, audio, voice_activity, lid_min_pob= 0.4, last_activity_threshold = 0.5, max_audio_length = 8, min_audio_length = 2):
        lang, lang_prob = self.LID_model.predict_language(torch.from_numpy(audio), langs= 'razi') 
        if lang_prob > lid_min_pob: 
            new_lang = lang.split(": ")[0]
            if self.last_lang != new_lang: self.textbuffer = ""
            self.last_lang = new_lang
        audio_dur = len(audio)/16000
        if audio_dur - voice_activity[-1]['end']/16000 > 0.5:
            # audio = self.VAD_model.cut_silence(audio, voice_activity)
            do_del_last_words = False
            next_begin = None
        # elif len(voice_activity) > 1:
            # audio = self.VAD_model.cut_silence(audio, voice_activity[:-1])
        elif voice_activity[-1]['end'] - voice_activity[-1]['start'] <= min_audio_length:
            audio = audio[:voice_activity[-2]['end']]
            do_del_last_words = False
            next_begin = voice_activity[-1]['start']
        else:
            do_del_last_words = True

        if self.last_lang == 'fa':
            STT_out = self.STT_model_persian.transcribe(audio, True, **{"language": "en"})
        else:
            STT_out = self.STT_model.transcribe(audio, True, **{"language": self.last_lang})
        src_lang = helper.STREAM_SUPPURTED_LANGUAGES_FLORES_200[helper.ISO_LANGUAGES[self.last_lang]]
        
        if do_del_last_words:
            transcription , _, next_begin = del_last_words(STT_out, n = 1)
        else: transcription = STT_output_to_text(STT_out)

        if transcription:
            # transcription_senteces = self.Translator_model.sentence_splitor.split_to_sentence(self.textbuffer + transcription, src_lang)
            # self.textbuffer = transcription_senteces[-1]
            # print("process_on:src_lang:  ", src_lang)
            # print("process_on:tar_lang:  ", self.langs)
            translation, _ = self.Translator_model.translate(transcription, self.langs.copy(), src_lang)
        else:
            translation = dict()
        return src_lang, transcription, translation, next_begin

def STT_output_to_text(STT_out, model_name = "medium-fine_tuned-ct2"):
    if "ct2" in model_name:
        text = ""
        for segment in STT_out:
            text += segment.text
    elif model_name == "medium-fine_tuned":
        text = STT_out['text']
    return text

def del_last_words(STT_out, n = 2, model_name = "medium-fine_tuned-ct2"):
    if "ct2" in model_name:
        text = text_temp = ""
        end = 0
        words = []
        deleted_text = ""
        for segment in STT_out:
            text_temp += segment.text
            words += segment.words
        if text_temp: deleted_text = segment.text
        num_words = len(words)
        if(num_words == 0): end = None
        elif(num_words > n):
            text = text_temp.split(" ")
            deleted_text = " ".join(text[-n:])
            text = " ".join(text[:-n])
            end = words[-n-1].end #+ segment.words[-2].start/2
    elif model_name == "medium-fine_tuned":
        text = ""
        deleted_text = STT_out['text']
        end = 0
        num_words = len(STT_out['chunks'])
        if(num_words == 0): end = None
        elif(num_words > n):
            text = deleted_text.split(" ")
            deleted_text = " ".join(text[-n:])
            text = " ".join(text[:-n])
            end =STT_out['chunks'][-n-1]['timestamp'][1] #+ segment.words[-2].start/2
    return text, deleted_text, end


