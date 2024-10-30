from audio_processor import STT_model, LID, VAD
from text_processor import Translator
import text_processor_helper as helper
import numpy as np
import librosa
import torch
import time 




class online_processor():
    def __init__(self):
        """
        Manage models and handle processes in each iteration for live subtitles.
        """
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
        """
        It will loads models and Initialize them for better performance.
        """
        #load models
        print("initialization ...")
        self.STT_model.load_model('faster-whisper', False, 'medium')
        self.STT_model_persian.load_model('whisperX', model_path= ".cache/whisper/medium-fine_tuned-ct2", **{'language' : "fa"})
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
        result = self.STT_model_persian.transcribe(audio_test_persian, True, **{"language": "fa"})
        print("per_model: ", time.time()-s)
        _, _ = self.Translator_model.translate(result[0]['segments'][0]['text'], self.langs)
        print("initialization done!\n")



    def pre_process_iter(self, audio):
        """
        Take audio chunk as input  and do some process on it to make Server ready for other processes.
        """
        speech_timestamps = self.VAD_model.get_timestamps(audio)
        print("preprocess:", speech_timestamps)
        return speech_timestamps



    def process_iter(self, audio, voice_activity, lid_min_pob= 0.4, max_nonactivity_threshold = 0.5, max_audio_length = 8, min_audio_length = 1):
        """
        Main processes for each iteration.
        auido: An np.ndarray() representing audio chunks.(sigle channel with samplerate of 16k)
        voice_activity: Timestamps of Voice activity. List of dictionaries like {'start': , 'end': }
        lid_min_prob: Minimum probability to accept LID output. Default value = 0.4.
        max_nonactivity_threshold: If last voice avtivity ends sooner than this parameter We consider it end of sentence and don't use Aggregation mechanisms. Default value = 0.5.
        min_audio_length: If last voice activity is so close to the end of audio chunk and It's duration is less than this parameter, we discard process of last voice activity and do process with next audio chunk. Default value = 1.
        
        Output:
        src_lang: Detected language of audio chunk in flores 200 format. (output of LID)
        transcription: transcription of input audio (selected parts of audio). (output of STT)
        translation: A dictionary which contains translations based on user requested languages. (output of translator)
        next_begin: start of next audio chunk.
        """
        translation = dict()
        transcription = ""
        src_lang = helper.STREAM_SUPPURTED_LANGUAGES_FLORES_200[helper.ISO_LANGUAGES[self.last_lang]]
        audio_dur = len(audio)/16000

        if audio_dur - voice_activity[-1]['end']/16000 > max_nonactivity_threshold:
            # audio = self.VAD_model.cut_silence(audio, voice_activity)
            do_del_last_words = False
            next_begin = None
        # elif len(voice_activity) > 1:
            # audio = self.VAD_model.cut_silence(audio, voice_activity[:-1])
        elif voice_activity[-1]['end'] - voice_activity[-1]['start'] <= min_audio_length * 16000:
            if len(voice_activity) == 1:
                next_begin = voice_activity[-1]['start'] - 0.1 if voice_activity[-1]['start'] >= 0.1 else 0
                return src_lang, transcription, translation, next_begin
            next_begin = voice_activity[-1]['start']/2 + voice_activity[-2]['end']/2 if voice_activity[-1]['start'] - voice_activity[-2]['end'] < 0.5 else voice_activity[-1]['start'] - 0.1
            audio = audio[:voice_activity[-2]['end']]
            do_del_last_words = False
        else:
            do_del_last_words = True
        
        lang, lang_prob = self.LID_model.predict_language(torch.from_numpy(audio), langs= 'razi') 
        if lang_prob > lid_min_pob: 
            new_lang = lang.split(": ")[0]
            if self.last_lang != new_lang: self.textbuffer = ""
            self.last_lang = new_lang
        

        if self.last_lang == 'fa':
            STT_out = self.STT_model_persian.transcribe(audio, True, **{"language": "fa"})
            STT_model_type = self.STT_model_persian.model_type
            # print("process_on:STT_OUT:  ", STT_out)
            # print("process_on:STT_OUT:  ", STT_out[0]['segments'][0]['text'])
            # print("process_on:STT_OUT:  ", STT_out[1])
        else:
            STT_out = self.STT_model.transcribe(audio, True, **{"language": self.last_lang})
            STT_model_type = self.STT_model.model_type
        src_lang = helper.STREAM_SUPPURTED_LANGUAGES_FLORES_200[helper.ISO_LANGUAGES[self.last_lang]]
        
        if do_del_last_words:
            transcription , _, next_begin = del_last_words(STT_out, 1, STT_model_type)
        else: transcription = STT_output_to_text(STT_out, STT_model_type)

        if transcription:
            # transcription_senteces = self.Translator_model.sentence_splitor.split_to_sentence(self.textbuffer + transcription, src_lang)
            # self.textbuffer = transcription_senteces[-1]
            print("process_on:src_lang:  ", src_lang)
            print("process_on:tar_lang:  ", self.langs)
            translation, _ = self.Translator_model.translate(transcription, self.langs.copy(), src_lang)
            
        return src_lang, transcription, translation, next_begin






def STT_output_to_text(STT_out, model_name = "faster_whisper"):
    """
    Takes STT model's output and return transcription based on model_name -> str() 
    model_name: model type used for STT. must be chosen between faster_whisper, whisperX and openai.
    """
    if "faster" in model_name:
        text = ""
        for segment in STT_out:
            text += segment.text
    elif model_name == "whisperX":
        text = ""
        if len(STT_out[0]['segments']):
            text = STT_out[0]['segments'][0]['text']
    elif model_name == "openai":
        text = STT_out['text']
    return text



def del_last_words(STT_out, n = 2, model_name = "faster_whisper"):
    """
    Takes STT model's output delete last n words of it.
    n: number words that must be deleted.
    model_name: model type used for STT. must be chosen between faster_whisper, whisperX and openai.

    Output:
    text: Part of transcription that want to be kept.
    deleted_text: Deleted Part of transcription.
    end: end time of kept part of transcription.
    """
    if "faster" in model_name:
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
    elif model_name == "whisperX":
        text = text_temp = ""
        if not len(STT_out[0]['segments']):
            text = text_temp = ""
            end = None
        else:
            text_temp = STT_out[0]['segments'][0]['text']
            end = 0
            words = []
            deleted_text = ""
            for segment in STT_out[1]['segments']:
                words += segment['words']
            if text_temp: deleted_text = segment['text']
            num_words = len(words)
            if(num_words == 0): end = None
            elif(num_words > n):
                text = text_temp.split(" ")
                deleted_text = " ".join(text[-n:])
                text = " ".join(text[:-n])
                end = words[-n-1]['end'] /2 + words[-n]['start']/2
    elif model_name == "openai":
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


