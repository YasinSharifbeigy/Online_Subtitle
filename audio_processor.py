import torch
import numpy as np
import faster_whisper
import whisper
from pathlib import Path
import os
from typing import Union, Optional
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, GenerationConfig
from silero_vad import get_speech_timestamps, collect_chunks
from speechbrain.inference.classifiers import EncoderClassifier
import whisperx

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class VAD():
    def __init__(self, load_model_flag= False, model = None, vad_thread_num= 1):
        self.model = None 
        self.utils = None
        torch.set_num_threads(vad_thread_num)
        self.cache_vad_path = Path(Path.cwd() / ".cache" / "silero-vad-v5")
        torch.hub.set_dir(str(Path(self.cache_vad_path).resolve()))
        if model is not None:
            self.model = model
        elif load_model_flag:
            self.load_model()

    def load_model(self):
        try:
            self.model, self.utils = torch.hub.load(trust_repo=True, skip_validation=True,
                                                            repo_or_dir="snakers4/silero-vad", model="silero_vad", onnx=False)
        except:
            try:
                self.model, self.utils = torch.hub.load(trust_repo=True, skip_validation=True,
                                                                source="local", model="silero_vad", onnx=False,
                                                                repo_or_dir=str(Path(
                                                                self.cache_vad_path / "snakers4_silero-vad_master").resolve()))
            except Exception as e:
                print("Error loading vad model.")
                print(e)

    def is_loaded(self):
        if self.model is None:
            return False
        return True

    def get_confidence(self, audio_data, sr= 16000):
        """
        audio_data: torch.tensor - single channel, dtype = float16 or float32
        """
        confdences = self.model.audio_forward(audio_data, sr=sr)
        return confdences.squeeze().mean() #a number representing confidence for all of audio
    
    def get_timestamps(self, audio_data):
        """
        audio_data: np.numpy or torch.tensor - single channel
        """
        speech_timestamps = get_speech_timestamps(audio_data, self.model)
        return speech_timestamps #dict : [{'start':, 'end':}, ..] if audio is not silence else []
    
    def cut_silence(self, audio_data, speech_timestamps= None):
        if speech_timestamps is not None:
            output_audio = collect_chunks(speech_timestamps, audio_data)
        else:
            speech_timestamps = get_speech_timestamps(audio_data, self.model)
            if speech_timestamps:
                output_audio = collect_chunks(speech_timestamps, audio_data)
            else: output_audio = audio_data[0:0]
        return output_audio #same as input type
    

class LID():
    def __init__(self, load_model_flag = False, model = None):
        self.model = None 
        self.cache_vad_path = Path(Path.cwd() / ".cache" / "lid")
        if model is not None:
            self.model = model
        elif load_model_flag:
            self.load_model()
        self.razi_ind2langs = {3: 'ar', 20: 'en', 25: 'fa'}
        self.razi_langs2ind = {'ar':3, 'en':20, 'fa':25}

    def load_model(self):
        try:
            self.model = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir=".cache/lid/lang-id-voxlingua107-ecapa", run_opts={"device":device})
        except Exception as e:
            print("Error loading lid model.")
            print(e)

    def predict_language(self, audio_data, langs = None, format = 'ISO'):
        """
        audio_daya: torch.tensor single channle whth 16000 samplerate
        langs : list of ISO or FLORES 200 type languages, None(defualt) for auto, 'razi' for Razi format
        format : auto, 
        """
        prediction =  self.model.classify_batch(audio_data)
        if langs is None:
            lang, prob = (prediction[3][0], prediction[1].exp().to('cpu'))
        else:
            if langs == 'razi':
                inxs = list(self.razi_langs2ind.values())
                prediction = prediction[0][0,inxs].to('cpu')
                ind = prediction.argmax()
                prob = torch.exp(prediction[ind])
                lang = self.razi_ind2langs[inxs[ind.item()]]
        return  lang, prob # (language, probablity)

STT_SUPPORTED_MODELS = {'whisper-transformers': "", 'whisper-openai': "openai.pt", 'faster-whisper': "ct2", "whisperX": "ct2"}
WHISPER_MODEL_SIZE = ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']
STT_MODELs_LOCAL_PATH = {}
# for model_type in STT_SUPPORTED_MODELS:
#     for size in WHISPER_MODEL_SIZE:
#         model_path = f".cache/whisper/{WHISPER_MODEL_SIZE[size]}-{STT_SUPPORTED_MODELS[model_type]}"
#         STT_MODELs_LOCAL_PATH[model_path] = False
#         if size == 'medium':
#             model_path = f".cache/whisper/{WHISPER_MODEL_SIZE[size]}-fine_tuned-{STT_SUPPORTED_MODELS[model_type]}"
#             STT_MODELs_LOCAL_PATH[model_path] = False
        

class STT_model():
    def __init__(self, model = None):
        self.model = model
        self.aligning_models = dict()
        self.utils = None
        self.processor = None
        self.pipe = None
        self.model_size = None
        self.model_type = None
        self.local = False
        self.cache_path = ".cache" + "/whisper/"
        self.device = device

    def load_model(self, model_type = None, fine_tuned = None, model_size = None, model_path = None, whisper_config_path = None, **kwargs):
        self.model_type = model_type
        self.model_size = model_size
        if model_path is not None and (model_path in STT_MODELs_LOCAL_PATH or (self.cache_path + model_path) in STT_MODELs_LOCAL_PATH):
            self.cache_path = model_path if model_path in STT_MODELs_LOCAL_PATH else self.cache_path + model_path
            self.model_type = STT_MODELs_LOCAL_PATH[self.cache_path]
            self.local = True
        elif isinstance(model_path, str):
            if model_type not in STT_SUPPORTED_MODELS: raise("we don't support this type of models")
            if os.path.exists(model_path): self.cache_path = model_path
            elif os.path.exists(self.cache_path + model_path): self.cache_path += model_path
            elif self.model_type != 'whisper-transformers':
                raise("Path does not exist")
            STT_MODELs_LOCAL_PATH[self.cache_path] = model_type
            self.local = False if self.model_type == 'whisper-transformers' else True
        elif model_type in STT_SUPPORTED_MODELS and model_size in WHISPER_MODEL_SIZE and isinstance(fine_tuned, bool):
            path_fine_tuned = "-fine_tuned-" if fine_tuned else "-"
            model_path = model_size + path_fine_tuned + STT_SUPPORTED_MODELS[model_type]
            self.cache_path += model_path
            if model_type == 'whisper-transformers': model_path = f"openai/whisper-{model_size}"
            STT_MODELs_LOCAL_PATH[self.cache_path] = model_type
            self.local = False
            if  model_type == 'faster-whisper' and os.path.exists(self.cache_path + "/model.bin"): self.local = True
            if  model_type == 'whisper-openai' and os.path.exists(self.cache_path): self.local = True
        else:
            raise("Specified model is not supported")
        # print(self.local, self.model_size, self.model_type, self.cache_path, fine_tuned, STT_MODELs_LOCAL_PATH)
        try:
            if self.model_type == "faster-whisper":
                os.system("./start.sh")
                path_or_size = self.cache_path if self.local else self.model_size
                self.model = faster_whisper.WhisperModel(path_or_size, device= device, 
                                                         local_files_only= self.local, download_root= self.cache_path)
            elif self.model_type == "whisperX":
                os.system("./start.sh")
                path_or_size = self.cache_path if self.local else self.model_size
                try:
                    self.model = whisperx.load_model(".cache/whisper/medium-ct2", device= 'cuda', compute_type= "float16", asr_options= {"hotwords": None})
                except:
                    self.model = whisperx.load_model(".cache/whisper/medium-ct2", device= 'cuda', compute_type= "float32", asr_options= {"hotwords": None})
                if 'language' in kwargs:
                    if 'device' in kwargs: self.device = kwargs['device']
                    if isinstance(kwargs['language'], str):
                        language = kwargs['language']
                        self.load_aligning_model(language= language, device= self.device)
                    else:
                        for language in kwargs['language']:
                            self.load_aligning_model(language= language, device= self.device)
            elif self.model_type == "whisper-openai":
                path_or_size = self.cache_path if self.local else self.model_size
                self.model = whisper.load_model(path_or_size, device= device, download_root= self.cache_path)
            else:
                model_path = self.cache_path if self.local else model_path
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, 
                                                                low_cpu_mem_usage=True, 
                                                                use_safetensors=True, 
                                                                attn_implementation="sdpa", **kwargs).to(device)
                if whisper_config_path is not None:
                    generation_config = GenerationConfig.from_pretrained("openai/whisper-base")# if you are using a multilingual model
                    self.model.generation_config = generation_config
                self.processor = AutoProcessor.from_pretrained(model_path, **kwargs)
        except Exception as e:
            STT_MODELs_LOCAL_PATH.pop(self.cache_path)
            self.local = False
            print("Error loading STT model.")
            raise(e)
            # segment = model.transcribe('test/test_per_2s.wav',beam_size=5, language= "fa",word_timestamps=True)
            #{'text':, 'segments': [{'id':, 'start':, 'end':, 'text':, 'tokens':[], 'words':[{'word':, 'start':, 'end':, 'probablity':}, ...]}, ...]}
        #load directly from huggingface for faster_whisper, whisper_openai, whisper_transformers
    

    def load_aligning_model(self, language, device= device):
        self.aligning_models[language] = whisperx.load_align_model(language_code= language, device=device) 


    def reset_pipe(self):
        self.pipe = pipeline("automatic-speech-recognition",
                        model= self.model,
                        tokenizer=self.processor.tokenizer,
                        feature_extractor=self.processor.feature_extractor,
                        torch_dtype=torch.float16,
                        device= self.device)
    
    def transcribe(self, audio_data: Union[str, np.ndarray, torch.Tensor], return_word_timestamps: bool = False, **kwargs):
        """
        model_type == 'whisper-transformers' and return_word_timestamps == True:
            result = {'text': , 'chunks': [{'text': , 'timestamp': (start, end)}, ...]}
        model_type == 'whisper-transformers' and return_word_timestamps == False:
            result = Text
        model_type == 'whisper-openai':
            result = {'text':, 'segments': [{'id':, 'start':, 'end':, 'text':, 'tokens':[], 'words':[{'word':, 'start':, 'end':, 'probablity':}, ...]}, ...]}
        model_type == 'faster-whisper':
            result = [Segment(id=, start=, end=, text= '', tokens = [], ..., words=[Word(start= , end= , word='', probability=), ...]), ...]
        """
        if self.model_type == 'whisper-transformers':
            if return_word_timestamps:
                if self.pipe is None: self.reset_pipe()
                if isinstance(audio_data, torch.Tensor): audio_data = np.array(audio_data)
                result = self.pipe(audio_data, return_timestamps = "word", generate_kwargs = kwargs)
            else:
                if isinstance(audio_data, str):
                    data, fs = librosa.load(audio_data, sr= 16000, dtype= np.float32, mono= True)
                with torch.no_grad():
                    input_features = self.processor(data, return_tensors='pt', sampling_rate=16000).input_features.to(self.device)
                    output = self.model.generate(input_features, **kwargs)
                    result = self.processor.decode(output[0].squeeze(), skip_special_tokens=True, normalize=True) # text
        elif self.model_type == 'whisper-openai':
            result = self.model.transcribe(audio_data, word_timestamps = return_word_timestamps, **kwargs) #for faster whisper BinaryIO is acceptable either
        elif self.model_type == 'whisperX':
            result = self.model.transcribe(audio_data, **kwargs)
            if return_word_timestamps:
                language = result["language"]
                if language not in self.aligning_models:
                    self.load_aligning_model(language = language)
                result_alignment = whisperx.align(result["segments"], self.aligning_models[language][0], 
                                        self.aligning_models[language][1], audio_data, 
                                        self.device, return_char_alignments=False)
                result = (result, result_alignment)
        else:
            result, _ = self.model.transcribe(audio_data, word_timestamps = return_word_timestamps, **kwargs) #for faster whisper BinaryIO is acceptable either
            result = list(result)
        return result

class STT_process(STT_model):
    def __init__(self, use_LID, lan, LID_threshold):
        super().__init__()
        self.lan = None

