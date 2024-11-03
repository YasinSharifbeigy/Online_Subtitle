import torch
import numpy as np
import faster_whisper
import whisper
from pathlib import Path
import os
from typing import Union, Optional
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, GenerationConfig
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from silero_vad import get_speech_timestamps, collect_chunks
from speechbrain.inference.classifiers import EncoderClassifier
import whisperx
import torchaudio



device = 'cuda' if torch.cuda.is_available() else 'cpu'



class VAD():
    def __init__(self, load_model_flag= False, model = None, vad_thread_num= 1):
        """
        load_model_flag: boolian variable. default value = False. If et True it loads model usinf load_model() function.
        model: If you have another loaded model you can pass it to Class using this. Default value = None. If you set thisparameter It doesn't load model anymore.
        vad_thread_num: It specify number of threads for torch. default value = 1.
        """
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
        """
        It will download models to ".cache/silero-vad-v5" or load it from here. 
        It doesn't have any input argument.
        """
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
        """
        Check if model is loaded or not! -> bool
        """
        if self.model is None:
            return False
        return True



    def get_confidence(self, audio_data, sr= 16000):
        """
        It returns confidence(a number in [0,1]) which shows How surely we can say there is non silence parts in the input audio.
        audio_data: input audio. Input audio must be in torch.tensor() type, single channel and It's dtype should be float16 or float32.
        sr: sample rate of the audio_data, by default is set to 16000.
        """
        confdences = self.model.audio_forward(audio_data, sr=sr)
        return confdences.squeeze().mean() #a number representing confidence for all of audio
    


    def get_timestamps(self, audio_data):
        """
        audio_data: input audio. It's type can be np.numpy.ndarray() or either torch.tensor() - It must be single channel.
        output: audio_timestamps. List of dictionaries which contains start and stop of non silence parts.
        """
        speech_timestamps = get_speech_timestamps(audio_data, self.model)
        return speech_timestamps #dict : [{'start':, 'end':}, ..] if audio is not silence else []
    


    def cut_silence(self, audio_data, speech_timestamps= None):
        """
        audio_data: input audio. It's type can be np.numpy.ndarray() or either torch.tensor() - It must be single channel.
        speech_timestamps: List of dictionaries which contains start and stop of non silence parts. Set this to None to get timestamps automaticly. default value = None.
        
        Output: An array in type of nput array which doesn't have silence part!
        """
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
        """
        load_model_flag: boolian variable. default value = False. If et True it loads model usinf load_model() function.
        model: If you have another loaded model you can pass it to Class using this. Default value = None. If you set thisparameter It doesn't load model anymore.
        """
        self.model = None 
        self.cache_vad_path = Path(Path.cwd() / ".cache" / "lid")
        if model is not None:
            self.model = model
        elif load_model_flag:
            self.load_model()
        self.razi_ind2langs = {3: 'ar', 20: 'en', 25: 'fa'}
        self.razi_langs2ind = {'ar':3, 'en':20, 'fa':25}



    def load_model(self):
        """
        It will download models to ".cache/lid" or load it from here. 
        It doesn't have any input argument.
        """
        try:
            self.model = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir=".cache/lid/lang-id-voxlingua107-ecapa", run_opts={"device":device})
        except Exception as e:
            print("Error loading lid model.")
            print(e)



    def predict_language(self, audio_data, langs = None, format = 'ISO'):
        """
        audio_data: must be torch.tensor(), single channle with 16000 samplerate
        langs : If set to None (default value) it will return language of the the audio or else It can be list of ISO or FLORES 200 type languages which predict the language between this list or it can be set to 'razi' which select language between Persian, English and Arabic.
        format : language format of languages in langs. default value = 'Iso' 
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
        """
        model: If you have loaded model you can pass it to Class using this. Default value = None.
        to load model use load_model() function.
        """
        self.model = model
        self.aligning_models = dict()
        self.utils = None
        self.processor = None
        self.pipe = None
        self.model_size = None
        self.model_type = None
        self.local = False
        self.cache_path = ".cache" + "/whisper/"
        self.aligning_model_path =  ".cache" + "/wav2vec/"
        self.device = device



    def load_model(self, model_type = None, fine_tuned = None, model_size = None, model_path = None, whisper_config_path = None, **kwargs):
        """
        model_type : Type of model that want to be loaded. Supported model_types are 'whisper-transformers', 'whisper-openai', 'faster-whisper' and 'whisperX'.It always must be set unless model_path is in STT_MODELs_LOCAL_PATH.
        fine_tuned : Boolian argument. for each model_type we have some fine_tuned models. If you want to use these models set it True.
        model_size : Size of whisper_model that want to be loaded. Supported model_sizes are 'tiny', 'base', 'small', 'medium', 'large', 'large-v2' and 'large-v3'.
        model_path : str() type. If there is a local model that want to loaded you can pass it to this argument. If you set this, there is no need to set model_size and fine_tuned.
        whisper_config_path : If you are using whisper-transformers type you can set generation_config of model by loading cofig from this path.
        kwargs : If you are using whisper-transformers type you can set other loading parameters and If usinf whisperX and want to set language and device to load aligning model.
        """
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
                # os.system("./start.sh")
                path_or_size = self.cache_path if self.local else self.model_size
                self.model = faster_whisper.WhisperModel(path_or_size, device= device, 
                                                         local_files_only= self.local, download_root= self.cache_path)
            elif self.model_type == "whisperX":
                path_or_size = self.cache_path if self.local else self.model_size
                try:
                    self.model = whisperx.load_model(path_or_size, device= 'cuda', compute_type= "float16", asr_options= {"hotwords": None})
                except:
                    self.model = whisperx.load_model(path_or_size, device= 'cuda', compute_type= "float32", asr_options= {"hotwords": None})
                if 'language' in kwargs:
                    if 'device' in kwargs: self.device = kwargs['device']
                    if isinstance(kwargs['language'], str):
                        language = kwargs['language']
                        self.load_align_model(language_code= language, device= self.device, model_dir= self.aligning_model_path)
                    else:
                        for language in kwargs['language']:
                            self.load_align_model(language_code= language, device= self.device, model_dir= self.aligning_model_path)
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



    def load_align_model(self, language_code, device = device, model_name=None, model_dir=None):
        """
        It Loads aligning model for whisperX type models.
        language_code: ISO language format.
        device: 'cuda' or 'cpu'. default value set automaticly based on hardware capabilities.
        model_name: wavw2vec model_name you want to be loaded. It must be in torchaudio available models or be available in huggingface. If It's set to None, It will be set automaticly. default value = None.
        model_dir: str() where to save and load model. If it's set to None model will be saved in '.cache/wav2vec' diractory. Default value = None.
        """
        if model_name is None:
            # Use default model
            if language_code in whisperx.alignment.DEFAULT_ALIGN_MODELS_TORCH:
                model_name = whisperx.alignment.DEFAULT_ALIGN_MODELS_TORCH[language_code]
            elif language_code in whisperx.alignment.DEFAULT_ALIGN_MODELS_HF:
                model_name = whisperx.alignment.DEFAULT_ALIGN_MODELS_HF[language_code]
            else:
                print(f"There is no default alignment model set for this language ({language_code}).\
                    Please find a wav2vec2.0 model finetuned on this language in https://huggingface.co/models, then pass the model name in --align_model [MODEL_NAME]")
                raise ValueError(f"No default align-model for language: {language_code}")

        # Ensure the model_dir exists
        if model_dir is not None:
            if model_name not in model_dir:
                model_dir = os.path.join(model_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
        else:
            model_dir = f".cache/wav2vec/{model_name}"
            os.makedirs(model_dir, exist_ok=True)
        # For torchaudio models
        if model_name in torchaudio.pipelines.__all__:
            pipeline_type = "torchaudio"
            bundle = torchaudio.pipelines.__dict__[model_name]
            
            # Download model to the specific directory, if provided
            align_model = bundle.get_model(dl_kwargs={"model_dir": model_dir}).to(device)
            labels = bundle.get_labels()
            align_dictionary = {c.lower(): i for i, c in enumerate(labels)}
        
        # For HuggingFace Wav2Vec2 models
        else:
            try:
                    processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=model_dir)
                    align_model = Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=model_dir)

            except Exception as e:
                print(e)
                print(f"Error loading model from Huggingface, check https://huggingface.co/models for finetuned wav2vec2.0 models")
                raise ValueError(f'The chosen align_model "{model_name}" could not be found in Huggingface (https://huggingface.co/models) or torchaudio (https://pytorch.org/audio/stable/pipelines.html#id14)')
            
            pipeline_type = "huggingface"
            align_model = align_model.to(device)
            labels = processor.tokenizer.get_vocab()
            align_dictionary = {char.lower(): code for char, code in labels.items()}

        align_metadata = {"language": language_code, "dictionary": align_dictionary, "type": pipeline_type}

        self.aligning_models[language_code] = (align_model, align_metadata)



    def reset_pipe(self):
        """
        This method build and rebuld the pipline If you are using whisper-transformers model type. -> None
        """
        self.pipe = pipeline("automatic-speech-recognition",
                        model= self.model,
                        tokenizer=self.processor.tokenizer,
                        feature_extractor=self.processor.feature_extractor,
                        torch_dtype=torch.float16,
                        device= self.device)
    


    def transcribe(self, audio_data: Union[str, np.ndarray, torch.Tensor], return_word_timestamps: bool = False, return_like_model_type: bool = False, **kwargs):
        """
        Function for transcribing the audio.
        audio_data: Path to audio (.wav or .mp3) in type of str() ar array representing data in type of np.ndarray() or torch.tensor()
        return_word_timestamps: Boolian parameter. default value = None. if you want to return word timestamps set it to True.
        kwargs: other arguments for transcribing for each model_type.

        Output format:
        if return_like_model_type:
            if: model_type == 'whisper-transformers' and return_word_timestamps == True:
                result = {'text': , 'chunks': [{'text': , 'timestamp': (start, end)}, ...]}
            if: model_type == 'whisper-transformers' and return_word_timestamps == False:
                result = Text
            if: model_type == 'whisperX' and return_word_timestamps == False:
                result = {'segments': [{'text': , 'start': , 'end': , 'clean_char': [], 'clean_cdx': [], 'clean_wdx': [], 'sentence_spans': [(,), ...]}, ...], 'language': }
            if: model_type == 'whisperX' and return_word_timestamps == True:
                result = (result of whisperX model which is looks like above, whisperX alignment model result)
                alignment model result : {'segments': [{'start': , 'end': , 'text': '' , 'words': [{'word': '', 'start': , 'end': , 'score': }, ...]}
            if: model_type == 'whisper-openai':
                result = {'text':, 'segments': [{'id':, 'start':, 'end':, 'text':, 'tokens':[], 'words':[{'word':, 'start':, 'end':, 'probablity':}, ...]}, ...]}
            if: model_type == 'faster-whisper':
                result = [Segment(id=, start=, end=, text= '', tokens = [], ..., words=[Word(start= , end= , word='', probability=), ...]), ...]
        else:
            if return_word_timestamps == True:
                result = {'text': " ", 'words'= [{'word': , 'start' , 'end': }, ...]}
            else:
                result = {'text': " "}
        """
        result = dict()
        if self.model_type == 'whisper-transformers':
            if return_word_timestamps:
                if self.pipe is None: self.reset_pipe()
                if isinstance(audio_data, torch.Tensor): audio_data = np.array(audio_data)
                model_result = self.pipe(audio_data, return_timestamps = "word", generate_kwargs = kwargs)
                if return_like_model_type:
                    result = model_result
                else:
                    result["text"] = model_result["text"]
                    result["words"] = []
                    for chunk in model_result["chunks"]:
                        result["words"].append({"word": chunk["text"], 
                                                "start": chunk["timestamp"][0], 
                                                "end": chunk["timestamp"][1]})
            else:
                if isinstance(audio_data, str):
                    data, fs = librosa.load(audio_data, sr= 16000, dtype= np.float32, mono= True)
                with torch.no_grad():
                    input_features = self.processor(data, return_tensors='pt', sampling_rate=16000).input_features.to(self.device)
                    output = self.model.generate(input_features, **kwargs)
                    model_result = self.processor.decode(output[0].squeeze(), skip_special_tokens=True, normalize=True) # text
                if return_like_model_type:
                    result = model_result
                else:
                    result["text"] = model_result["text"]
        elif self.model_type == 'whisper-openai':
            model_result = self.model.transcribe(audio_data, word_timestamps = return_word_timestamps, **kwargs) #for faster whisper BinaryIO is acceptable either
            if return_like_model_type:
                result = model_result
            else:
                result["text"] = model_result["text"]
                if return_word_timestamps:
                    result['words'] = []
                    for segment in model_result['segments']:
                        for word in segment['words']:
                            result['words'].append({'word': word['word'], 
                                                    'start': word['start'], 
                                                    'end': word['end']})
        elif self.model_type == 'whisperX':
            model_result = self.model.transcribe(audio_data, **kwargs)
            if return_like_model_type:
                result = model_result
            else:
                result['text'] = ""
                for segment in model_result['segments']:
                    result['text'] += segment['text']
            if return_word_timestamps:
                language = model_result["language"]
                if language not in self.aligning_models:
                    self.load_align_model(language_code = language, device= self.device, model_dir= self.aligning_model_path)
                result_alignment = whisperx.align(model_result["segments"], self.aligning_models[language][0], 
                                        self.aligning_models[language][1], audio_data, 
                                        self.device, return_char_alignments=False)
                if return_like_model_type:
                    result = (result, result_alignment)
                else:
                    result['words'] = []
                    for segment in result_alignment['segments']:
                        for word in segment['words']:
                            result['words'].append({'word': word['word'], 
                                                    'start': word['start'], 
                                                    'end': word['end']})
        else:
            model_result, _ = self.model.transcribe(audio_data, word_timestamps = return_word_timestamps, **kwargs) #for faster whisper BinaryIO is acceptable either
            model_result = list(model_result)
            if return_like_model_type:
                result = model_result
            else:
                result['text'] = ""
                for segment in model_result:
                    result['text'] += segment.text
                if return_word_timestamps:
                    result['words'] = []
                    for segment in model_result:
                        for word in segment.words:
                            result['words'].append({'word': word.word, 
                                                    'start': word.start, 
                                                    'end': word.end})
        return result

