import os
import sys
from pathlib import Path
import torch
import nltk
from typing import Union, List, Tuple
import fasttext
import downloader
import text_processor_helper as helper
import sentencepiece as spm
import ctranslate2



device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Tokenizer():
    def __init__(self, model_apth: Union[str, Path] = "auto", load_model_flag: bool = True) -> None:
        self.NLTK_LANGUAGE_CODES = {key: value for keys, value in helper.NLTK_LANGUAGE_CODES_GROUPED.items() for key in keys}
        self.nltk_path = Path(Path.cwd() / ".cache" / "nltk")
        if load_model_flag:
            self.load_model(model_apth)

    def get_nltk_language_code(self, input_code: str):
        # Use input_code as is if it exists in the dictionary
        return self.NLTK_LANGUAGE_CODES.get(input_code.lower(), "English")
    
    def load_model(self, nltk_path: Union[str, Path] = 'auto') -> None:
        if nltk_path != 'auto' and isinstance(nltk_path, str):
            self.nltk_path = Path(nltk_path)
        elif nltk_path != "auto": self.nltk_path = nltk_path
        os.makedirs(self.nltk_path, exist_ok=True)
        os.environ["NLTK_DATA"] = str(self.nltk_path.resolve())

        if not getattr(sys, 'frozen', False) and not hasattr(sys, '_MEIPASS'):
            # load nltk sentence splitting dependency
            if not Path(self.nltk_path / "tokenizers" / "punkt").is_dir() or not Path(
                    self.nltk_path / "tokenizers" / "punkt" / "english.pickle").is_file():
                nltk.download('punkt', download_dir=str(self.nltk_path.resolve()))
                
    def split_to_sentence(self, text, language='english') -> List[str]:
        nltk_sentence_split_lang = self.get_nltk_language_code(language).lower()
        return nltk.tokenize.sent_tokenize(text, language=nltk_sentence_split_lang)

class LID():
    def __init__(self, model_path: Union[str, Path] = "auto", load_model_flag: bool = True) -> None:
        if model_path == "auto": self.model_path = Path(Path.cwd() / ".cache" / "lid")
        else: 
            self.model_path = self.model_path = model_path
        self.model = None
        if load_model_flag:
            self.load_model(model_path)

    def load_model(self, model_path: Union[str, Path] = "auto"):
        if model_path != "auto" and isinstance(model_path, str):
            self.model_path = Path(model_path)
        elif model_path != "auto": self.model_path = model_path
        os.makedirs(self.model_path, exist_ok=True)
        pretrained_lang_model_file = Path(self.model_path / "lid218e.bin")
        if not pretrained_lang_model_file.is_file():
            print(f"Downloading LID (language identification) model...")
            downloader.download_extract(helper.LID_MODEL_LINKS["lid218e"]["urls"], str(self.model_path.resolve()), self.MODEL_LINKS["lid218e"]["checksum"], "language identification")
        self.model = fasttext.load_model(str(pretrained_lang_model_file.resolve()))

    def classify(self, text: str):
        text = text.replace("\n", " ")
        predictions = self.model.predict(text, k=1)
        return predictions[0][0].replace('__label__', '')

class Translator():
    def __init__(self, size = "small", compute_type = "float32", 
                 translator_path: Union[str, Path]= "auto", 
                 sentencepiece_path: Union[str, Path]= "auto", 
                 sentence_splitor: Union[str, Tokenizer] = "auto", 
                 language_identifire: Union[str, LID] = "auto") -> None:
        self.translator_path = Path(Path.cwd() / ".cache" / "nllb200_ct2")
        self.sentencepiece_path = Path(self.translator_path / "flores200_sacrebleu_tokenizer_spm.model")
        self.translator_path = Path(self.translator_path / size)
        self.sentencepiece = None
        self.translator = None
        if isinstance(sentence_splitor, Tokenizer): self.sentence_splitor = sentence_splitor 
        else: self.sentence_splitor = Tokenizer(sentence_splitor)
        if isinstance(language_identifire, LID): self.LID = language_identifire 
        else: self.LID = LID(language_identifire)
    
    def load_model(self, size = "small", compute_type = "float32", 
                   translator_path: Union[str, Path]= "auto", 
                 sentencepiece_path: Union[str, Path]= "auto"):
        if translator_path == "auto":
            self.translator_path = Path(Path.cwd() / ".cache" / "nllb200_ct2" / size)
        elif isinstance(translator_path, str): self.translator_path = Path(translator_path)
        else: self.translator_path = translator_path
        if sentencepiece_path == "auto": Path(self.translator_path.parent / "flores200_sacrebleu_tokenizer_spm.model")
        elif isinstance(sentencepiece_path, str): self.sentencepiece_path = Path(sentencepiece_path)
        else: self.sentencepiece_path = sentencepiece_path

        os.makedirs(self.translator_path.parent, exist_ok=True)

        print(f"NLLB-200_CTranslate2 {size} is Loading to {device} using {compute_type} precision...")

        pretrained_lang_model_file = Path(self.translator_path / "model.bin")

        if not self.translator_path.exists() or not pretrained_lang_model_file.is_file():
            print(f"Downloading {size} NLLB-200 model...")
            downloader.download_extract(helper.TRANSLATOR_MODEL_LINKS[size]["urls"], str(self.translator_path.parent.resolve()), helper.TRANSLATOR_MODEL_LINKS[size]["checksum"], title="Text Translation (NLLB200CT2)")

        if not self.sentencepiece_path.is_file():
            print(f"Downloading sentencepiece model...")
            downloader.download_extract(helper.TRANSLATOR_MODEL_LINKS["sentencepiece"]["urls"], str(self.translator_path.parent.resolve()), helper.TRANSLATOR_MODEL_LINKS["sentencepiece"]["checksum"], title="Text Translation (Sentencepiece)")

        self.sentencepiece = spm.SentencePieceProcessor()
        self.sentencepiece.load(str(self.sentencepiece_path.resolve()))


        self.translator = ctranslate2.Translator(str(self.translator_path.resolve()), device=device, compute_type=compute_type)
        print(f"NLLB-200_CTranslate2 model loaded.")


    def translate(self, text: str, tar_lang: Union[str, List[str]], src_lang: str = "auto", split_sentence = True, as_iso1=False):
        # print("tranlator: tar_lang = ", tar_lang)
        if as_iso1 and src_lang in helper.LANGUAGES_ISO1_TO_ISO3:
            src_lang = helper.LANGUAGES_ISO1_TO_ISO3[src_lang][0]
        if isinstance(tar_lang, str): tar_lang = [tar_lang]
        N_lang = len(tar_lang)
        if as_iso1:
            for i in range(N_lang):
                if tar_lang[i] in helper.LANGUAGES_ISO1_TO_ISO3:
                    tar_lang[i] = helper.LANGUAGES_ISO1_TO_ISO3[tar_lang[i]][0]
        if src_lang == "auto":
            src_lang = self.LID.classify(text)

        language_unsupported = False
        if src_lang not in helper.TRANSLATOR_SUPPORTED_LANGUAGES:
            print(f"error translating. {src_lang} not supported.")
            language_unsupported = True
        # print("target_lang == ", tar_lang, ", supported == ", tar_lang in SUPPORTED_LANGUAGES, "\n")
        for i in range(N_lang):
            if tar_lang[i] not in helper.TRANSLATOR_SUPPORTED_LANGUAGES:
                print(f"error translating. {tar_lang[i]} not supported.")
                language_unsupported = True
                break

        if language_unsupported:
            return dict(), src_lang
        
        translation_multilang = {lang: None for lang in tar_lang}

        if src_lang in tar_lang:
            translation_multilang[src_lang] = text
            tar_lang.remove(src_lang)
            N_lang = len(tar_lang)

        if len(tar_lang) == 0:
            return translation_multilang, src_lang

        # Split the source text into sentences
        # print("sentence_split .. \n")
        if split_sentence:
            sentences = self.sentence_splitor.split_to_sentence(text, language=src_lang)
        else: sentences = [text]
        subwords = []
        for sentence in sentences:
            # Tokenize the source text
            # print("encode .. \n")
            # print("in translator :")
            # print(sentence)
            source_text_subworded = self.sentencepiece.encode([sentence], out_type=str)[0] + ["</s>", src_lang]
            subwords.append(source_text_subworded)
        for i in range(N_lang):
            translated_sentences = []
            # Add target language code as the target prefix
            source_sents_target_prefix = [[tar_lang[i]]]
            for source_text_subworded in subwords:
                translated_tokens = self.translator.translate_batch([source_text_subworded], batch_type="tokens", max_batch_size=2024, target_prefix=source_sents_target_prefix, beam_size=4)
                translated_tokens = [NLLBtranslation[0]['tokens'] for NLLBtranslation in translated_tokens]
                for translation in translated_tokens:
                    if tar_lang[i] in translation:
                        translation.remove(tar_lang[i])
                
                # print("decode .. \n")
                translated_sentence = self.sentencepiece.decode(translated_tokens[0])
                translated_sentences.append(translated_sentence)

            # Join the translated sentences into a single text
            translation = ' '.join(translated_sentences)
            translation_multilang[tar_lang[i]]= translation

        return translation_multilang, src_lang
