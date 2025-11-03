import torch

class SpeakerManager():
    def __init__(self, speaker_file_path=None):
        self.speakers = torch.load(speaker_file_path)

    @property
    def name_to_id(self):
        return self.speakers.keys()
    
    @property
    def num_speakers(self):
        return len(self.name_to_id)
    
    @property
    def speaker_names(self):
        return list(self.name_to_id.keys())
    

class LanguageManager():
    def __init__(self, config):
        # config may be a dict (loaded JSON) or a Coqpit/XttsConfig object.
        # Support both forms to be robust when called from notebooks or scripts.
        try:
            # prefer attribute access when available
            self.langs = getattr(config, "languages")
            if self.langs is None:
                raise AttributeError()
        except Exception:
            # fallback to dict-like access
            self.langs = config["languages"]

    @property
    def name_to_id(self):
        return self.langs
    
    @property
    def num_languages(self):
        return len(self.name_to_id)
    
    @property
    def language_names(self):
        return list(self.name_to_id)
