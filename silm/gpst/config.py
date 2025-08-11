from transformers import PretrainedConfig
#print("conf1")

class GPSTConfig(PretrainedConfig):
    model_type = "gpst"
    
    def __init__(self, r2d2=None, gpt=None, **kwargs):#, gptconfig, r2d2config, **kwargs):

        self.gptconfig = gpt
        self.r2d2config = r2d2
        super().__init__(**kwargs)

#print("conf2")
