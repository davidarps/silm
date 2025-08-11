from silm.udgn import UDGN, UDGNConfig
from silm.structformer_in_parser import *
from transformers import AutoConfig, AutoModelForMaskedLM

UDGNConfig.register_for_auto_class()
UDGN.register_for_auto_class()
AutoConfig.register("udgn", UDGNConfig)
AutoModelForMaskedLM.register(UDGNConfig, UDGN)
StructFormer_In_ParserConfig.register_for_auto_class()
StructFormer_In_ParserModel.register_for_auto_class()
AutoConfig.register("structformer_in_parser", StructFormer_In_ParserConfig)
AutoModelForMaskedLM.register(StructFormer_In_ParserConfig, StructFormer_In_ParserModel)
