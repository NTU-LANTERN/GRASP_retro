import argparse
import warnings

import onmt
import onmt.inputters
import onmt.model_builder
import onmt.modules
import onmt.opts
import onmt.translate
import torch
from onmt.translate.translator import build_translator
from rdkit import Chem

from GRASP.utils.misc import tokenizer, detokenizer, canonicalize

warnings.filterwarnings("ignore")

class Mt:
    def __init__(self, model_path, vocab_path, gpu, beam_size, batch_size=64, model_type='retro'):
        self.load_model_translator = model_type
        parser = argparse.ArgumentParser(description='translate.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        onmt.opts.translate_opts(parser)
        self.gpu = gpu
        if gpu != -1:
            torch.cuda.set_device(gpu)
        parse_command = f'-model {model_path} \
                          -src None \
                          -tgt None \
                          -batch_size {batch_size} \
                          -replace_unk \
                          -max_length 200 \
                          -beam_size {beam_size} \
                          -n_best {beam_size} \
                          -gpu {gpu}'
        self.model_path = model_path
        self.opt = parser.parse_args(parse_command.split())
        self.model_translator = build_translator(self.opt, report_score=False)
        print()
        self.source_vocab = vocab_path
    
    def inference(self, src_list):
        if not self.model_translator:
            print('No single-step MT models loadad, please check loading options.')
            return None, None
        # try:
        src_list = [tokenizer(smi) for smi in src_list]
        # if target_list:
        #     target_list = [tokenizer(canonicalize(smi)) for smi in target_list]
        #     result = self.model_translator.translate(src=src_list,
        #                                              tgt=target_list,
        #                                                 batch_size=self.beam_size)
        #     scores, predictions = result
        #     scores = torch.exp(torch.Tensor(scores)).tolist()
        #     predictions = [[canonicalize(detokenizer(p)) for p in prediction] for
        #                     prediction in predictions]
        # else:
        result = self.model_translator.translate(src=src_list,
                                                batch_size=self.opt.beam_size)     
        
        scores, predictions = result
        scores = torch.exp(torch.Tensor(scores)).tolist()
        predictions = [[canonicalize(detokenizer(p)) for p in prediction] for
                        prediction in predictions]
        return scores, predictions
        # except Exception as e:
        #     print('exception occurs', e)
        #     return [[]], [[]]
    
    def gold(self, src_data, tgt_data):
        scores = self.model_translator.translate_gold_scores(
            src_data_iter=src_data,
            tgt_data_iter=tgt_data,
            batch_size=self.beam_size
        )
        gold_scores = torch.exp(torch.Tensor(scores)).tolist()
        return gold_scores, None
    