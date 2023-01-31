import pickle as pkl
from loguru import logger
from GRASP.opt import tc_opt
from GRASP.single_step.tc_base import Mt

class TC_api(object):
    def __init__(self, opt, debug=True):
        self.opt = opt
        self.debug = debug
        self._tc_cache = {} # For quick inference
        self._build_service()
    
    def _build_service(self):
        self.rwt = Mt(self.opt.rmt_ckpt, self.opt.vocab_ckpt, self.opt.rmt_device, self.opt.topk, model_type='retro')
        logger.info("Init single-step rwt complete")
        self.fwt = Mt(self.opt.fmt_ckpt, self.opt.vocab_ckpt, self.opt.fmt_device, 1, model_type='forward')    # Fwt only for round-trip confidence check
        logger.info("Init single-step fwt complete")

    def single_exp_check(self, target):
        if target in self._tc_cache:
            return self._tc_cache[target]
        else:
            if self.debug:
                logger.info('Expanding molecule %s'%(target))
            result = []
            scores, predictions = self.rwt.inference([target])
            if not predictions:
                return []
            for (score, smiles) in zip(scores[0], predictions[0]):
                if smiles:
                    result.append({'score': score, "precursors": smiles})
            precursors = list(set([res['precursors'] for res in result if res['precursors'] != '']))
            
            pred_scores, pred_targets = self.fwt.inference(precursors)
            out = []
            for i, (precursor, pred_target) in enumerate(zip(precursors, pred_targets)):
                if target == pred_target[0]:
                    if pred_scores[i][0] >= self.opt.conf_cut_off:
                        out.append({
                                    "reactants": precursor.split('.'),
                                    "reaction": precursor+">>"+target, 
                                    "mt_score": pred_scores[i][0]
                                   })
            if self.debug:
                logger.info('Generated %d reactions'%(len(out)))
            self._tc_cache[target] = out
            if self.debug:
                self.save_tc_cache('./model_ckpt')
            return out

    def multi_exp_check(self, target_list, **conf):
        '''
            [WIP] For parallelization 
        '''
        pass

    def save_tc_cache(self, dir):
        with open(dir + '/tc_cache_cut_off_' + str(self.opt.conf_cut_off) + '.pkl', 'wb') as f:
            pkl.dump(self._tc_cache, f)
        return

    def load_tc_cache(self, file):
        with open(file, 'rb') as f:
            self._tc_cache = pkl.load(f)
        logger.info('Tc Cache Loaded from %s'%(file))
        return
    


tc_api = TC_api(tc_opt, debug=True)