import os
import re
import evaluate
import numpy as np
from lm_eval.base import Task, rf
from lm_eval.metrics import mean, bleu, chrf, ter
from .utils import process_text\

from .zhutils import process_zhtext
from seqeval.metrics import f1_score as entity_score
from sklearn.metrics import f1_score, matthews_corrcoef, mean_squared_error
from metrics.BartScore.bart_score import BARTScorer


class Classification(Task):
    CALCULATE_MCC = True
    LOWER_CASE = True
    VERSION = 1
    EVAL_LAST_TURN = True

    def reformulate_turn_req(self, req, turn_request, turn):
        return req

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def construct_requests(self, doc, ctx):
    
        cont_request = rf.greedy_until(ctx, {"until": None})
        return cont_request

    def doc_to_decontamination_query(self, doc):
        return doc["text"]

    def doc_to_text(self, doc):

        return doc["query"]

    def doc_to_target(self, doc):

        return doc["answer"]

    def process_results(self, doc, results):
        gold: str = doc["choices"][doc["gold"]]
        if self.LOWER_CASE:
            gold = gold.lower()
        ini_result = results[0].strip()
        if self.LOWER_CASE:
            ini_result = ini_result.lower()

        result = None
        for choice in doc["choices"]:
            if self.LOWER_CASE:
                choice = choice.lower()
            if choice in ini_result:
                result = choice
                break
        if result is None:
            result = "missing"

        acc = 1.0 if gold == result else 0.0

        results = {
            "acc": acc,
            "missing": int(result == "missing"),
            "f1": (result, gold),
            "macro_f1": (result, gold),
        }

        if self.CALCULATE_MCC:
            results["mcc"] = (result, gold)

        return results

    def higher_is_better(self):
        metrics = {
            "acc": True,
            "f1": True,
            "macro_f1": True,
            "missing": False,
        }
        if self.CALCULATE_MCC:
            metrics["mcc"] = True
        return metrics

    def weighted_f1(self, items):
        preds, golds = zip(*items)
        labels = list(set(golds))
        preds = np.array(preds)
        golds = np.array(golds)
        f1 = f1_score(golds, preds, average="weighted", labels=labels)
        return f1

    def macro_f1(self, items):
        preds, golds = zip(*items)
        labels = list(set(golds))
        preds = np.array(preds)
        golds = np.array(golds)
        f1 = f1_score(golds, preds, average="macro", labels=labels)
        return f1

    def matthews_corrcoef(self, items):
        preds, golds = zip(*items)
        labels = {label: i for i, label in enumerate(list(set(golds)))}
        preds = [labels.get(pred, -1) for pred in preds]
        golds = [labels.get(gold, -1) for gold in golds]
        return matthews_corrcoef(golds, preds)

    def aggregation(self):
        metrics = {
            "acc": mean,
            "missing": mean,
            "f1": self.weighted_f1,
            "macro_f1": self.macro_f1,
        }
        if self.CALCULATE_MCC:
            metrics["mcc"] = self.matthews_corrcoef
        return metrics

class ACLUE(Classification):
    DATASET_PATH = "DATASET_PATH"

class AuthIDE(Classification):
    DATASET_PATH = "DATASET_PATH"

class CritBias(Classification):
    DATASET_PATH = "DATASET_PATH"

# 生成式摘要生成
class AbstractiveSummarization(Task):
    VERSION = 1
    DATASET_NAME = None
    EVAL_LAST_TURN = True

    def reformulate_turn_req(self, req, turn_request, turn):
        return req

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return doc["query"]

    def doc_to_target(self, doc):
        return doc["answer"]

    def process_results(self, doc, results):
        return {
            "rouge1": (doc["answer"], results[0]),
            "rouge2": (doc["answer"], results[0]),
            "rougeL": (doc["answer"], results[0]),
            "bert_score_f1": (doc["answer"], results[0]),
            "bart_score": (doc["answer"], results[0]),
        }

    def higher_is_better(self):
        return {
            "rouge1": True,
            "rouge2": True,
            "rougeL": True,
            "bert_score_f1": True,
            "bart_score": True,
        }

    def construct_requests(self, doc, ctx):
        cont_request = rf.greedy_until(ctx, {"until": None})
        return cont_request

    def rouge_score(self, items):
        golds, preds = zip(*items)
        rouge = evaluate.load("src/metrics/rouge")
        results = rouge.compute(predictions=preds, references=golds)
        return results

    def rouge1(self, items):
        results = self.rouge_score(items)
        return results["rouge1"]

    def rouge2(self, items):
        results = self.rouge_score(items)
        return results["rouge2"]

    def rougeL(self, items):
        results = self.rouge_score(items)
        return results["rougeL"]
   
   
    def bert_score(self, items):
        if getattr(self, "_cache_bertscore", None) is None:
            golds, preds = zip(*items)
            bertscore = evaluate.load("src/metrics/bertscore")
            self._cache_bertscore = bertscore.compute(
                predictions=preds,
                references=golds,
                model_type="google-bert/bert-base-multilingual-cased",
            )
            return self._cache_bertscore
        else:
            return self._cache_bertscore
    
    def bert_score_f1(self, items):
        res = self.bert_score(items)
        return sum(res["f1"]) / len(res["f1"])

    def bart_score(self, items):
        golds, preds = zip(*items)
        bart_scorer = BARTScorer(device="cuda", checkpoint="facebook/bart-large-cnn")
        bart_scorer.load(path="src/metrics/BARTScore/bart_score.pth")
        res = bart_scorer.score(srcs=preds, tgts=golds, batch_size=8)
        return sum(res) / len(res)

    def aggregation(self):
        return {
            "rouge1": self.rouge1,
            "rouge2": self.rouge2,
            "rougeL": self.rougeL,
            "bert_score_f1": self.bert_score_f1,
            "bart_score": self.bart_score,
        }

class ClaTrans(AbstractiveSummarization):
    DATASET_PATH = "DATASET_PATH"


class QA(Task):
    VERSION = 1
    DATASET_NAME = None
    EVAL_LAST_TURN = True

    def reformulate_turn_req(self, req, turn_request, turn):
        return req

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["text"]

    def doc_to_text(self, doc):
        return doc["query"]

    def construct_requests(self, doc, ctx):
        cont_request = rf.greedy_until(ctx, {"until": None})
        return cont_request

    def doc_to_target(self, doc):
        return doc["answer"]

    def process_results(self, doc, results):
        gold = doc["answer"]

        acc = 1.0 if results[0].strip() == gold else 0.0

        return {
            "acc": acc,
        }

    def higher_is_better(self):
        return {
            "acc": True,
        }

    def aggregation(self):
        return {
            "acc": mean,
        }

class ReadCom(QA):
    DATASET_PATH = "DATASET_PATH"

class CritPred(QA):
    DATASET_PATH = "DATASET_PATH"

class NER(Task):
    VERSION = 1
    DATASET_PATH = "DATASET_PATH"
    DATASET_NAME = None
    EVAL_LAST_TURN = True

    def reformulate_turn_req(self, req, turn_request, turn):
        return req

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["text"]

    def doc_to_text(self, doc):
        return doc["query"]

    def construct_requests(self, doc, ctx):
   
        cont_request = rf.greedy_until(ctx, {"until": None})
        return cont_request

    def doc_to_target(self, doc):
        return doc["answer"]

    def process_results(self, doc, results):
        text = doc["text"]
        pred = process_text(results[0], text)

        return {"entity_f1": (pred, doc["label"], results[0])}

    def higher_is_better(self):
        return {
            "entity_f1": True,
        }

    @classmethod
    def entity_f1(cls, items):
        preds, golds, _ = zip(*items)
        f1 = entity_score(golds, preds)
        return f1

    def aggregation(self):
        return {
            "entity_f1": self.entity_f1,
        }

class LitNRE(NER):
    DATASET_PATH = "DATASET_PATH"

    def process_results(self, doc, results):
        text = ' '.join(doc["text"])
        pred = process_zhtext(results[0], text)

        return {"entity_f1": (pred, doc["label"], results[0])}

