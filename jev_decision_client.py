"""Exact declared-option probabilities through llama.cpp's /completion endpoint.

No artificial probability floor. Candidate count is increased if an option is
missing; if complete finite log-probabilities cannot be obtained, fail explicitly.
This client requires llama.cpp's native endpoint, not generic chat logprobs APIs.
"""
import argparse
import json
import math
import urllib.request

LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
HEADER="You are a decision function. Read the state, then answer the question by choosing exactly one option."


class Client:
    def __init__(self,url="http://127.0.0.1:8080",temperature=1.0):
        if not math.isfinite(temperature) or temperature<=0:raise ValueError("invalid calibration temperature")
        self.url=url.rstrip("/");self.temperature=temperature;self.label_ids={}

    def post(self,path,payload):
        request=urllib.request.Request(self.url+path,json.dumps(payload).encode(),{"Content-Type":"application/json"})
        with urllib.request.urlopen(request,timeout=120) as response:return json.load(response)

    def labels(self,n):
        for letter in LETTERS[:n]:
            if letter not in self.label_ids:
                ids=self.post("/tokenize",{"content":" "+letter,"add_special":False})["tokens"]
                if len(ids)!=1 or not isinstance(ids[0],int):raise ValueError("option label must be a single token")
                self.label_ids[letter]=ids[0]
        return [self.label_ids[c] for c in LETTERS[:n]]

    def prompt_logits(self,prompt,n):
        wanted=self.labels(n)
        for count in [64,256,1024,4096]:
            result=self.post("/completion",{"prompt":prompt,"n_predict":1,"temperature":-1.0,
                "n_probs":count,"post_sampling_probs":False,"cache_prompt":False,
                "repeat_penalty":1.0,"presence_penalty":0.,"frequency_penalty":0.})
            positions=result.get("completion_probabilities",result.get("probs"))
            if not positions:raise RuntimeError("server did not return token probabilities")
            candidates=positions[0].get("top_logprobs")
            if candidates is None:raise RuntimeError("server must return raw finite log-probabilities")
            by_id={c["id"]:c["logprob"] for c in candidates}
            if all(i in by_id and by_id[i] is not None and math.isfinite(by_id[i]) for i in wanted):
                return [by_id[i] for i in wanted],count
        raise RuntimeError("Some declared option probabilities are missing/nonfinite. Use native exact-logit inference; no values have been guessed.")

    def decide(self,state,question,options):
        if not 2<=len(options)<=26 or len(set(options))!=len(options):raise ValueError("need 2–26 unique options")
        lines="\n".join(f"{LETTERS[i]}. {o}" for i,o in enumerate(options))
        prompt=f"{HEADER}\n\n[State]\n{state}\n\n[Question]\n{question}\n\n[Options]\n{lines}\n\nAnswer:"
        logits,count=self.prompt_logits(prompt,len(options))
        z=[x/self.temperature for x in logits];m=max(z)
        p=[math.exp(x-m) for x in z];total=sum(p);p=[x/total for x in p]
        return {"choice":options[max(range(len(p)),key=p.__getitem__)],"probabilities":dict(zip(options,p)),"candidate_count":count}

    def decide_bool(self,state,proposition):
        return self.decide(state,proposition,["yes","no"])["probabilities"]["yes"]

    def decide_score(self,state,question,levels):
        result=self.decide(state,question,levels)
        result["expected_level"]=sum(i*result["probabilities"][level] for i,level in enumerate(levels))
        return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--url",default="http://127.0.0.1:8080")
    p.add_argument("--calibration");p.add_argument("--state",required=True);p.add_argument("--question",required=True)
    p.add_argument("--options",nargs="+",required=True);a=p.parse_args()
    t=1.0
    if a.calibration:
        with open(a.calibration) as f:t=json.load(f)["temperature"]
    print(json.dumps(Client(a.url,t).decide(a.state,a.question,a.options),ensure_ascii=False,indent=2))


if __name__=="__main__":main()
