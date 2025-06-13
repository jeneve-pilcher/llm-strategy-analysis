import yaml
from typing import List, Dict, Any
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.base import BaseLanguageModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek

class ModelClientFactory:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.clients = self._init_clients()

    def _init_clients(self) -> Dict[str, BaseLanguageModel]:
        clients = {}
        if "gpt-4" in self.config:
            clients["gpt-4"] = ChatOpenAI(
                model="gpt-4",
                api_key=self.config["gpt-4"]["api_key"]
            )
        if "claude-sonnet" in self.config:
            clients["claude-sonnet"] = ChatAnthropic(
                model="claude-3-sonnet-20240229",
                api_key=self.config["claude-sonnet"]["api_key"]
            )
        if "gemini-2.5-flash" in self.config:
            clients["gemini-2.5-flash"] = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash-latest",
                api_key=self.config["gemini-2.5-flash"]["api_key"]
            )
        if "deepseek-v3" in self.config:
            clients["deepseek-v3"] = ChatDeepSeek(
                model="deepseek-chat",
                api_key=self.config["deepseek-v3"]["api_key"]
            )
        return clients

    def get_clients(self) -> Dict[str, BaseLanguageModel]:
        return self.clients

def standardize_strategies(raw_output: str) -> List[Dict[str, Any]]:
    """
    Parse the raw output into a list of dicts with keys: strategy, risk, reward.
    Assumes each strategy is on a separate line or numbered.
    """
    import re
    strategies = []
    lines = [l for l in raw_output.split('\n') if l.strip()]
    for line in lines:
        # Try to extract: 1. Strategy: ... Risk: ... Reward: ...
        match = re.match(r"^\d+\.?\s*(.*?)(?:Risk:|risk:)(.*?)(?:Reward:|reward:)(.*)", line)
        if match:
            strategy = match.group(1).strip()
            risk = match.group(2).strip()
            reward = match.group(3).strip()
            strategies.append({"strategy": strategy, "risk": risk, "reward": reward})
        else:
            # Fallback: just store the line as strategy
            strategies.append({"strategy": line.strip(), "risk": "", "reward": ""})
    # Ensure only 10
    return strategies[:10]

def generate_and_score_strategies(config_path: str, mode: str = "business") -> pd.DataFrame:
    """
    mode: 'business' or 'product'
    Returns: list of dicts with prompt, response, generating_model, scoring_model, scoring_result
    """
    factory = ModelClientFactory(config_path)
    clients = factory.get_clients()
    records = []

    # 1. Generation prompt
    gen_prompt = (
        f"Can you provide the best 10 {mode} strategies you can think of and calibrate the risk and reward for each?. "
        "Please format each strategy as: '1. [Strategy]. Risk: [risk]. Reward: [reward].'"
    )
    gen_prompt_template = PromptTemplate.from_template(gen_prompt)

    # 2. Generate strategies with all models
    generations = {}
    for model_name, client in clients.items():
        response = client.invoke(gen_prompt)
        standardized = standardize_strategies(response.content if hasattr(response, "content") else str(response))
        generations[model_name] = standardized
        records.append({
            "prompt": gen_prompt,
            "response": standardized,
            "generating_model": model_name,
            "scoring_model": None,
            "scoring_result": None
        })

    # 3. Scoring prompt
    for gen_model, strategies in generations.items():
        # Prepare strategies text for scoring
        strategies_text = "\n".join(
            f"{i+1}. {s['strategy']}. Risk: {s['risk']}. Reward: {s['reward']}."
            for i, s in enumerate(strategies)
        )
        score_prompt = (
            f"Please score these risk and reward calibrated {mode} strategies out of 10 based on their effectiveness and likelihood to succeed.\n"
            f"Return the scores in the same order, left to right, as a single line of floats for a CSV.\n\n"
            f"{strategies_text}"
        )
        score_prompt_template = PromptTemplate.from_template(score_prompt)
        for scoring_model, client in clients.items():
            score_response = client.invoke(score_prompt)
            # Parse scores as floats
            import re
            scores = re.findall(r"[-+]?\d*\.\d+|\d+", score_response.content if hasattr(score_response, "content") else str(score_response))
            scores = [float(s) for s in scores][:10]
            records.append({
                "prompt": score_prompt,
                "response": strategies,
                "generating_model": gen_model,
                "scoring_model": scoring_model,
                "scoring_result": scores
            })
    df = pd.DataFrame(records)
    return df