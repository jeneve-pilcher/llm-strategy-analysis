import yaml
from typing import List, Dict, Any
import pandas as pd
import logging
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.base import BaseLanguageModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek
import re

logging.basicConfig(level=logging.INFO)

class ModelClientFactory:
    """
    Factory for creating LLM clients from a YAML config file.
    """
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
    strategies = []
    lines = [l for l in raw_output.split('\n') if l.strip()]
    for line in lines:
        match = re.match(r"^\d+\.?\s*(.*?)(?:Risk:|risk:)(.*?)(?:Reward:|reward:)(.*)", line)
        if match:
            strategy = match.group(1).strip()
            risk = match.group(2).strip()
            reward = match.group(3).strip()
            strategies.append({"strategy": strategy, "risk": risk, "reward": reward})
        else:
            strategies.append({"strategy": line.strip(), "risk": "", "reward": ""})
    return strategies[:10]

def generate_and_score_strategies(config_path: str, mode: str = "business") -> pd.DataFrame:
    """
    Generate and score business or product strategies using multiple LLM clients.

    Args:
        config_path (str): Path to YAML config file with model API keys.
        mode (str): 'business' or 'product'.

    Returns:
        pd.DataFrame: DataFrame with columns for prompt, response, generating_model, scoring_model, and scoring_result.
    """
    factory = ModelClientFactory(config_path)
    clients = factory.get_clients()
    records = []

    gen_prompt = (
        f"Can you provide the best 10 {mode} strategies you can think of and calibrate the risk and reward for each?. "
        "Please format each strategy as: '1. [Strategy]. Risk: [risk]. Reward: [reward].'"
    )

    generations = {}
    for model_name, client in clients.items():
        try:
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
        except Exception as e:
            logging.error(f"Error generating strategies with {model_name}: {e}")

    for gen_model, strategies in generations.items():
        strategies_text = "\n".join(
            f"{i+1}. {s['strategy']}. Risk: {s['risk']}. Reward: {s['reward']}."
            for i, s in enumerate(strategies)
        )
        score_prompt = (
            f"Please score these risk and reward calibrated {mode} strategies out of 10 based on their effectiveness and likelihood to succeed.\n"
            f"Return the scores in the same order, left to right, as a single line of floats for a CSV.\n\n"
            f"{strategies_text}"
        )
        for scoring_model, client in clients.items():
            try:
                score_response = client.invoke(score_prompt)
                scores = re.findall(r"[-+]?\d*\.\d+|\d+", score_response.content if hasattr(score_response, "content") else str(score_response))
                scores = [float(s) for s in scores][:10]
                records.append({
                    "prompt": score_prompt,
                    "response": strategies,
                    "generating_model": gen_model,
                    "scoring_model": scoring_model,
                    "scoring_result": scores
                })
            except Exception as e:
                logging.error(f"Error scoring strategies with {scoring_model}: {e}")
    df = pd.DataFrame(records)
    return df

def main():
    """
    Script entry point for generating and scoring strategies.
    Reads config path and mode from command line arguments.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Generate and score business/product strategies with multiple LLMs.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file with model API keys."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["business", "product"],
        default="business",
        help="Choose 'business' or 'product' strategies."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional: Path to save the resulting DataFrame as CSV."
    )

    args = parser.parse_args()

    df = generate_and_score_strategies(args.config, args.mode)
    print(df)

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()