import argparse
from collections import defaultdict
import json
import random
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import re
import time
from tqdm import tqdm
from prompt import prompt_generate_issue


class RuleGeneration:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        max_workers: int = 4,
        temperature: float = 0.3,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: int = 5,
        task_config_path: str = "task_config.json",
    ):
        """
        Initialize rule generation class
        
        :param model: Model name
        :param base_url: API base URL
        :param api_key: API key
        :param max_workers: Maximum number of parallel threads
        :param temperature: Temperature parameter
        :param timeout: Timeout for each request (seconds)
        :param max_retries: Maximum number of retries
        :param retry_delay: Retry delay (seconds)
        :param task_config_path: Task configuration file path
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_workers = max_workers
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Read task configuration
        with open(task_config_path, 'r') as f:
            self.task_config = json.load(f)

    def _safe_request_with_retry(self, prompt: str, label_types: list):
        """
        Safe request method with retry mechanism
        
        :param prompt: Prompt text
        :param label_types: List of allowed label types
        :return: Parsed result list
        """
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    seed=1031,
                    timeout=self.timeout,
                )
                result = response.choices[0].message.content.strip()
                
                try:
                    # First try to locate ```json
                    if "```json" in result:
                        result = result.split("```json")[1].split("```")[0].strip()
                    # If ```json not found, locate [ ]
                    elif "[" in result and "]" in result:
                        result = result.split("[")[1].split("]")[0].strip()
                    else:
                        raise ValueError("Cannot find appropriate JSON format")
                    
                    # Try to parse as JSON
                    result_dict = json.loads(result)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Returned content is not valid JSON: {e}\nContent: {result[:100]}")

                # Check if structure is valid
                if not isinstance(result_dict, list):
                    raise ValueError(f"Top-level structure should be a list, but got {type(result_dict).__name__}")

                for i, item in enumerate(result_dict):
                    if not isinstance(item, dict):
                        raise ValueError(f"Element {i} is not a dictionary: {item}")

                    # Check required fields
                    missing_keys = [k for k in ["rule", "explanation", "label_type"] if k not in item]
                    if missing_keys:
                        raise ValueError(f"Element {i} missing fields: {missing_keys}")

                    # Check if label_type is in allowed values
                    label = item["label_type"]
                    if label not in label_types:
                        raise ValueError(
                            f"Element {i} has label_type='{label}' not in allowed set {label_types}"
                        )
                return result_dict
                
            except Exception as e:
                last_error = e
                print(f"⚠️  Attempt {attempt}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
        
        # If all attempts failed
        raise Exception(f"Request failed after {self.max_retries} retries: {last_error}")

    def _generate_llm_rule(self, description: str, official_rule: str):
        """
        Generate rules using LLM
        
        :param description: Description
        :param official_rule: Official rule
        :return: Generated results grouped by error type
        """
        output = {}

        for error_type, error_detail in self.task_config.items():
            try:
                examples_lines = [f"- {label}: {desc}" for label, desc in error_detail.items()]
                error_details_str = "\n".join(examples_lines)
                label_type_str = " | ".join(error_detail.keys())

                prompt = prompt_generate_issue.format(
                    description=description,
                    official_rule=official_rule,
                    error_type=error_type,
                    error_details=error_details_str,
                    label_type=label_type_str,
                )

                result_list = self._safe_request_with_retry(prompt, list(error_detail.keys()))
                output[error_type] = result_list

            except Exception as e:
                print(f"⚠️  Failed to generate {error_type}: {e}")
                continue

        return output

    def generate_rule(self, item: dict):
        """Generate a single rule"""
        description = item.get("description", "")
        official_rule = item.get("rule", "")
        return self._generate_llm_rule(description, official_rule)

    def generate_batch_rules(self, items: List[dict]):
        """Generate rules in batch"""
        results = []
        indexed_items = [(index, item) for index, item in enumerate(items)]
        
        def generate_rules_for_item(index: int, item: dict):
            try:
                rule = self.generate_rule(item)
                return (index, rule)
            except Exception as e:
                return (index, f"Error: {str(e)}")
            
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(generate_rules_for_item, index, item): (index, item) 
                for index, item in indexed_items
            }

            for f in tqdm(as_completed(futures), total=len(futures), desc="Generating Rules", ncols=100):
                index, result = f.result()
                # Skip completely failed items
                if isinstance(result, str) and result.startswith("Error:"):
                    print(f"⚠️  Skipping rule {index} due to generation error: {result}")
                    continue
                results.append((index, result))

        results.sort(key=lambda x: x[0])
        return [result for _, result in results]


def run_generation(
    model: str,
    base_url: str,
    api_key: str,
    rule_language: str,
    max_rules: int = None,
    max_workers: int = 200,
    temperature: float = 1.0,
    timeout: int = 960,
    max_retries: int = 3,
    retry_delay: int = 8,
):
    """
    Run rule generation experiment
    
    :param model: Model name
    :param base_url: API base URL
    :param api_key: API key
    :param rule_language: Rule language
    :param max_rules: Maximum number of rules to process (None for all)
    :param max_workers: Maximum number of parallel worker threads
    :param temperature: Temperature parameter
    :param timeout: Request timeout
    :param max_retries: Maximum number of retries
    :param retry_delay: Retry delay
    """
    import os
    
    input_file = f"./dataset/{rule_language}_detections.json"
    output_file = f"./dataset/{rule_language}_detections_error.json"

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"❌ File not found: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        rules_data = json.load(f)
    
    # Limit the number of rules to process
    if max_rules is not None and max_rules > 0:
        original_count = len(rules_data)
        rules_data = rules_data[:max_rules]
        print(f"📊 Limited processing rules: {original_count} -> {len(rules_data)}")

    generator = RuleGeneration(
        model=model,
        base_url=base_url,
        api_key=api_key,
        max_workers=max_workers,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )

    print(f"✅ Read {len(rules_data)} detection rules, starting to generate LLM_rule...\n")

    all_llm_rules = generator.generate_batch_rules(rules_data)

    result = defaultdict(list)

    for item, error_type_llm_rules in zip(rules_data, all_llm_rules):
        for error_type, llm_rules in error_type_llm_rules.items():
            for llm_rule in llm_rules:
                if (llm_rule.get("rule") is not None and 
                    llm_rule.get("explanation") is not None and 
                    llm_rule.get("label_type") is not None and 
                    llm_rule.get("rule") != item["rule"]):
                    result[error_type].append({
                        "generated_rule": llm_rule.get("rule"),
                        "explanation": llm_rule.get("explanation"),
                        "label_type": llm_rule.get("label_type"),
                        "description": item["description"],
                        "official_rule": item["rule"]
                    })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"\n✅ All rule generation completed, results saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run rule generation experiment")
    
    # Model configuration parameters
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--base_url", type=str, required=True, help="API base URL")
    parser.add_argument("--api_key", type=str, required=True, help="API key")
    
    # Task parameters
    parser.add_argument("--rule_language", type=str, required=True, help="Rule language (support: snort,es,splunk)")
    parser.add_argument("--max_rules", type=int, default=None, help="Maximum number of rules to process (process all if not specified)")
    
    # Optional parameters
    parser.add_argument("--max_workers", type=int, default=200, help="Maximum number of parallel worker threads")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature parameter")
    parser.add_argument("--timeout", type=int, default=960, help="Request timeout (seconds)")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of retries")
    parser.add_argument("--retry_delay", type=int, default=8, help="Retry delay (seconds)")

    args = parser.parse_args()

    run_generation(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        rule_language=args.rule_language,
        max_rules=args.max_rules,
        max_workers=args.max_workers,
        temperature=args.temperature,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )
