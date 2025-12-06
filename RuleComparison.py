import argparse
from collections import defaultdict
import json
import random
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import time
from tqdm import tqdm
from prompt import prompt_judge_explanation, prompt_judge_issue
import os

class RuleComparison:
    def __init__(
        self,
        test_model: str,
        test_base_url: str,
        test_api_key: str,
        base_model: str,
        base_base_url: str,
        base_api_key: str,
        max_workers: int = 4,
        test_temperature: float = 0.3,
        base_temperature: float = 0.3,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: int = 5,
        task_config_path: str = "task_config.json",
    ):
        """
        Initialize rule comparison class
        
        :param test_model: Test model name
        :param test_base_url: Test model API base URL
        :param test_api_key: Test model API key
        :param base_model: Base model name
        :param base_base_url: Base model API base URL
        :param base_api_key: Base model API key
        :param max_workers: Maximum number of parallel threads
        :param test_temperature: Test model temperature parameter
        :param base_temperature: Base model temperature parameter
        :param timeout: Timeout for each request (seconds)
        :param max_retries: Maximum number of retries
        :param retry_delay: Retry delay (seconds)
        :param task_config_path: Task configuration file path
        """
        # Initialize test model client
        self.test_client = OpenAI(
            base_url=test_base_url,
            api_key=test_api_key
        )
        
        # Initialize base model client
        self.base_client = OpenAI(
            base_url=base_base_url,
            api_key=base_api_key
        )
        
        self.test_model = test_model
        self.base_model = base_model
        self.max_workers = max_workers
        self.test_temperature = test_temperature
        self.base_temperature = base_temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Load task configuration
        with open(task_config_path, 'r') as f:
            self.task_config = json.load(f)

    def _safe_request_with_retry(self, prompt: str, role: str):
        """
        Safe request method with retry mechanism
        
        :param prompt: Prompt text
        :param role: Role, 'test' or 'base'
        :return: API response
        """
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if role == "test":
                    response = self.test_client.chat.completions.create(
                        model=self.test_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.test_temperature,
                        timeout=self.timeout,
                    )
                    return response
                elif role == "base":
                    response = self.base_client.chat.completions.create(
                        model=self.base_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.base_temperature,
                        seed=1031,
                        timeout=self.timeout,
                    )
                    return response
                else:
                    raise ValueError(f"Invalid role: {role}")
            except Exception as e:
                last_error = e
                print(f"⚠️  Attempt {attempt}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
        
        # If all attempts failed
        raise Exception(f"Request failed after {self.max_retries} retries: {last_error}")

    def _remove_think_tag(self, text: str):
        """Remove <think> tag content"""
        think_end_tag = "</think>"
        if think_end_tag in text:
            text = text.split(think_end_tag, 1)[1].strip()
        return text

    def _compare_llm_rule(
        self, 
        official_explanation: str, 
        description: str, 
        official_rule: str, 
        generated_rule: str, 
        error_type: str
    ):
        """
        Compare two rules using LLM
        
        :param official_explanation: Official explanation
        :param description: Description
        :param official_rule: Official rule
        :param generated_rule: Generated rule
        :param error_type: Error type
        :return: Comparison result dictionary
        """
        if error_type in self.task_config.keys():
            error_detail = self.task_config[error_type]
        else:
            raise ValueError(f"error type {error_type} not in config")
        
        try:
            rules = [("official_rule", official_rule), ("generated_rule", generated_rule)]
            random.shuffle(rules)
            examples_lines = [f"- {label}: {desc}" for label, desc in error_detail.items()]
            error_details_str = "\n".join(examples_lines)
            label_type_str = " | ".join(error_detail.keys())

            prompt = prompt_judge_issue.format(
                description=description,
                rule1=rules[0][1],
                rule2=rules[1][1],
                error_type=error_type,
                error_details=error_details_str,
                label_type=label_type_str,
            )

            response = self._safe_request_with_retry(prompt=prompt, role="test")
            compare_output = response.choices[0].message.content.strip()
            compare_output = self._remove_think_tag(compare_output)
            if "```json" in compare_output:
                compare_output = compare_output.split("```json")[1].split("```")[0].strip()
            
            # Try to parse JSON result
            try:
                compare_dict = json.loads(compare_output)
            except json.JSONDecodeError:
                compare_dict = {"error": "JSONDecodeError", "raw_output": compare_output}

            generated_explanation = compare_dict.get("explanation")
            if generated_explanation:
                try:
                    prompt = prompt_judge_explanation.format(
                        explanation_1=generated_explanation,
                        explanation_2=official_explanation
                    )
                    response = self._safe_request_with_retry(prompt=prompt, role="base")
                    judge_output = response.choices[0].message.content.strip()
                    judge_output = self._remove_think_tag(judge_output)
                    if "True" in judge_output:
                        compare_dict["explanation_judge"] = "True"
                    elif "False" in judge_output:
                        compare_dict["explanation_judge"] = "False"
                    else:
                        compare_dict["explanation_judge"] = "ERROR"
                except Exception as e:
                    compare_dict["explanation_judge"] = f"ERROR: {type(e).__name__}: {str(e)}"

            # Record weaker_rule correspondence
            weaker_rule = compare_dict.get("weaker_rule")
            if weaker_rule == "version_1":
                compare_dict["weaker_version"] = rules[0][0]
            elif weaker_rule == "version_2":
                compare_dict["weaker_version"] = rules[1][0]
            else:
                compare_dict["weaker_version"] = "unknown"

        except Exception as e:
            compare_dict = {
                "error_type": error_type,
                "error": f"{type(e).__name__}: {str(e)}"
            }

        return compare_dict

    def compare_rule(self, item: dict, error_type: str):
        """Compare a single rule"""
        description = item.get("description", "")
        official_rule = item.get("official_rule", "")
        generated_rule = item.get("generated_rule", "")
        official_explanation = item.get("explanation", "")
        return self._compare_llm_rule(
            official_explanation=official_explanation,
            description=description,
            official_rule=official_rule,
            generated_rule=generated_rule,
            error_type=error_type
        )

    def compare_batch_rules(self, items: List[dict], error_type: str):
        """Compare rules in batch"""
        results = []
        indexed_items = [(index, item) for index, item in enumerate(items)]
        
        def compare_rules_for_item(index: int, item: dict, error_type: str):
            try:
                rule = self.compare_rule(item=item, error_type=error_type)
                return (index, rule)
            except Exception as e:
                return (index, f"Error: {str(e)}")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(compare_rules_for_item, index, item, error_type): (index, item) 
                for index, item in indexed_items
            }

            for f in tqdm(as_completed(futures), total=len(futures), desc="Comparing Rules", ncols=100):
                index, result = f.result()
                results.append((index, result))

        results.sort(key=lambda x: x[0])
        return [result for _, result in results]


def run_comparison(
    test_model: str,
    test_base_url: str,
    test_api_key: str,
    base_model: str,
    base_base_url: str,
    base_api_key: str,
    rule_language: str,
    test_num: int,
    max_workers: int = 100,
    test_temperature: float = 1.0,
    base_temperature: float = 1.0,
    timeout: int = 600,
    max_retries: int = 3,
    retry_delay: int = 30,
):
    """
    Run rule comparison experiment
    
    :param test_model: Test model name
    :param test_base_url: Test model API URL
    :param test_api_key: Test model API key
    :param base_model: Base model name
    :param base_base_url: Base model API URL
    :param base_api_key: Base model API key
    :param rule_language: Rule language
    :param test_num: Test number
    :param max_workers: Maximum number of parallel worker threads
    :param test_temperature: Test model temperature
    :param base_temperature: Base model temperature
    :param timeout: Request timeout
    :param max_retries: Maximum number of retries
    :param retry_delay: Retry delay
    """
    input_file = f"./dataset/{rule_language}_detections_error.json"
    output_file = f"./result/{rule_language}_{test_model}_{base_model}_{test_num}_error.json"
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"❌ File not found: {input_file}")
    if os.path.exists(output_file):
        raise FileExistsError(f"❌ File already exists: {output_file}")
    
    with open(input_file, "r", encoding="utf-8") as f:
        generated_dataset = json.load(f)

    generator = RuleComparison(
        test_model=test_model,
        test_base_url=test_base_url,
        test_api_key=test_api_key,
        base_model=base_model,
        base_base_url=base_base_url,
        base_api_key=base_api_key,
        max_workers=max_workers,
        test_temperature=test_temperature,
        base_temperature=base_temperature,
        timeout=timeout,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )
    
    random.seed(1031)
    result = defaultdict(list)

    for error_type, rules_data in generated_dataset.items():
        current_test_num = min(test_num, len(rules_data))
        rules_data = random.sample(rules_data, current_test_num)

        llm_rules = generator.compare_batch_rules(items=rules_data, error_type=error_type)

        for item, llm_rule in zip(rules_data, llm_rules):
            item["Test"] = llm_rule

        result[error_type] = rules_data

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"\n✅ All rule generation completed, results saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run rule comparison experiment")
    
    # Test model parameters
    parser.add_argument("--test_model", type=str, required=True, help="Test model name")
    parser.add_argument("--test_base_url", type=str, required=True, help="Test model API base URL")
    parser.add_argument("--test_api_key", type=str, required=True, help="Test model API key")
    
    # Base model parameters
    parser.add_argument("--base_model", type=str, required=True, help="Base model name")
    parser.add_argument("--base_base_url", type=str, required=True, help="Base model API base URL")
    parser.add_argument("--base_api_key", type=str, required=True, help="Base model API key")
    
    # Other parameters
    parser.add_argument("--rule_language", type=str, required=True, help="Rule language, supports snort, es, splunk")
    parser.add_argument("--test_num", type=int, default=10, help="Test number")
    parser.add_argument("--max_workers", type=int, default=100, help="Maximum number of parallel worker threads")
    parser.add_argument("--test_temperature", type=float, default=1.0, help="Test model temperature parameter")
    parser.add_argument("--base_temperature", type=float, default=1.0, help="Base model temperature parameter")
    parser.add_argument("--timeout", type=int, default=600, help="Request timeout (seconds)")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of retries")
    parser.add_argument("--retry_delay", type=int, default=30, help="Retry delay (seconds)")

    args = parser.parse_args()

    run_comparison(
        test_model=args.test_model,
        test_base_url=args.test_base_url,
        test_api_key=args.test_api_key,
        base_model=args.base_model,
        base_base_url=args.base_base_url,
        base_api_key=args.base_api_key,
        rule_language=args.rule_language,
        test_num=args.test_num,
        max_workers=args.max_workers,
        test_temperature=args.test_temperature,
        base_temperature=args.base_temperature,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )
