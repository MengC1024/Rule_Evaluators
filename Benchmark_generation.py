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
        初始化规则生成类
        
        :param model: 模型名称
        :param base_url: API base URL
        :param api_key: API key
        :param max_workers: 并行处理的最大线程数
        :param temperature: 温度参数
        :param timeout: 每次请求的超时时间（秒）
        :param max_retries: 最大重试次数
        :param retry_delay: 重试间隔时间（秒）
        :param task_config_path: 任务配置文件路径
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_workers = max_workers
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # 读取任务配置
        with open(task_config_path, 'r') as f:
            self.task_config = json.load(f)

    def _safe_request_with_retry(self, prompt: str, label_types: list):
        """
        带重试机制的安全请求方法
        
        :param prompt: 提示词
        :param label_types: 允许的标签类型列表
        :return: 解析后的结果列表
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
                    # 先尝试定位 ```json
                    if "```json" in result:
                        result = result.split("```json")[1].split("```")[0].strip()
                    # 如果没有找到 ```json，则定位到 [ ]
                    elif "[" in result and "]" in result:
                        result = result.split("[")[1].split("]")[0].strip()
                    else:
                        raise ValueError("无法找到合适的 JSON 格式")
                    
                    # 尝试解析为 JSON
                    result_dict = json.loads(result)
                except json.JSONDecodeError as e:
                    raise ValueError(f"返回内容不是合法 JSON：{e}\n内容：{result[:100]}")

                # 检查结构是否符合
                if not isinstance(result_dict, list):
                    raise ValueError(f"顶层结构应为列表，但实际是 {type(result_dict).__name__}")

                for i, item in enumerate(result_dict):
                    if not isinstance(item, dict):
                        raise ValueError(f"第 {i} 个元素不是字典：{item}")

                    # 检查必须字段
                    missing_keys = [k for k in ["rule", "explanation", "label_type"] if k not in item]
                    if missing_keys:
                        raise ValueError(f"第 {i} 个元素缺少字段：{missing_keys}")

                    # 检查 label_type 是否在枚举值中
                    label = item["label_type"]
                    if label not in label_types:
                        raise ValueError(
                            f"第 {i} 个元素的 label_type='{label}' 不在允许集合 {label_types}"
                        )
                return result_dict
                
            except Exception as e:
                last_error = e
                print(f"⚠️  第 {attempt}/{self.max_retries} 次请求失败：{e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
        
        # 如果所有尝试都失败
        raise Exception(f"请求在重试 {self.max_retries} 次后仍失败: {last_error}")

    def _generate_llm_rule(self, description: str, official_rule: str):
        """
        使用 LLM 生成规则
        
        :param description: 描述
        :param official_rule: 官方规则
        :return: 按错误类型分组的生成结果
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
                print(f"⚠️  生成 {error_type} 失败：{e}")
                continue

        return output

    def generate_rule(self, item: dict):
        """生成单个规则"""
        description = item.get("description", "")
        official_rule = item.get("rule", "")
        return self._generate_llm_rule(description, official_rule)

    def generate_batch_rules(self, items: List[dict]):
        """批量生成规则"""
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
                # 跳过完全错误的项
                if isinstance(result, str) and result.startswith("Error:"):
                    print(f"⚠️  跳过第 {index} 条规则，因生成错误: {result}")
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
    运行规则生成实验
    
    :param model: 模型名称
    :param base_url: API base URL
    :param api_key: API key
    :param rule_language: 规则语言
    :param max_rules: 最大处理规则数量（None表示处理全部）
    :param max_workers: 最大并行工作线程数
    :param temperature: 温度参数
    :param timeout: 请求超时时间
    :param max_retries: 最大重试次数
    :param retry_delay: 重试延迟
    """
    import os
    
    input_file = f"./dataset/{rule_language}_detections.json"
    output_file = f"./dataset/{rule_language}_detections_error.json"

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"❌ 文件未找到: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        rules_data = json.load(f)
    
    # 限制处理的规则数量
    if max_rules is not None and max_rules > 0:
        original_count = len(rules_data)
        rules_data = rules_data[:max_rules]
        print(f"📊 限制处理规则数量：{original_count} -> {len(rules_data)}")

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

    print(f"✅ 读取到 {len(rules_data)} 条检测规则，开始生成 LLM_rule...\n")

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

    print(f"\n✅ 已完成所有规则生成，结果已保存到：{output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行规则生成实验")
    
    # 模型配置参数
    parser.add_argument("--model", type=str, required=True, help="模型名称")
    parser.add_argument("--base_url", type=str, required=True, help="API base URL")
    parser.add_argument("--api_key", type=str, required=True, help="API key")
    
    # 任务参数
    parser.add_argument("--rule_language", type=str, required=True, help="规则语言 (如 en, zh)")
    parser.add_argument("--max_rules", type=int, default=None, help="最大处理规则数量（不指定则处理全部）")
    
    # 可选参数
    parser.add_argument("--max_workers", type=int, default=200, help="最大并行工作线程数")
    parser.add_argument("--temperature", type=float, default=1.0, help="温度参数")
    parser.add_argument("--timeout", type=int, default=960, help="请求超时时间（秒）")
    parser.add_argument("--max_retries", type=int, default=3, help="最大重试次数")
    parser.add_argument("--retry_delay", type=int, default=8, help="重试延迟（秒）")

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
