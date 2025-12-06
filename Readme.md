This version is to show the idea of our work:

Can Large Language Models Evaluate Cybersecurity Detection Rules? Framework and Empirical Analysis

to be more reproducible, so it will continuous update.

However, this version is still the stable version, and **can be run to verify our work**, the large update will be updated together before 2025.12.30, the future version will be easy to use.

You can run it like:

conda create -n ENV_NAME python=3.13

conda activate ENV_NAME

pip install -r pip install -r requirements.txt

python Benchmark_generation.py --model deepseek-chat --base_url https://api.deepseek.com --api_key sk-xxxxxxxx --rule_language es --max_rules 20

python RuleComparison.py --test_model deepseek-chat --test_base_url https://api.deepseek.com --test_api_key sk-xxx --base_model deepseek-chat --base_base_url https://api.deepseek.com --base_api_key sk-xxx --rule_language es
