import json
import numpy as np
from typing import List, Dict, Optional, Literal, Tuple
from collections import defaultdict
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class AmbiStoryParser:
    """Parse and organize AmbiStory dataset"""
    
    def __init__(self, data):
        self.data = data
    
    def get_samples(self):
        """Get all samples as list of dictionaries"""
        samples = []
        for item_id, item in self.data.items():
            samples.append({
                'id': item_id,
                'homonym': item['homonym'],
                'judged_meaning': item['judged_meaning'],
                'precontext': item['precontext'],
                'sentence': item['sentence'],
                'ending': item.get('ending', ''),
                'example_sentence': item['example_sentence'],
                'full_context': self._build_full_context(item),
                'choices': item['choices'],
                'average': item['average'],
                'stdev': item['stdev'],
                'nonsensical': item['nonsensical']
            })
        return samples
    
    def _build_full_context(self, item):
        """Build full story context"""
        context = item['precontext'] + ' ' + item['sentence']
        if item.get('ending'):
            context += ' ' + item['ending']
        return context.strip()


try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: transformers not available. Free models won't work.")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai not available. GPT models won't work.")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: anthropic not available. Claude models won't work.")


# ============================================================================
# LLM MODEL CONFIGURATIONS
# ============================================================================

LLM_MODEL_CONFIGS = {
    'gemma-3-270m': {
        'name': 'google/gemma-3-270m-it',
        'description': 'Gemma 3 270M Instruct',
        'type': 'huggingface',
        'max_tokens': 100,
        'temperature': 0.1,
    },
    'gemma-3-4b-it': {
        'name': 'google/gemma-3-4b-it',
        'description': 'Gemma 3 4B Instruct',
        'type': 'huggingface',
        'max_tokens': 100,
        'temperature': 0.1,
    },
    'gemma-3-12b-it': {
        'name': 'google/gemma-3-12b-it',
        'description': 'Gemma 3 12B Instruct',
        'type': 'huggingface',
        'max_tokens': 100,
        'temperature': 0.1,
    },
    'gemma-2b': {
        'name': 'google/gemma-2b-it',
        'description': 'Gemma 2B Instruct',
        'type': 'huggingface',
        'max_tokens': 100,
        'temperature': 0.1,
    },
    'gemma-7b-it': {
        'name': 'google/gemma-7b-it',
        'description': 'Gemma 7B Instruct',
        'type': 'huggingface',
        'max_tokens': 100,
        'temperature': 0.1,
    },
    'phi-3-mini': {
        'name': 'microsoft/Phi-3-mini-4k-instruct',
        'description': 'Phi-3 Mini',
        'type': 'huggingface',
        'max_tokens': 100,
        'temperature': 0.1,
    },


    'mistral-7b': {
        'name': 'mistralai/Mistral-7B-Instruct-v0.2',
        'description': 'Mistral 7B Instruct',
        'type': 'huggingface',
        'max_tokens': 100,
        'temperature': 0.1,
    },
    'ministral-3-8b-it-2512': {
        'name': 'mistralai/Ministral-3-8B-Instruct-2512',
        'description': 'Ministral 3 8B Instruct 2512',
        'type': 'mistral',
        'max_tokens': 100,
        'temperature': 0.1,
    },


    'llama-4-scout-17b-16e': {
        'name': 'meta-llama/Llama-4-Scout-17B-16E-Instruct',
        'description': 'Llama 4 Scout 17B 16E Instruct',
        'type': 'huggingface',
        'max_tokens': 100,
        'temperature': 0.1,
        'requires_auth': True,  
    },
    'llama-4-scout-17b-128e': {
        'name': 'meta-llama/Llama-4-Scout-17B-128E-Instruct',
        'description': 'Llama 4 Scout 17B 128E Instruct',
        'type': 'huggingface',
        'max_tokens': 100,
        'temperature': 0.1,
        'requires_auth': True,  
    },
    'llama-3-8b': {
        'name': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'description': 'Llama 3 8B Instruct',
        'type': 'huggingface',
        'max_tokens': 100,
        'temperature': 0.1,
        'requires_auth': True,  
    },    
    'llama-3.2-3b-instruct': {
        'name': 'meta-llama/Llama-3.2-3B-Instruct',
        'description': 'Llama 3.2 3B Instruct',
        'type': 'huggingface',
        'max_tokens': 100,
        'temperature': 0.1,
        'requires_auth': True,  
    },
    
    'gpt-4o-mini': {
        'name': 'gpt-4o-mini',
        'description': 'GPT-4o Mini',
        'type': 'openai',
        'max_tokens': 100,
        'temperature': 0.1,
    },
    'gpt-4o': {
        'name': 'gpt-4o',
        'description': 'GPT-4o',
        'type': 'openai',
        'max_tokens': 100,
        'temperature': 0.1,
    },
    'gpt-4.1': {
        'name': 'gpt-4.1',
        'description': 'GPT-4.1',
        'type': 'openai',
        'max_tokens': 100,
        'temperature': 0.1,
    },
    'gpt-5': {
        'name': 'gpt-5',
        'description': 'GPT-5',
        'type': 'openai_gpt5',
        'max_tokens': 100,
        'temperature': 0.1,
    },
    'gpt-5-mini': {
        'name': 'gpt-5-mini',
        'description': 'GPT-5 Mini',
        'type': 'openai_gpt5',
        'max_tokens': 100,
        'temperature': 0.1,
    },
    'gpt-5.2': {
        'name': 'gpt-5.2',
        'description': 'GPT-5.2',
        'type': 'openai_gpt5',
        'max_tokens': 100,
        'temperature': 0.1,
    },


    'claude-3-haiku': {
        'name': 'claude-3-haiku-20240307',
        'description': 'Claude 3 Haiku',
        'type': 'anthropic',
        'max_tokens': 100,
        'temperature': 0.1,
    },
    'claude-3-5-sonnet': {
        'name': 'claude-3-5-sonnet-20241022',
        'description': 'Claude 3.5 Sonnet',
        'type': 'anthropic',
        'max_tokens': 100,
        'temperature': 0.1,
    },
}


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

class PromptTemplate:
    """Manages different prompt strategies for plausibility rating"""
    
    @staticmethod
    def create_basic_prompt(sample: Dict) -> str:
        prompt = f"""Rate the plausibility of a specific word meaning in context.

Word: {sample['homonym']}
Meaning: {sample['judged_meaning']}

NARRATIVE CONTEXT:
Precontext (beginning): {sample['precontext']}
Sentence with word: {sample['sentence']}
Ending: {sample['ending']}

On a scale of 1 to 5, how plausible is this meaning of "{sample['homonym']}" in the given narrative?

1 = Not plausible at all
5 = Highly plausible

Provide only a single number between 1 and 5 as your answer."""
        return prompt
        
    @staticmethod
    def create_criteria_prompt(sample: Dict) -> str:
        prompt = f"""You are evaluating whether a proposed meaning of a word is supported by its narrative context.

Word: {sample['homonym']}
Proposed meaning: {sample['judged_meaning']}

Narrative context
- Beginning (precontext): {sample['precontext']}
- Sentence containing the word: {sample['sentence']}
- Ending (conclusion): {sample['ending']}

Task
Assess how plausible it is that the word {sample['homonym']} is used with the proposed meaning {sample['judged_meaning']} in this narrative.
Evaluate the following components:
1. Precontext: Does the setup make this meaning likely or expected?
2. Target sentence: Does the local usage of the word support this meaning?
3. Ending: Does the conclusion reinforce, clarify, or confirm this meaning? This is the most important factor.

Rating scale
- 1 Completely implausible. The meaning clearly conflicts with the narrative.
- 2 Mostly implausible. Weak or contradictory support.
- 3 Moderately plausible. Possible but ambiguous or weakly supported.
- 4 Very plausible. Strong and consistent support.
- 5 Highly plausible. Clearly intended and strongly confirmed.

Output format
Return only a single integer from 1 to 5.
Do not include explanations, comments, or extra text."""
        return prompt
      
    @staticmethod
    def create_improved_criteria_prompt(sample: Dict) -> str:
        prompt = f"""You are an impartial evaluator assessing whether a proposed meaning of a word is supported by the provided narrative context. Base your judgment only on the text given.

Word: {sample['homonym']}
Proposed meaning: {sample['judged_meaning']}

Narrative context
- Beginning (precontext): {sample['precontext']}
- Sentence containing the word: {sample['sentence']}
- Ending (conclusion): {sample['ending']}

Task
Rate the plausibility that the word {sample['homonym']} is used with the proposed meaning {sample['judged_meaning']} in this narrative.

Evaluation criteria
1. Precontext: Does the setup make this meaning likely or expected?
2. Target sentence: Does the local usage support this meaning?
3. Ending: Does the conclusion reinforce or confirm this meaning? This is the strongest source of evidence.

Decision rules
- If the ending clearly contradicts the proposed meaning, the rating must be 1 or 2.
- If evidence is mixed or unclear, choose the lower plausible rating.
- A rating of 5 requires explicit confirmation in the ending and no contradictions elsewhere.

Rating scale
1 Completely implausible: Clear contradiction.
2 Mostly implausible: Weak or conflicting evidence.
3 Moderately plausible: Possible but ambiguous.
4 Very plausible: Strong and consistent support.
5 Highly plausible: Clearly intended and explicitly confirmed.

Output format
Return only a single integer from 1 to 5.
Do not include explanations, comments, or any extra text."""
        return prompt
    

# ============================================================================
# BASE LLM SCORER
# ============================================================================

class BaseLLMScorer:
    """Base class for LLM-based scoring"""
    
    def __init__(self, model_key: str, prompt_strategy: str = 'basic'):
        self.model_key = model_key
        self.config = LLM_MODEL_CONFIGS[model_key]
        self.prompt_strategy = prompt_strategy
        self.prompt_template = PromptTemplate()
        
    def create_prompt(self, sample: Dict) -> str:
        """Create prompt based on strategy"""
        if self.prompt_strategy == 'basic':
            return self.prompt_template.create_basic_prompt(sample)
        elif self.prompt_strategy == 'criteria':
            return self.prompt_template.create_criteria_prompt(sample)
        elif self.prompt_strategy == 'improved_criteria':
            return self.prompt_template.create_improved_criteria_prompt(sample)
    
    def extract_rating(self, response: str) -> float:
        """Extract numeric rating from response"""
        # Try to find rating in various formats
        import re
        
        # Look for "Rating: X" pattern
        match = re.search(r'[Rr]ating[:\s]+([1-5](?:\.[0-9]+)?)', response)
        if match:
            return float(match.group(1))
        
        # Look for standalone number at end
        match = re.search(r'\b([1-5](?:\.[0-9]+)?)\b(?!.*\b[1-5])', response)
        if match:
            return float(match.group(1))
        
        # Look for any number between 1-5
        match = re.search(r'\b([1-5])\b', response)
        if match:
            return float(match.group(1))
        
        # Default to middle rating if parsing fails
        print(f"Warning: Could not extract rating from: {response[:100]}")
        return 3.0
    
    def score_plausibility(self, sample: Dict) -> float:
        """Score plausibility - to be implemented by subclasses"""
        raise NotImplementedError


# ============================================================================
# HUGGING FACE LLM SCORER
# ============================================================================

class HuggingFaceLLMScorer(BaseLLMScorer):
    """Scorer using Hugging Face models"""
    
    def __init__(self, model_key: str, prompt_strategy: str = 'basic', 
                 device: str = None, hf_token: Optional[str] = None):
        super().__init__(model_key, prompt_strategy)
        
        if not HF_AVAILABLE:
            raise ImportError("transformers library not available")
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.hf_token = hf_token
        
        print(f"Loading {self.config['description']}...")
        
        # Load model and tokenizer
        model_name = self.config['name']
        
        if self.config.get('requires_auth') and not hf_token:
            print(f"Warning: {model_name} requires HF authentication token")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            token=hf_token,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            device_map='auto' if self.device == 'cuda' else None,
            trust_remote_code=False
        )
        
        if self.device == 'cpu':
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
        # Set pad token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✓ Model loaded on {self.device}")
    
    def generate_response(self, prompt: str) -> str:
        """Generate response from model"""
        inputs = self.tokenizer(
            prompt, 
            return_tensors='pt', 
            truncation=True, 
            max_length=2048
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config['max_tokens'],
                temperature=self.config['temperature'],
                do_sample=self.config['temperature'] > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the generated part
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response
    
    def score_plausibility(self, sample: Dict) -> float:
        """Score plausibility using HF model"""
        prompt = self.create_prompt(sample)
        response = self.generate_response(prompt)
        rating = self.extract_rating(response)
        return np.clip(rating, 1.0, 5.0), response


class HuggingFaceMistralLLMScorer(BaseLLMScorer):
    """Scorer using Hugging Face models"""
    
    def __init__(self, model_key: str, prompt_strategy: str = 'basic', 
                 device: str = None, hf_token: Optional[str] = None):
        super().__init__(model_key, prompt_strategy)
        
        if not HF_AVAILABLE:
            raise ImportError("transformers library not available")
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.hf_token = hf_token
        
        print(f"Loading {self.config['description']}...")
        
        # Load model and tokenizer
        model_name = self.config['name']
        
        if self.config.get('requires_auth') and not hf_token:
            print(f"Warning: {model_name} requires HF authentication token")
        
        self.tokenizer = MistralCommonBackend.from_pretrained(
            model_name, 
            token=hf_token,
            trust_remote_code=True
        )
        
        self.model = Mistral3ForConditionalGeneration.from_pretrained(
            model_name,
            token=hf_token,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            device_map='auto' if self.device == 'cuda' else None,
            trust_remote_code=False
        )
        
        if self.device == 'cpu':
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
        # Set pad token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✓ Model loaded on {self.device}")
    
    def generate_response(self, prompt: str) -> str:
        """Generate response from model"""

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ]

        tokenized = self.tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True)

        tokenized["input_ids"] = tokenized["input_ids"].to(device="cuda")
        
        output = self.model.generate(
            **tokenized,
            max_new_tokens=512,
        )[0]

        response = self.tokenizer.decode(output[len(tokenized["input_ids"][0]):])
        
        return response
    
    def score_plausibility(self, sample: Dict) -> float:
        """Score plausibility using HF model"""
        prompt = self.create_prompt(sample)
        response = self.generate_response(prompt)
        rating = self.extract_rating(response)
        return np.clip(rating, 1.0, 5.0), response


# ============================================================================
# OPENAI LLM SCORER
# ============================================================================

class OpenAILLMScorer(BaseLLMScorer):
    """Scorer using OpenAI models"""
    
    def __init__(self, model_key: str, prompt_strategy: str = 'basic',
                 api_key: Optional[str] = None):
        super().__init__(model_key, prompt_strategy)
        
        if not OPENAI_AVAILABLE:
            raise ImportError("openai library not available")
        
        if api_key:
            openai.api_key = api_key
        
        print(f"Initialized {self.config['description']}")
    
    def generate_response(self, prompt: str) -> str:
        """Generate response from OpenAI"""
        try:
            response = openai.chat.completions.create(
                model=self.config['name'],
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that rates word sense plausibility."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config['max_tokens'],
                temperature=self.config['temperature'],
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return "3"  # Default to middle rating on error
    
    def score_plausibility(self, sample: Dict) -> Tuple[float, str]:
        """Score plausibility using OpenAI"""
        prompt = self.create_prompt(sample)
        response = self.generate_response(prompt)
        rating = self.extract_rating(response)
        return np.clip(rating, 1.0, 5.0), response


# ============================================================================
# OPENAI LLM SCORER (GPT-5)
# ============================================================================

class OpenAILLMScorerGPT5(BaseLLMScorer):
    """Scorer using OpenAI GPT-5 models"""

    def __init__(self, model_key: str, prompt_strategy: str = 'basic',
                 api_key: Optional[str] = None):
        super().__init__(model_key, prompt_strategy)

        if not OPENAI_AVAILABLE:
            raise ImportError("openai library not available")

        if api_key:
            openai.api_key = api_key

        print(f"Initialized {self.config['description']}")

    def generate_response(self, prompt: str) -> str:
        """Generate response from OpenAI GPT-5"""
        try:
            response = openai.responses.create(
                model=self.config['name'],
                input=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "You are a helpful assistant that rates word sense plausibility."
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": prompt
                            }
                        ],
                    }
                ],
                max_output_tokens=self.config['max_tokens'],
            )

            return response.output_text

        except Exception as e:
            print(f"OpenAI API error: {e}")
            return "3"

    def score_plausibility(self, sample: Dict) -> Tuple[float, str]:
        """Score plausibility using OpenAI"""
        prompt = self.create_prompt(sample)
        response = self.generate_response(prompt)
        rating = self.extract_rating(response)
        return np.clip(rating, 1.0, 5.0), response



# ============================================================================
# ANTHROPIC LLM SCORER
# ============================================================================

class AnthropicLLMScorer(BaseLLMScorer):
    """Scorer using Anthropic Claude models"""
    
    def __init__(self, model_key: str, prompt_strategy: str = 'basic',
                 api_key: Optional[str] = None):
        super().__init__(model_key, prompt_strategy)
        
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic library not available")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        print(f"Initialized {self.config['description']}")
    
    def generate_response(self, prompt: str) -> str:
        """Generate response from Claude"""
        try:
            message = self.client.messages.create(
                model=self.config['name'],
                max_tokens=self.config['max_tokens'],
                temperature=self.config['temperature'],
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content[0].text
        except Exception as e:
            print(f"Anthropic API error: {e}")
            return "3"  # Default to middle rating on error
    
    def score_plausibility(self, sample: Dict) -> float:
        """Score plausibility using Claude"""
        prompt = self.create_prompt(sample)
        response = self.generate_response(prompt)
        rating = self.extract_rating(response)
        return np.clip(rating, 1.0, 5.0), response


# ============================================================================
# FACTORY
# ============================================================================

def create_llm_scorer(model_key: str, prompt_strategy: str = 'basic',
                     openai_api_key: Optional[str] = None,
                     anthropic_api_key: Optional[str] = None,
                     hf_token: Optional[str] = None) -> BaseLLMScorer:
    """Factory function to create appropriate scorer"""
    
    if model_key not in LLM_MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_key}")
    
    config = LLM_MODEL_CONFIGS[model_key]
    
    if config['type'] == 'huggingface':
        return HuggingFaceLLMScorer(model_key, prompt_strategy, hf_token=hf_token)
    elif config['type'] == 'mistral':
        return HuggingFaceMistralLLMScorer(model_key, prompt_strategy, hf_token=hf_token)
    elif config['type'] == 'openai':
        return OpenAILLMScorer(model_key, prompt_strategy, api_key=openai_api_key)
    elif config['type'] == 'openai_gpt5':
        return OpenAILLMScorerGPT5(model_key, prompt_strategy, api_key=openai_api_key)
    elif config['type'] == 'anthropic':
        return AnthropicLLMScorer(model_key, prompt_strategy, api_key=anthropic_api_key)
    else:
        raise ValueError(f"Unknown model type: {config['type']}")


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_llm_scorer(
    scorer: BaseLLMScorer,
    samples: List[Dict],
    verbose: bool = True
) -> Dict:
    
    predictions = []
    ground_truth = []
    raw_responses = []
    input_data = []

    iterator = tqdm(samples, desc=f"Evaluating {scorer.config['description']}") if verbose else samples

    for sample in iterator:
        pred, raw_response = scorer.score_plausibility(sample)
        predictions.append(pred)
        raw_responses.append(raw_response)
        input_data.append(sample)

        avg = sample.get('average', None)
        if avg is not None and avg != "(???)":
            ground_truth.append(avg)

        # Rate limiting for API calls
        if scorer.config.get('type') in ['openai', 'anthropic']:
            time.sleep(0.5)

    predictions = np.array(predictions)

    results = {
        'model': scorer.model_key,
        'prompt_strategy': scorer.prompt_strategy,
        'predictions': predictions.tolist(),
        'raw_responses': raw_responses,
        'input_data': input_data
    }

    if len(ground_truth) == len(predictions):
        from scipy.stats import spearmanr, pearsonr

        ground_truth = np.array(ground_truth)

        spearman = spearmanr(predictions, ground_truth)
        pearson = pearsonr(predictions, ground_truth)
        mae = np.mean(np.abs(predictions - ground_truth))
        rmse = np.sqrt(np.mean((predictions - ground_truth) ** 2))
        within_1 = np.mean(np.abs(predictions - ground_truth) <= 1.0) * 100

        results['metrics'] = {
            'spearman': spearman.correlation,
            'pearson': pearson[0],
            'mae': mae,
            'rmse': rmse,
            'within_1': within_1
        }
        results['ground_truth'] = ground_truth.tolist()
    else:
        # Test set: no usable ground truth
        results['metrics'] = None
        results['ground_truth'] = None

    return results



# ============================================================================
# MAIN COMPARISON
# ============================================================================

def compare_llm_models(data_path: str,
                      model_keys: Optional[List[str]] = None,
                      prompt_strategies: Optional[List[str]] = None,
                      max_samples: Optional[int] = None,
                      openai_api_key: Optional[str] = None,
                      anthropic_api_key: Optional[str] = None,
                      hf_token: Optional[str] = None):
    """Compare multiple LLM models and prompt strategies"""
    
    # Default configurations
    if model_keys is None:
        model_keys = ['gemma-2b', 'phi-3-mini', 'mistral-7b', 'gpt-4o-mini', 'claude-3-haiku']
    
    if prompt_strategies is None:
        prompt_strategies = ['basic']
    
    # Load data
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    parser = AmbiStoryParser(data)
    samples = parser.get_samples()
    
    if max_samples:
        samples = samples[:max_samples]
    
    print(f"\nEvaluating on {len(samples)} samples")
    print(f"Models: {len(model_keys)}")
    print(f"Prompt strategies: {len(prompt_strategies)}")
    print("="*80)
    
    results = []
    
    for model_key in model_keys:
        for prompt_strategy in prompt_strategies:
            print(f"\n{'='*80}")
            print(f"Model: {LLM_MODEL_CONFIGS[model_key]['description']}")
            print(f"Prompt: {prompt_strategy}")
            print(f"{'='*80}")
            
            try:
                # Create scorer
                scorer = create_llm_scorer(
                    model_key=model_key,
                    prompt_strategy=prompt_strategy,
                    openai_api_key=openai_api_key,
                    anthropic_api_key=anthropic_api_key,
                    hf_token=hf_token
                )
                
                # Evaluate
                result = evaluate_llm_scorer(scorer, samples, verbose=True)
                results.append(result)
                
                if result['metrics']:
                    print(f"\nResults:")
                    print(f"  Spearman: {result['metrics']['spearman']:.4f}")
                    print(f"  Pearson: {result['metrics']['pearson']:.4f}")
                    print(f"  MAE: {result['metrics']['mae']:.4f}")
                    print(f"  Within 1 point: {result['metrics']['within_1']:.1f}%")
                
            except Exception as e:
                print(f"\nFailed: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    return results