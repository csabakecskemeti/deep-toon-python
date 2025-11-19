#!/usr/bin/env python3
"""
2-Way LLM Comprehension Test: JSON vs Deep-TOON
Includes Ground Truth Verification

Tests and compares how LLMs understand and work with data in two formats:
- JSON (baseline)
- Deep-TOON (our format)

COST CONTROL: Hard limit adjustable (default 100).
"""

import json
import os
import sys
import argparse
import re
import time
import openai
import tiktoken
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Tuple, Optional
from dotenv import load_dotenv
from pydantic import BaseModel

# Pydantic model for structured judge output
class EquivalenceJudgment(BaseModel):
    """Structured output from LLM judge for equivalence checking."""
    equivalent: bool  # True if responses are semantically equivalent
    confidence: float  # 0.0-1.0 confidence score
    explanation: str  # Brief explanation of the decision


# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deep_toon import DeepToonEncoder, DeepToonDecoder
from evaluation.test_data_questions import generate_comprehensive_test_cases, TestCase, Question


# Load environment variables
load_dotenv(override=True)

# Global API call counter for cost control
API_CALL_COUNT = 0
MAX_API_CALLS = 100

# Debug settings - can be enabled via command line or environment variable
DEBUG_MODE = os.getenv('LLM_TEST_DEBUG', '').lower() in ('true', '1', 'yes')

# Default confidence threshold for accepting equivalence
DEFAULT_CONFIDENCE_THRESHOLD = 0.8

# Failure analysis mode - get detailed feedback on why responses differ
ANALYZE_FAILURES = False

# Model Configuration - Using GPT-5-mini for all operations
MODEL_NAME = "gpt-5-mini"  # Main test model (400k context)
JUDGE_MODEL = "gpt-5-mini"  # Judge & analysis model (same as main)

# Pricing constants (per 1M tokens) - GPT-5-mini
# Source: https://platform.openai.com/docs/models/gpt-5-mini (as of 2025)
PRICE_INPUT_PER_1M = 0.25    # $0.25 per 1M input tokens
PRICE_OUTPUT_PER_1M = 2.00   # $2.00 per 1M output tokens


@dataclass
class EncodingResult:
    original_json: str
    deep_toon: str
    original_tokens: int
    deep_toon_tokens: int
    deep_toon_compression: float
    deep_toon_roundtrip: bool
    deep_toon_format: str = "Deep-TOON"


@dataclass
class LLMResponse:
    content: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float


@dataclass
class ComparisonResult:
    question: str
    json_response: str
    deep_toon_response: str
    json_vs_deep_equivalent: bool
    json_vs_deep_confidence: float
    deep_toon_savings: int
    notes: str
    json_cost: float
    deep_toon_cost: float
    judge_cost: float
    failure_analysis: Optional[str] = None
    failure_analysis_cost: float = 0.0
    # Ground Truth Metrics
    expected_answer: Any = None
    json_correct: bool = False
    deep_toon_correct: bool = False


class APICallLimitExceeded(Exception):
    """Raised when API call limit is exceeded."""
    pass


def check_api_limit():
    """Check if we've exceeded the API call limit."""
    global API_CALL_COUNT
    if API_CALL_COUNT >= MAX_API_CALLS:
        raise APICallLimitExceeded(f"Exceeded maximum API calls limit: {MAX_API_CALLS}")


def increment_api_calls():
    """Increment and check API call counter."""
    global API_CALL_COUNT
    API_CALL_COUNT += 1
    check_api_limit()


def count_tokens(text: str) -> int:
    """Count GPT tokens in text."""
    encoding = tiktoken.encoding_for_model(MODEL_NAME)
    return len(encoding.encode(text))


def normalize_for_comparison(obj):
    """
    Normalize JSON for semantic comparison.
    Treats missing fields as equivalent to null values.
    
    This is necessary because Deep-TOON normalizes arrays by adding null fields.
    See test_compression_fidelity.py for detailed explanation.
    """
    if isinstance(obj, dict):
        return {k: normalize_for_comparison(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        normalized = [normalize_for_comparison(item) for item in obj]
        
        # For arrays of objects, ensure all objects have the same keys (with null for missing)
        if normalized and all(isinstance(item, dict) for item in normalized):
            all_keys = set()
            for item in normalized:
                all_keys.update(item.keys())
            
            for item in normalized:
                for key in all_keys:
                    if key not in item:
                        item[key] = None
        
        return normalized
    else:
        return obj


def encode_and_validate(test_case: TestCase, smart_threshold: Optional[float] = None) -> EncodingResult:
    """Encode test data in Deep-TOON format."""
    
    # Original JSON
    original_json = json.dumps(test_case.data, separators=(',', ':'))
    original_tokens = count_tokens(original_json)
    
    # Deep-TOON encoding
    encoder = DeepToonEncoder()
    decoder = DeepToonDecoder()
    
    if smart_threshold is not None:
        deep_toon = encoder.smart_encode(test_case.data, threshold=smart_threshold)
        # Check if it fell back to JSON
        is_deep_toon = not (deep_toon.startswith('{') or deep_toon.startswith('['))
        format_used = "Deep-TOON" if is_deep_toon else "JSON (Smart Fallback)"
    else:
        deep_toon = encoder.encode(test_case.data)
        format_used = "Deep-TOON"

    deep_toon_tokens = count_tokens(deep_toon)
    
    # Deep-TOON roundtrip validation with semantic comparison
    try:
        deep_decoded = decoder.decode(deep_toon)
        # Normalize both for semantic comparison (missing fields = null)
        normalized_original = normalize_for_comparison(test_case.data)
        normalized_decoded = normalize_for_comparison(deep_decoded)
        deep_roundtrip_success = (normalized_original == normalized_decoded)
    except Exception:
        deep_roundtrip_success = False
    
    # Calculate compression ratios
    deep_toon_compression = (original_tokens - deep_toon_tokens) / original_tokens * 100
    
    return EncodingResult(
        original_json=original_json,
        deep_toon=deep_toon,
        original_tokens=original_tokens,
        deep_toon_tokens=deep_toon_tokens,
        deep_toon_compression=deep_toon_compression,
        deep_toon_roundtrip=deep_roundtrip_success,
        deep_toon_format=format_used
    )


def query_llm(question: str, data: str, format_name: str) -> LLMResponse:
    """Query OpenAI with data in specified format."""
    
    increment_api_calls()
    
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    prompt = f"""Question: {question}

Please provide a clear, precise answer. If the question asks for a number, provide just the number.
If the answer is a string, provide just the string without quotes.

Data:
{data}"""

    if DEBUG_MODE:
        print(f"\n🔍 DEBUG - {format_name} Query:")
        print("=" * 60)
        print("FULL PROMPT:")
        print(prompt)
        print("=" * 60)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,  # gpt-5-mini
            messages=[
                {"role": "system", "content": "You are a helpful data analyst. Provide accurate, concise answers based on the given data."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=8*1024  # GPT-5 uses max_completion_tokens, no temperature support
        )
        
        content = response.choices[0].message.content.strip()
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens
        
        cost = (input_tokens / 1_000_000 * PRICE_INPUT_PER_1M) + (output_tokens / 1_000_000 * PRICE_OUTPUT_PER_1M)
        
        if DEBUG_MODE:
            print(f"\n📤 DEBUG - {format_name} Response:")
            print("-" * 40)
            print(f"FULL RESPONSE: {content}")
            print(f"INPUT TOKENS: {input_tokens}")
            print(f"OUTPUT TOKENS: {output_tokens}")
            print(f"TOTAL TOKENS: {total_tokens}")
            print(f"COST: ${cost:.6f}")
            print("-" * 40)
        
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost=cost
        )
        
    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
        return LLMResponse(content=f"ERROR: {e}", input_tokens=0, output_tokens=0, total_tokens=0, cost=0.0)



def check_ground_truth(response: str, expected: Any, expected_type: str, question: str = "") -> bool:
    """Verify if the response matches the ground truth using LLM judge."""
    # Short-circuit: if exact match (case-insensitive), no need for LLM
    if str(response).strip().lower() == str(expected).strip().lower():
        return True
    
    # Reuse the same LLM judge that compares responses for equivalence
    # Compare the actual response against the expected ground truth
    is_equivalent, confidence, _, _ = llm_judge_equivalence(
        question, 
        str(expected),  # Treat expected as "response A"
        response        # Treat actual response as "response B"
    )
    # Require at least 70% confidence
    return is_equivalent and confidence >= 0.7




def llm_judge_equivalence(question: str, json_resp: str, toon_resp: str) -> Tuple[bool, float, str, float]:
    """Use LLM as a judge to determine if two responses are equivalent."""
    
    increment_api_calls()
    
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    judge_prompt = f"""You are evaluating whether two AI responses to the same question are equivalent in meaning.

Question: {question}

Response A: {json_resp}

Response B: {toon_resp}

Are these responses equivalent in meaning? Consider:
- Do they provide the same factual information?
- Small differences in wording are acceptable (e.g., "Not found" vs "No product found")
- Different formats of the same data are acceptable (e.g., "123" vs "123.0")
- Semantic equivalence matters more than exact text match

IMPORTANT Examples:
- "Not found", "No product found", "None", "Nothing found" are ALL equivalent
- "Washington" vs "washington" are equivalent (case-insensitive)
- "123" vs "123.0" are equivalent (same number)

Provide your judgment as a JSON object with:
- equivalent: boolean (true if semantically equivalent, false otherwise)
- confidence: number between 0.0 and 1.0
- explanation: brief explanation of your decision"""

    if DEBUG_MODE:
        print(f"\n🔍 DEBUG - LLM Judge Query:")
        print("=" * 60)
        print("JUDGE PROMPT:")
        print(judge_prompt)
        print("=" * 60)

    try:
        # Use structured output with Pydantic model
        response = client.beta.chat.completions.parse(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert evaluator who determines if two responses are equivalent in meaning. Be precise and consistent."},
                {"role": "user", "content": judge_prompt}
            ],
            response_format=EquivalenceJudgment,
            max_completion_tokens=200
        )
        
        # Parse structured output
        judgment = response.choices[0].message.parsed
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost = (input_tokens / 1_000_000 * PRICE_INPUT_PER_1M) + (output_tokens / 1_000_000 * PRICE_OUTPUT_PER_1M)
        
        if DEBUG_MODE:
            print(f"\n📤 DEBUG - LLM Judge Response:")
            print("-" * 40)
            print(f"JUDGE RESPONSE: {judgment}")
            print(f"COST: ${cost:.6f}")
            print("-" * 40)
        
        return judgment.equivalent, judgment.confidence, judgment.explanation, cost
        
    except Exception as e:
        print(f"❌ LLM Judge error: {e}")
        # Fallback to simple comparison
        return json_resp.lower().strip() == toon_resp.lower().strip(), 0.5, f"Judge failed: {e}", 0.0


def analyze_failure_deep(question: str, json_data: str, toon_data: str, json_resp: str, toon_resp: str, judge_verdict: str, ground_truth: Any) -> Tuple[str, float]:
    """Deep analysis of why responses differed - helps understand format issues."""
    
    increment_api_calls()
    
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    analysis_prompt = f"""You are a data format expert analyzing why two AI systems gave different answers to the same question using different data formats.

CONTEXT:
- Question: {question}
- The SAME data was provided in two formats: JSON and Deep-TOON (a compact format to save tokens)
- Two identical AI models answered the same question using each format
- A judge determined the answers are NOT equivalent

GROUND TRUTH:
- Expected Correct Answer: {ground_truth}
- IMPORTANT: First verify if this ground truth seems correct by examining the data below
- If ground truth appears incorrect, note this in your analysis

FORMATS AND RESPONSES:

=== JSON FORMAT DATA ===
{json_data}

=== JSON AI RESPONSE ===
{json_resp}

=== DEEP-TOON FORMAT DATA ===
{toon_data}

=== DEEP-TOON AI RESPONSE ===
{toon_resp}

=== JUDGE VERDICT ===
{judge_verdict}

ANALYSIS TASK:
Please analyze why these responses differ. Consider:

0. **Ground Truth Verification**:
   - Does the expected answer ({ground_truth}) appear correct given the data?
   - Can you manually verify the answer from the JSON data?
   - If ground truth seems wrong, explain why and provide the correct answer

1. **Accuracy Assessment**:
   - Which response (if any) matches the ground truth?
   - Is either response actually correct regardless of ground truth?
   - Are both responses wrong?

2. **Data Interpretation Issues**: 
   - Does the Deep-TOON format lose important information?
   - Are there ambiguities in how the compressed format represents the data?
   - Could field names or structure be misinterpreted?

3. **AI Processing Differences**:
   - Does the compact format make the data harder to parse mentally?
   - Are there calculation errors due to format complexity?
   - Does the structure affect how the AI navigates the data?

4. **Format-Specific Problems**:
   - Are there specific Deep-TOON syntax elements that could confuse AI?
   - Would clearer separators, labels, or structure help?
   - Are there missing context clues that JSON provides but Deep-TOON doesn't?

5. **Randomness vs Systematic Issues**:
   - Does this seem like AI randomness/inconsistency?
   - Or is there a systematic issue with the Deep-TOON format?
   - What specific improvements to Deep-TOON might prevent this issue?

Provide a detailed analysis focusing on actionable insights for improving the Deep-TOON format."""

    if DEBUG_MODE:
        print(f"\n🔍 DEBUG - Failure Analysis Query:")
        print("=" * 60)
        print("ANALYSIS PROMPT:")
        print(analysis_prompt[:500] + "..." if len(analysis_prompt) > 500 else analysis_prompt)
        print("=" * 60)

    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,  # Use gpt5-mini for better reasoning
            messages=[
                {"role": "system", "content": "You are an expert in data formats and AI behavior analysis. Provide detailed, actionable insights about format-related issues."},
                {"role": "user", "content": analysis_prompt}
            ],
            max_completion_tokens=16384  # GPT-5 uses max_completion_tokens instead of max_tokens
        )
        
        analysis = response.choices[0].message.content.strip()
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost = (input_tokens / 1_000_000 * PRICE_INPUT_PER_1M) + (output_tokens / 1_000_000 * PRICE_OUTPUT_PER_1M)
        
        if DEBUG_MODE:
            print(f"\n📤 DEBUG - Failure Analysis Response:")
            print("-" * 40)
            print(f"ANALYSIS: {analysis[:300]}...")
            print(f"COST: ${cost:.6f}")
            print("-" * 40)
        
        return analysis, cost
        
    except Exception as e:
        print(f"❌ Failure analysis error: {e}")
        return f"Analysis failed: {e}", 0.0


def compare_responses(question: Question, json_resp: LLMResponse, toon_resp: LLMResponse, 
                     json_data: str, toon_data: str, expected_answer: Any,
                     confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> ComparisonResult:
    """Compare LLM responses for equivalence and against ground truth."""
    
    # 1. Check against Ground Truth using LLM
    json_correct = check_ground_truth(json_resp.content, expected_answer, question.expected_type, question.text)
    deep_toon_correct = check_ground_truth(toon_resp.content, expected_answer, question.expected_type, question.text)
    
    # 2. Check Equivalence (Judge)
    # Short-circuit: if responses are identical, no need to judge
    if json_resp.content.strip() == toon_resp.content.strip():
        judge_equivalent = True
        confidence = 1.0
        judge_notes = "Responses are identical"
        judge_cost = 0.0
    else:
        judge_equivalent, confidence, judge_notes, judge_cost = llm_judge_equivalence(
            question.text, json_resp.content, toon_resp.content
        )
    
    # Apply confidence threshold
    equivalent = judge_equivalent and confidence >= confidence_threshold
    
    token_savings = json_resp.input_tokens - toon_resp.input_tokens
    
    # Perform deep failure analysis if responses are not equivalent and analysis is enabled
    failure_analysis = None
    failure_analysis_cost = 0.0
    
    if not equivalent and ANALYZE_FAILURES:
        # Print detailed context before analysis
        print(f"     ")
        print(f"     📊 DETAILED RESPONSES (before analysis):")
        print(f"     " + "─" * 70)
        print(f"     JSON Response: {json_resp.content}")
        print(f"     Deep-TOON Response: {toon_resp.content}")
        print(f"     " + "─" * 70)
        print(f"     ⚖️  Judge Verdict: {judge_notes}")
        print(f"     " + "─" * 70)
        print(f"     🔍 Performing deep failure analysis...")
        failure_analysis, failure_analysis_cost = analyze_failure_deep(
            question.text, json_data, toon_data, 
            json_resp.content, toon_resp.content, judge_notes,
            expected_answer  # Pass ground truth for verification
        )
    
    return ComparisonResult(
        question=question.text,
        json_response=json_resp.content,
        deep_toon_response=toon_resp.content,
        json_vs_deep_equivalent=equivalent,
        json_vs_deep_confidence=confidence,
        deep_toon_savings=token_savings,
        notes=judge_notes,
        json_cost=json_resp.cost,
        deep_toon_cost=toon_resp.cost,
        judge_cost=judge_cost,
        failure_analysis=failure_analysis,
        failure_analysis_cost=failure_analysis_cost,
        expected_answer=expected_answer,
        json_correct=json_correct,
        deep_toon_correct=deep_toon_correct
    )


def run_llm_comprehension_tests(confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD, smart_threshold: Optional[float] = None):
    """Run the complete 2-way LLM comprehension test suite."""
    
    print("🤖 2-WAY LLM COMPREHENSION TEST: JSON vs Deep-TOON")
    print("=" * 70)
    print(f"Cost Control: Maximum {MAX_API_CALLS} API calls")
    print(f"Confidence Threshold: {confidence_threshold:.1f}")
    if smart_threshold is not None:
        print(f"Smart Encoding: Enabled (threshold: {smart_threshold})")
    print()
    
    # Generate test cases
    print("📋 Generating test cases...")
    test_cases = generate_comprehensive_test_cases()
    print(f"Generated {len(test_cases)} test cases")
    print()
    
    results = []
    total_savings = 0
    total_questions = 0
    equivalent_responses = 0
    
    # Accuracy stats
    total_json_correct = 0
    total_deep_toon_correct = 0
    
    # Cost tracking
    total_json_cost = 0.0
    total_deep_toon_cost = 0.0
    total_judge_cost = 0.0
    total_analysis_cost = 0.0
    
    # Token tracking
    total_json_input_tokens = 0
    total_deep_toon_input_tokens = 0
    
    try:
        for i, test_case in enumerate(test_cases, 1):
            print(f"🧪 Test Case {i}: {test_case.name}")
            print("-" * 40)
            
            # Encode and validate
            encoding_result = encode_and_validate(test_case, smart_threshold)
            
            print(f"Original tokens: {encoding_result.original_tokens}")
            print(f"Deep-TOON tokens: {encoding_result.deep_toon_tokens} (compression: {encoding_result.deep_toon_compression:.1f}%)")
            print(f"Format used: {encoding_result.deep_toon_format}")
            print(f"Deep-TOON roundtrip: {'✅' if encoding_result.deep_toon_roundtrip else '❌'}")
            
            if not encoding_result.deep_toon_roundtrip:
                print("⚠️  Roundtrip failure detected but continuing with LLM test")
            
            if encoding_result.deep_toon_compression < 10 and "Fallback" not in encoding_result.deep_toon_format:
                print("⚠️  Low compression achieved but continuing with LLM test")
            
            print()  # Add spacing
            
            # Test each question
            for j, question in enumerate(test_case.questions):
                print(f"  Q{j+1}: {question.text}")
                
                # Calculate Ground Truth
                try:
                    expected_answer = question.ground_truth_func(test_case.data)
                    print(f"     🎯 Expected: {expected_answer}")
                except Exception as e:
                    expected_answer = "ERROR"
                    print(f"     ⚠️ Could not calculate ground truth: {e}")
                
                # Query with both formats
                json_response = query_llm(question.text, encoding_result.original_json, "JSON")
                time.sleep(0.5)  # Rate limiting
                
                deep_toon_response = query_llm(question.text, encoding_result.deep_toon, "Deep-TOON")
                time.sleep(0.5)  # Rate limiting
                
                # Compare responses
                comparison = compare_responses(
                    question, json_response, deep_toon_response,
                    encoding_result.original_json, encoding_result.deep_toon,
                    expected_answer,
                    confidence_threshold
                )
                results.append(comparison)
                
                # Track statistics
                total_questions += 1
                total_savings += comparison.deep_toon_savings
                if comparison.json_vs_deep_equivalent:
                    equivalent_responses += 1
                
                if comparison.json_correct:
                    total_json_correct += 1
                if comparison.deep_toon_correct:
                    total_deep_toon_correct += 1
                
                # Track costs and tokens
                total_json_cost += comparison.json_cost
                total_deep_toon_cost += comparison.deep_toon_cost
                total_judge_cost += comparison.judge_cost
                total_analysis_cost += comparison.failure_analysis_cost
                
                total_json_input_tokens += json_response.input_tokens
                total_deep_toon_input_tokens += deep_toon_response.input_tokens
                
                # Print results
                json_status = "✅" if comparison.json_correct else "❌"
                deep_status = "✅" if comparison.deep_toon_correct else "❌"
                equiv_status = "✅" if comparison.json_vs_deep_equivalent else "❌"
                
                test_cost = comparison.json_cost + comparison.deep_toon_cost
                test_savings = comparison.json_cost - comparison.deep_toon_cost
                
                print(f"     JSON: {json_status} | Deep-TOON: {deep_status} | Equivalent: {equiv_status}")
                print(f"     💾 Deep-TOON savings: {comparison.deep_toon_savings} tokens")
                print(f"     💰 Cost: ${test_cost:.5f} (Savings: ${test_savings:.5f})")
                print(f"     📝 JSON: {comparison.json_response[:40]}...")
                print(f"     📝 Deep-TOON: {comparison.deep_toon_response[:40]}...")
                print(f"     🔍 Notes: {comparison.notes[:100]}...")
                
                # Display failure analysis if available
                if comparison.failure_analysis:
                    print(f"     🔬 Analysis: {comparison.failure_analysis}")
                
                print(f"     🔍 API calls used: {API_CALL_COUNT}/{MAX_API_CALLS}")
                print()
                
                # Check if we're approaching the limit
                if API_CALL_COUNT >= MAX_API_CALLS - 4:
                    print("⚠️  Approaching API call limit, stopping tests")
                    break
            
            print()
            
            if API_CALL_COUNT >= MAX_API_CALLS - 4:
                break
    
    except APICallLimitExceeded:
        print("🛑 API call limit exceeded, stopping tests")
    
    # Final summary
    print("📊 FINAL RESULTS")
    print("=" * 60)
    print(f"Total questions tested: {total_questions}")
    
    if total_questions > 0:
        json_acc = total_json_correct / total_questions * 100
        deep_acc = total_deep_toon_correct / total_questions * 100
        equiv_rate = equivalent_responses / total_questions * 100
        
        print(f"JSON Accuracy:      {total_json_correct}/{total_questions} ({json_acc:.1f}%)")
        print(f"Deep-TOON Accuracy: {total_deep_toon_correct}/{total_questions} ({deep_acc:.1f}%)")
        print(f"Equivalence Rate:   {equivalent_responses}/{total_questions} ({equiv_rate:.1f}%)")
        print("-" * 30)
        
        if deep_acc >= 80:
            print("🎉 SUCCESS: Deep-TOON format is highly effective!")
        elif deep_acc >= 60:
            print("🟡 PARTIAL: Deep-TOON format is generally understood.")
        else:
            print("❌ NEEDS IMPROVEMENT: Significant accuracy drop with Deep-TOON.")
            
    print(f"Average token savings: {total_savings/max(total_questions,1):.1f} tokens per question")
    print(f"Total API calls used: {API_CALL_COUNT}/{MAX_API_CALLS}")
    
    print("\n💰 COST ANALYSIS")
    print("=" * 60)
    print(f"Test Cost (Queries):      ${total_json_cost + total_deep_toon_cost:.4f}")
    print(f"  - JSON Cost:            ${total_json_cost:.4f}")
    print(f"  - Deep-TOON Cost:       ${total_deep_toon_cost:.4f}")
    print(f"  - Net Savings:          ${total_json_cost - total_deep_toon_cost:.4f} ({(total_json_cost - total_deep_toon_cost)/max(total_json_cost, 0.0001)*100:.1f}%)")
    print("-" * 30)
    print(f"Judge Cost (Eval):        ${total_judge_cost:.4f}")
    print(f"Analysis Cost (Debug):    ${total_analysis_cost:.4f}")
    print("-" * 30)
    print(f"TOTAL ESTIMATED COST:     ${total_json_cost + total_deep_toon_cost + total_judge_cost + total_analysis_cost:.4f}")
    
    print("\n📉 TOKEN ANALYSIS (Input)")
    print("=" * 60)
    print(f"Total JSON Input Tokens:      {total_json_input_tokens:,}")
    print(f"Total Deep-TOON Input Tokens: {total_deep_toon_input_tokens:,}")
    print(f"Total Token Savings:          {total_json_input_tokens - total_deep_toon_input_tokens:,} ({(total_json_input_tokens - total_deep_toon_input_tokens)/max(total_json_input_tokens, 1)*100:.1f}%)")
    
    return results


def main():
    """Main function with command line argument parsing."""
    global DEBUG_MODE
    
    parser = argparse.ArgumentParser(description="2-Way LLM Comprehension Test: JSON vs Deep-TOON")
    parser.add_argument("--debug", "-d", action="store_true", 
                       help="Enable debug mode (shows full prompts and responses)")
    parser.add_argument("--max-calls", type=int, default=100,
                       help="Maximum number of API calls (default: 100)")
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
                       help=f"Minimum confidence for accepting equivalence (default: {DEFAULT_CONFIDENCE_THRESHOLD})")
    parser.add_argument("--smart-threshold", type=float, help="Use smart encoding with specified threshold (e.g. 0.1)")
    parser.add_argument("--analyze-failures", action="store_true",
                       help="Enable deep failure analysis for non-equivalent responses (requires more API calls)")
    
    args = parser.parse_args()
    
    # Override debug mode if specified via command line
    if args.debug:
        DEBUG_MODE = True
    
    # Override max API calls if specified
    global MAX_API_CALLS
    if args.analyze_failures:
        # Increase limit for failure analysis mode
        MAX_API_CALLS = max(args.max_calls, 150)
    else:
        MAX_API_CALLS = args.max_calls
    
    # Enable failure analysis if requested
    global ANALYZE_FAILURES
    ANALYZE_FAILURES = args.analyze_failures
    
    if DEBUG_MODE:
        print("🐛 DEBUG MODE ENABLED")
        print("Will show full prompts and responses")
        print()
    
    if ANALYZE_FAILURES:
        print("🔍 FAILURE ANALYSIS ENABLED")
        print(f"Will perform deep analysis on non-equivalent responses")
        print(f"API call limit increased to {MAX_API_CALLS}")
        print()
    
    try:
        results = run_llm_comprehension_tests(args.confidence_threshold, args.smart_threshold)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")


if __name__ == "__main__":
    main()
