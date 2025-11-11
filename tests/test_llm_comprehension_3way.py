#!/usr/bin/env python3
"""
3-Way LLM Comprehension Test: JSON vs Deep-TOON vs Original TOON

Tests and compares how LLMs understand and work with data in three formats:
- JSON (baseline)
- Deep-TOON (our format)
- Original TOON (reference format)

COST CONTROL: Hard limit adjustable (default 100 for 3-way testing).
"""

import json
import os
import time
import requests
import tiktoken
import argparse
import sys
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import openai
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deep_toon import DeepToonEncoder, DeepToonDecoder
from toon import encode as toon_encode, decode as toon_decode
from tests.test_data_questions import generate_comprehensive_test_cases, TestCase


# Load environment variables
load_dotenv(override=True)

# Global API call counter for cost control
API_CALL_COUNT = 0
MAX_API_CALLS = 100  # Increased for 3-way testing

# Debug settings - can be enabled via command line or environment variable
DEBUG_MODE = os.getenv('LLM_TEST_DEBUG', '').lower() in ('true', '1', 'yes')

# Default confidence threshold for accepting equivalence
DEFAULT_CONFIDENCE_THRESHOLD = 0.8

# Failure analysis mode - get detailed feedback on why responses differ
ANALYZE_FAILURES = False


@dataclass
class EncodingResult:
    original_json: str
    deep_toon: str
    original_toon: str
    original_tokens: int
    deep_toon_tokens: int
    original_toon_tokens: int
    deep_toon_compression: float
    original_toon_compression: float
    deep_toon_roundtrip: bool
    original_toon_roundtrip: bool


@dataclass
class LLMResponse:
    content: str
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass
class ComparisonResult:
    question: str
    json_response: str
    deep_toon_response: str
    original_toon_response: str
    json_vs_deep_equivalent: bool
    json_vs_original_equivalent: bool
    deep_vs_original_equivalent: bool
    json_vs_deep_confidence: float
    json_vs_original_confidence: float
    deep_vs_original_confidence: float
    deep_toon_savings: int
    original_toon_savings: int
    notes: str
    failure_analysis: Optional[str] = None


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
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    return len(encoding.encode(text))




def encode_and_validate(test_case: TestCase) -> EncodingResult:
    """Encode test data in both Deep-TOON and Original TOON formats."""
    
    # Original JSON
    original_json = json.dumps(test_case.data, separators=(',', ':'))
    original_tokens = count_tokens(original_json)
    
    # Deep-TOON encoding
    encoder = DeepToonEncoder()
    decoder = DeepToonDecoder()
    
    deep_toon = encoder.encode(test_case.data)
    deep_toon_tokens = count_tokens(deep_toon)
    
    # Deep-TOON roundtrip validation
    try:
        deep_decoded = decoder.decode(deep_toon)
        deep_roundtrip_success = (test_case.data == deep_decoded)
    except Exception:
        deep_roundtrip_success = False
    
    # Original TOON encoding
    original_toon = toon_encode(test_case.data)
    original_toon_tokens = count_tokens(original_toon)
    
    # Original TOON roundtrip validation
    try:
        original_decoded = toon_decode(original_toon)
        original_roundtrip_success = (test_case.data == original_decoded)
    except Exception:
        original_roundtrip_success = False
    
    # Calculate compression ratios
    deep_toon_compression = (original_tokens - deep_toon_tokens) / original_tokens * 100
    original_toon_compression = (original_tokens - original_toon_tokens) / original_tokens * 100
    
    return EncodingResult(
        original_json=original_json,
        deep_toon=deep_toon,
        original_toon=original_toon,
        original_tokens=original_tokens,
        deep_toon_tokens=deep_toon_tokens,
        original_toon_tokens=original_toon_tokens,
        deep_toon_compression=deep_toon_compression,
        original_toon_compression=original_toon_compression,
        deep_toon_roundtrip=deep_roundtrip_success,
        original_toon_roundtrip=original_roundtrip_success
    )


def query_llm(question: str, data: str, format_name: str) -> LLMResponse:
    """Query OpenAI with data in specified format."""
    
    increment_api_calls()
    
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    prompt = f"""Question: {question}

Please provide a clear, precise answer. If the question asks for a number, provide just the number with appropriate precision.

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
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful data analyst. Provide accurate, concise answers based on the given data."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=8*1024,
            temperature=0
        )
        
        content = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens
        
        if DEBUG_MODE:
            print(f"\n📤 DEBUG - {format_name} Response:")
            print("-" * 40)
            print(f"FULL RESPONSE: {content}")
            print(f"INPUT TOKENS: {input_tokens}")
            print(f"OUTPUT TOKENS: {output_tokens}")
            print(f"TOTAL TOKENS: {total_tokens}")
            print("-" * 40)
        
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens
        )
        
    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
        return LLMResponse(content=f"ERROR: {e}", input_tokens=0, output_tokens=0, total_tokens=0)


def llm_judge_equivalence(question: str, json_resp: str, toon_resp: str) -> Tuple[bool, float, str]:
    """Use LLM as a judge to determine if two responses are equivalent."""
    
    increment_api_calls()
    
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    judge_prompt = f"""You are evaluating whether two AI responses to the same question are equivalent in meaning.

Question: {question}

Response A: {json_resp}

Response B: {toon_resp}

Are these responses equivalent in meaning? Consider:
- Do they provide the same factual information?
- For numerical values: Are they identical when rounded to the specified precision in the question?
- For text values: Do they refer to the same exact entity or result?
- For lists: Do they contain the same items in any order?
- Ignore minor formatting differences like spacing, decimals vs integers if the question specifies rounding
- Focus on whether both responses correctly answer the specific question asked

Examples of EQUIVALENT:
- "25.67" vs "25.670" (same number, different decimal formatting)
- "34" vs "34.0" (same when rounded to integer as requested)
- "New York" vs "New York" (exact match)
- "desktop" vs "desktop" (exact field value match)

Examples of NOT_EQUIVALENT:
- "34.25" vs "34.125" (different numbers even when rounded)
- "x.dummyjson.com" vs "dummyjson.com" (different domain levels)
- "Alice, Bob" vs "Charlie, Diana" (different people)

Respond with:
1. "EQUIVALENT" or "NOT_EQUIVALENT"
2. A confidence score from 0.0 to 1.0
3. Brief explanation

Format: EQUIVALENT|0.95|Both responses provide the same numerical result when rounded to specified precision"""

    if DEBUG_MODE:
        print(f"\n🔍 DEBUG - LLM Judge Query:")
        print("=" * 60)
        print("JUDGE PROMPT:")
        print(judge_prompt)
        print("=" * 60)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert evaluator who determines if two responses are equivalent in meaning. Be precise and consistent."},
                {"role": "user", "content": judge_prompt}
            ],
            max_tokens=200,
            temperature=0
        )
        
        judge_response = response.choices[0].message.content.strip()
        
        if DEBUG_MODE:
            print(f"\n📤 DEBUG - LLM Judge Response:")
            print("-" * 40)
            print(f"JUDGE RESPONSE: {judge_response}")
            print("-" * 40)
        
        # Parse the structured response
        parts = judge_response.split('|')
        if len(parts) >= 3:
            equivalent_str = parts[0].strip().upper()
            confidence = float(parts[1].strip())
            explanation = parts[2].strip()
            equivalent = equivalent_str == "EQUIVALENT"
        else:
            # Fallback parsing
            equivalent = "EQUIVALENT" in judge_response.upper()
            confidence = 0.8 if equivalent else 0.2
            explanation = judge_response
        
        return equivalent, confidence, explanation
        
    except Exception as e:
        print(f"❌ LLM Judge error: {e}")
        # Fallback to simple comparison
        return json_resp.lower().strip() == toon_resp.lower().strip(), 0.5, f"Judge failed: {e}"


def analyze_failure_deep(question: str, json_data: str, toon_data: str, json_resp: str, toon_resp: str, judge_verdict: str) -> str:
    """Deep analysis of why responses differed - helps understand format issues."""
    
    increment_api_calls()
    
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    analysis_prompt = f"""You are a data format expert analyzing why two AI systems gave different answers to the same question using different data formats.

CONTEXT:
- Question: {question}
- The SAME data was provided in two formats: JSON and Deep-TOON (a compact format to save tokens)
- Two identical AI models answered the same question using each format
- A judge determined the answers are NOT equivalent

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

1. **Data Interpretation Issues**: 
   - Does the Deep-TOON format lose important information?
   - Are there ambiguities in how the compressed format represents the data?
   - Could field names or structure be misinterpreted?

2. **AI Processing Differences**:
   - Does the compact format make the data harder to parse mentally?
   - Are there calculation errors due to format complexity?
   - Does the structure affect how the AI navigates the data?

3. **Format-Specific Problems**:
   - Are there specific Deep-TOON syntax elements that could confuse AI?
   - Would clearer separators, labels, or structure help?
   - Are there missing context clues that JSON provides but Deep-TOON doesn't?

4. **Randomness vs Systematic Issues**:
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
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in data formats and AI behavior analysis. Provide detailed, actionable insights about format-related issues."},
                {"role": "user", "content": analysis_prompt}
            ],
            max_tokens=16384,  # Large context for detailed analysis
            temperature=0
        )
        
        analysis = response.choices[0].message.content.strip()
        
        if DEBUG_MODE:
            print(f"\n📤 DEBUG - Failure Analysis Response:")
            print("-" * 40)
            print(f"ANALYSIS: {analysis[:300]}...")
            print("-" * 40)
        
        return analysis
        
    except Exception as e:
        print(f"❌ Failure analysis error: {e}")
        return f"Analysis failed: {e}"


def compare_responses(question: str, json_resp: LLMResponse, toon_resp: LLMResponse, 
                     json_data: str, toon_data: str, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> ComparisonResult:
    """Compare LLM responses for equivalence using LLM judge."""
    
    # Use LLM as judge for equivalence
    judge_equivalent, confidence, judge_notes = llm_judge_equivalence(
        question, json_resp.content, toon_resp.content
    )
    
    # Apply confidence threshold - equivalent only if confidence meets threshold
    equivalent = judge_equivalent and confidence >= confidence_threshold
    
    token_savings = json_resp.input_tokens - toon_resp.input_tokens
    
    # Perform deep failure analysis if responses are not equivalent and analysis is enabled
    failure_analysis = None
    if not equivalent and ANALYZE_FAILURES:
        print(f"     🔍 Performing deep failure analysis...")
        failure_analysis = analyze_failure_deep(
            question, json_data, toon_data, 
            json_resp.content, toon_resp.content, judge_notes
        )
    
    return ComparisonResult(
        question=question,
        json_response=json_resp.content,
        deep_toon_response=toon_resp.content,
        original_toon_response="",  # Will be updated in 3-way function
        json_vs_deep_equivalent=equivalent,
        json_vs_original_equivalent=False,  # Will be updated
        deep_vs_original_equivalent=False,  # Will be updated
        json_vs_deep_confidence=confidence,
        json_vs_original_confidence=0.0,  # Will be updated
        deep_vs_original_confidence=0.0,  # Will be updated
        deep_toon_savings=token_savings,
        original_toon_savings=0,  # Will be updated
        notes=judge_notes,
        failure_analysis=failure_analysis
    )


def compare_all_three_formats(question: str, json_resp: LLMResponse, deep_toon_resp: LLMResponse, 
                             original_toon_resp: LLMResponse, json_data: str, deep_toon_data: str,
                             original_toon_data: str, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> ComparisonResult:
    """Compare responses from all three formats: JSON, Deep-TOON, and Original TOON."""
    
    # Compare JSON vs Deep-TOON
    json_vs_deep_equiv, json_vs_deep_conf, json_vs_deep_notes = llm_judge_equivalence(
        question, json_resp.content, deep_toon_resp.content
    )
    json_vs_deep_equiv = json_vs_deep_equiv and json_vs_deep_conf >= confidence_threshold
    
    # Compare JSON vs Original TOON  
    json_vs_orig_equiv, json_vs_orig_conf, json_vs_orig_notes = llm_judge_equivalence(
        question, json_resp.content, original_toon_resp.content
    )
    json_vs_orig_equiv = json_vs_orig_equiv and json_vs_orig_conf >= confidence_threshold
    
    # Compare Deep-TOON vs Original TOON
    deep_vs_orig_equiv, deep_vs_orig_conf, deep_vs_orig_notes = llm_judge_equivalence(
        question, deep_toon_resp.content, original_toon_resp.content
    )
    deep_vs_orig_equiv = deep_vs_orig_equiv and deep_vs_orig_conf >= confidence_threshold
    
    # Calculate token savings
    deep_toon_savings = json_resp.input_tokens - deep_toon_resp.input_tokens
    original_toon_savings = json_resp.input_tokens - original_toon_resp.input_tokens
    
    # Combine notes
    combined_notes = f"JSON vs Deep: {json_vs_deep_notes} | JSON vs Orig: {json_vs_orig_notes} | Deep vs Orig: {deep_vs_orig_notes}"
    
    # Failure analysis if any comparison fails
    failure_analysis = None
    if ANALYZE_FAILURES and (not json_vs_deep_equiv or not json_vs_orig_equiv):
        print(f"     🔍 Performing deep failure analysis...")
        failure_analysis = analyze_failure_deep(
            question, json_data, deep_toon_data if not json_vs_deep_equiv else original_toon_data,
            json_resp.content, deep_toon_resp.content if not json_vs_deep_equiv else original_toon_resp.content,
            json_vs_deep_notes if not json_vs_deep_equiv else json_vs_orig_notes
        )
    
    return ComparisonResult(
        question=question,
        json_response=json_resp.content,
        deep_toon_response=deep_toon_resp.content,
        original_toon_response=original_toon_resp.content,
        json_vs_deep_equivalent=json_vs_deep_equiv,
        json_vs_original_equivalent=json_vs_orig_equiv,
        deep_vs_original_equivalent=deep_vs_orig_equiv,
        json_vs_deep_confidence=json_vs_deep_conf,
        json_vs_original_confidence=json_vs_orig_conf,
        deep_vs_original_confidence=deep_vs_orig_conf,
        deep_toon_savings=deep_toon_savings,
        original_toon_savings=original_toon_savings,
        notes=combined_notes,
        failure_analysis=failure_analysis
    )


def run_llm_comprehension_tests(confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD):
    """Run the complete 3-way LLM comprehension test suite."""
    
    print("🤖 3-WAY LLM COMPREHENSION TEST: JSON vs Deep-TOON vs Original TOON")
    print("=" * 70)
    print(f"Cost Control: Maximum {MAX_API_CALLS} API calls")
    print(f"Confidence Threshold: {confidence_threshold:.1f} (equivalence requires both judge approval and this confidence)")
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
    
    try:
        for i, test_case in enumerate(test_cases, 1):
            print(f"🧪 Test Case {i}: {test_case.name}")
            print("-" * 40)
            
            # Encode and validate
            encoding_result = encode_and_validate(test_case)
            
            print(f"Original tokens: {encoding_result.original_tokens}")
            print(f"Deep-TOON tokens: {encoding_result.deep_toon_tokens} (compression: {encoding_result.deep_toon_compression:.1f}%)")
            print(f"Original TOON tokens: {encoding_result.original_toon_tokens} (compression: {encoding_result.original_toon_compression:.1f}%)")
            print(f"Deep-TOON roundtrip: {'✅' if encoding_result.deep_toon_roundtrip else '❌'}")
            print(f"Original TOON roundtrip: {'✅' if encoding_result.original_toon_roundtrip else '❌'}")
            
            if not encoding_result.deep_toon_roundtrip or not encoding_result.original_toon_roundtrip:
                print("⚠️  Roundtrip failure detected but continuing with LLM test")
            
            if encoding_result.deep_toon_compression < 10 and encoding_result.original_toon_compression < 10:
                print("⚠️  Low compression achieved on both formats but continuing with LLM test")
            
            print()  # Add spacing
            
            # Test each question
            for j, question in enumerate(test_case.questions):
                print(f"  Q{j+1}: {question}")
                
                # Query with all three formats
                json_response = query_llm(question, encoding_result.original_json, "JSON")
                time.sleep(0.5)  # Rate limiting
                
                deep_toon_response = query_llm(question, encoding_result.deep_toon, "Deep-TOON")
                time.sleep(0.5)  # Rate limiting
                
                original_toon_response = query_llm(question, encoding_result.original_toon, "Original TOON")
                time.sleep(0.5)  # Rate limiting
                
                # Compare all three responses
                comparison = compare_all_three_formats(
                    question, json_response, deep_toon_response, original_toon_response,
                    encoding_result.original_json, encoding_result.deep_toon, encoding_result.original_toon,
                    confidence_threshold
                )
                results.append(comparison)
                
                # Track statistics
                total_questions += 1
                total_savings += comparison.deep_toon_savings
                if comparison.json_vs_deep_equivalent:
                    equivalent_responses += 1
                
                # Print results for all 3 comparisons
                json_deep_status = "✅" if comparison.json_vs_deep_equivalent else "❌"
                json_orig_status = "✅" if comparison.json_vs_original_equivalent else "❌"
                deep_orig_status = "✅" if comparison.deep_vs_original_equivalent else "❌"
                
                print(f"     JSON vs Deep-TOON: {json_deep_status} | {comparison.json_vs_deep_confidence:.1f} confidence")
                print(f"     JSON vs Original TOON: {json_orig_status} | {comparison.json_vs_original_confidence:.1f} confidence") 
                print(f"     Deep vs Original TOON: {deep_orig_status} | {comparison.deep_vs_original_confidence:.1f} confidence")
                print(f"     💾 Deep-TOON savings: {comparison.deep_toon_savings} tokens")
                print(f"     💾 Original TOON savings: {comparison.original_toon_savings} tokens")
                print(f"     📝 JSON: {comparison.json_response[:40]}...")
                print(f"     📝 Deep-TOON: {comparison.deep_toon_response[:40]}...")
                print(f"     📝 Original TOON: {comparison.original_toon_response[:40]}...")
                print(f"     🔍 Notes: {comparison.notes[:100]}...")
                
                # Display failure analysis if available
                if comparison.failure_analysis:
                    print(f"     🔬 Analysis: {comparison.failure_analysis}")
                
                print(f"     🔍 API calls used: {API_CALL_COUNT}/{MAX_API_CALLS}")
                print()
                
                # Check if we're approaching the limit (need 6 calls per question: 3 queries + 3 judges)
                if API_CALL_COUNT >= MAX_API_CALLS - 6:
                    print("⚠️  Approaching API call limit, stopping tests")
                    break
            
            print()
            
            if API_CALL_COUNT >= MAX_API_CALLS - 6:
                break
    
    except APICallLimitExceeded:
        print("🛑 API call limit exceeded, stopping tests")
    
    # Final summary
    print("📊 FINAL RESULTS")
    print("=" * 60)
    print(f"Total questions tested: {total_questions}")
    print(f"Equivalent responses: {equivalent_responses}/{total_questions} ({equivalent_responses/max(total_questions,1)*100:.1f}%)")
    print(f"Average token savings: {total_savings/max(total_questions,1):.1f} tokens per question")
    print(f"Total API calls used: {API_CALL_COUNT}/{MAX_API_CALLS}")
    
    if total_questions > 0:
        success_rate = equivalent_responses / total_questions
        if success_rate >= 0.8:
            print("🎉 SUCCESS: LLMs can understand Deep-TOON format effectively!")
        elif success_rate >= 0.6:
            print("🟡 PARTIAL: LLMs show good understanding of Deep-TOON format")
        else:
            print("❌ NEEDS IMPROVEMENT: LLM comprehension issues detected")
        
        # Additional 3-way comparison summary
        if results:
            json_vs_orig_successes = sum(1 for r in results if r.json_vs_original_equivalent)
            deep_vs_orig_successes = sum(1 for r in results if r.deep_vs_original_equivalent)
            
            print(f"\n📊 3-WAY COMPARISON SUMMARY")
            print(f"JSON vs Deep-TOON: {equivalent_responses}/{total_questions} equivalent ({equivalent_responses/max(total_questions,1)*100:.1f}%)")
            print(f"JSON vs Original TOON: {json_vs_orig_successes}/{total_questions} equivalent ({json_vs_orig_successes/max(total_questions,1)*100:.1f}%)")
            print(f"Deep-TOON vs Original TOON: {deep_vs_orig_successes}/{total_questions} equivalent ({deep_vs_orig_successes/max(total_questions,1)*100:.1f}%)")
    
    print(f"\n💰 Estimated cost: ${API_CALL_COUNT * 0.0002:.4f}")
    
    return results


def main():
    """Main function with command line argument parsing."""
    global DEBUG_MODE
    
    parser = argparse.ArgumentParser(description="3-Way LLM Comprehension Test: JSON vs Deep-TOON vs Original TOON")
    parser.add_argument("--debug", "-d", action="store_true", 
                       help="Enable debug mode (shows full prompts and responses)")
    parser.add_argument("--max-calls", type=int, default=100,
                       help="Maximum number of API calls (default: 100)")
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
                       help=f"Minimum confidence for accepting equivalence (default: {DEFAULT_CONFIDENCE_THRESHOLD})")
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
        results = run_llm_comprehension_tests(args.confidence_threshold)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")


if __name__ == "__main__":
    main()
