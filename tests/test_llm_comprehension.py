#!/usr/bin/env python3
"""
LLM Comprehension Test for Deep-TOON Format

Tests whether LLMs can understand and work with Deep-TOON encoded data 
as effectively as original JSON, while measuring token efficiency.

Uses modular test data and questions across complexity levels.
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
from tests.test_data_questions import generate_comprehensive_test_cases, TestCase

# Load environment variables
load_dotenv(override=True)

# Global API call counter for cost control
API_CALL_COUNT = 0
MAX_API_CALLS = 50

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
    original_tokens: int
    toon_tokens: int
    compression_ratio: float
    roundtrip_success: bool


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
    toon_response: str
    equivalent: bool
    confidence: float
    token_savings: int
    notes: str
    failure_analysis: Optional[str] = None  # Detailed analysis when responses differ


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
    encoding = tiktoken.encoding_for_model("gpt-4")
    tokens = encoding.encode(text)
    return len(tokens)


def encode_and_validate(test_case: TestCase) -> EncodingResult:
    """Encode test data and validate compression and roundtrip."""
    
    # Original JSON
    original_json = json.dumps(test_case.data, separators=(',', ':'))
    original_tokens = count_tokens(original_json)
    
    # Deep-TOON encoding
    encoder = DeepToonEncoder()
    decoder = DeepToonDecoder()
    
    deep_toon = encoder.encode(test_case.data)
    toon_tokens = count_tokens(deep_toon)
    
    # Validate roundtrip
    try:
        decoded = decoder.decode(deep_toon)
        roundtrip_success = (test_case.data == decoded)
    except Exception:
        roundtrip_success = False
    
    # Calculate compression
    compression_ratio = (original_tokens - toon_tokens) / original_tokens * 100
    
    return EncodingResult(
        original_json=original_json,
        deep_toon=deep_toon,
        original_tokens=original_tokens,
        toon_tokens=toon_tokens,
        compression_ratio=compression_ratio,
        roundtrip_success=roundtrip_success
    )


def query_llm(question: str, data: str, format_name: str) -> LLMResponse:
    """Query OpenAI with data in specified format."""
    
    increment_api_calls()
    
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    prompt = f"""Please answer the following question based on the data provided.

Question: {question}

Data:
{data}

Provide only the exact answer requested, nothing more."""

    if DEBUG_MODE:
        print(f"\n📤 DEBUG - {format_name} Query:")
        print("=" * 60)
        print("PROMPT:")
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
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

Rules for evaluation:
1. Responses are EQUIVALENT if they contain the same factual information, even if worded differently
2. For numerical answers, they must match exactly (34 ≠ 35)
3. For lists, order doesn't matter but content must match
4. For text answers, focus on factual content not style
5. Ignore minor formatting differences

Provide your evaluation in this exact format:
EQUIVALENT: [YES/NO]
CONFIDENCE: [0.0-1.0]
REASON: [Brief explanation of your decision]"""

    if DEBUG_MODE:
        print(f"\n🔍 DEBUG - Judge Query:")
        print("=" * 60)
        print("JUDGE PROMPT:")
        print(judge_prompt[:500] + "..." if len(judge_prompt) > 500 else judge_prompt)
        print("=" * 60)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert judge evaluating response equivalence. Be precise and consistent."},
                {"role": "user", "content": judge_prompt}
            ],
            max_tokens=1000,
            temperature=0
        )
        
        judge_content = response.choices[0].message.content.strip()
        
        if DEBUG_MODE:
            print(f"\n🔍 DEBUG - Judge Response:")
            print("-" * 40)
            print(f"JUDGE RESPONSE: {judge_content}")
            print("-" * 40)
        
        # Parse judge response
        equivalent = False
        confidence = 0.0
        notes = judge_content
        
        for line in judge_content.split('\n'):
            if line.startswith('EQUIVALENT:'):
                equivalent = 'YES' in line.upper()
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence = float(line.split(':')[1].strip())
                except:
                    confidence = 0.0
            elif line.startswith('REASON:'):
                notes = line.split(':', 1)[1].strip()
        
        return equivalent, confidence, notes
        
    except Exception as e:
        print(f"❌ LLM Judge error: {e}")
        return False, 0.0, f"Judge error: {e}"


def analyze_failure_deep(question: str, json_data: str, toon_data: str, json_resp: str, toon_resp: str, judge_verdict: str) -> str:
    """Deep analysis of why responses differed - helps understand format issues."""
    
    increment_api_calls()
    
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    analysis_prompt = f"""You are analyzing why an LLM gave different responses to the same question when provided data in different formats.

QUESTION: {question}

JSON FORMAT RESPONSE: {json_resp}

DEEP-TOON FORMAT RESPONSE: {toon_resp}

JUDGE VERDICT: {judge_verdict}

JSON DATA (first 1000 chars): {json_data[:1000]}

DEEP-TOON DATA (first 1000 chars): {toon_data[:1000]}

Analyze why the responses differ and categorize the issue:

1. Data interpretation issues (loss of information, ambiguities, field names/structure)
2. AI processing differences (parsing complexity, calculation errors, structural navigation)
3. Format-specific problems (syntax confusion, need for clear separators/labels, missing context clues)
4. Randomness vs systematic issues (AI randomness/inconsistency, systematic issues with Deep-TOON, improvements to Deep-TOON)

Provide actionable insights for improving the Deep-TOON format."""

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
        toon_response=toon_resp.content,
        equivalent=equivalent,
        confidence=confidence,
        token_savings=token_savings,
        notes=judge_notes,
        failure_analysis=failure_analysis
    )


def run_llm_comprehension_tests(confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD):
    """Run the complete LLM comprehension test suite."""
    
    print("🤖 LLM COMPREHENSION TEST FOR DEEP-TOON FORMAT")
    print("=" * 60)
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
            print(f"Deep-TOON tokens: {encoding_result.toon_tokens}")
            print(f"Compression: {encoding_result.compression_ratio:.1f}%")
            print(f"Roundtrip: {'✅' if encoding_result.roundtrip_success else '❌'}")
            
            if not encoding_result.roundtrip_success:
                print("⚠️  Roundtrip failure detected but continuing with LLM test")
            
            if encoding_result.compression_ratio < 10:
                print("⚠️  Low compression achieved but continuing with LLM test")
            
            print()  # Add spacing
            
            # Test each question
            for j, question in enumerate(test_case.questions):
                print(f"  Q{j+1}: {question}")
                
                # Query with JSON format
                json_response = query_llm(question, encoding_result.original_json, "JSON")
                time.sleep(0.5)  # Rate limiting
                
                # Query with Deep-TOON format  
                toon_response = query_llm(question, encoding_result.deep_toon, "Deep-TOON")
                time.sleep(0.5)  # Rate limiting
                
                # Compare responses using LLM judge
                comparison = compare_responses(question, json_response, toon_response, 
                                             encoding_result.original_json, encoding_result.deep_toon, confidence_threshold)
                time.sleep(0.5)  # Rate limiting for judge
                results.append(comparison)
                
                # Track statistics
                total_questions += 1
                total_savings += comparison.token_savings
                if comparison.equivalent:
                    equivalent_responses += 1
                
                # Print results - fix the checkmark logic
                status = "✅" if comparison.equivalent else "❌"
                print(f"     {status} Equivalent | {comparison.confidence:.1f} confidence")
                print(f"     💾 Token savings: {comparison.token_savings}")
                print(f"     📝 JSON: {comparison.json_response[:50]}...")
                print(f"     📝 TOON: {comparison.toon_response[:50]}...")
                print(f"     🔍 Judge: {comparison.notes}")
                
                # Display failure analysis if available
                if comparison.failure_analysis:
                    print(f"     🔬 Analysis: {comparison.failure_analysis}")
                
                print(f"     🔍 API calls used: {API_CALL_COUNT}/{MAX_API_CALLS}")
                print()
                
                # Check if we're approaching the limit (need 3 calls per question)
                if API_CALL_COUNT >= MAX_API_CALLS - 3:
                    print("⚠️  Approaching API call limit, stopping tests")
                    break
            
            print()
            
            if API_CALL_COUNT >= MAX_API_CALLS - 3:
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
    
    print(f"\n💰 Estimated cost: ${API_CALL_COUNT * 0.0002:.4f}")
    
    return results


def main():
    """Main function with command line argument parsing."""
    global DEBUG_MODE
    
    parser = argparse.ArgumentParser(description="LLM Comprehension Test for Deep-TOON Format")
    parser.add_argument("--debug", "-d", action="store_true", 
                       help="Enable debug mode (shows full prompts and responses)")
    parser.add_argument("--max-calls", type=int, default=50,
                       help="Maximum number of API calls (default: 50)")
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
        MAX_API_CALLS = max(args.max_calls, 100)
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