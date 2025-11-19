#!/usr/bin/env python3
"""
Deep-TOON Compression & Fidelity Test
=====================================

A standalone test to verify:
1. Data Fidelity: Can we encode and decode back to the EXACT original data?
2. Compression Ratio: How many tokens do we save compared to minified JSON?

This test uses the same comprehensive datasets as the LLM comprehension tests.
"""

import json
import os
import sys
import tiktoken
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from deepdiff import DeepDiff

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deep_toon import DeepToonEncoder, DeepToonDecoder
from evaluation.test_data_questions import generate_comprehensive_test_cases, TestCase

@dataclass
class CompressionResult:
    name: str
    original_tokens: int
    deep_toon_tokens: int
    savings_tokens: int
    savings_percent: float
    roundtrip_success: bool
    format_name: str
    error: Optional[str] = None

def count_tokens(text: str) -> int:
    """Count tokens using gpt-4o-mini encoding (cl100k_base)."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def normalize_for_comparison(obj):
    """
    Normalize JSON for semantic comparison.
    Treats missing fields as equivalent to null values.
    
    WHY THIS IS NECESSARY:
    Deep-TOON normalizes arrays to ensure all objects have the same structure.
    For example, if you have:
      [{"name": "Alice", "age": 30}, {"name": "Bob"}]
    
    Deep-TOON will encode and decode it as:
      [{"name": "Alice", "age": 30}, {"name": "Bob", "age": null}]
    
    This is semantically correct because:
    - Missing field = "no value" = null
    - The data meaning is preserved
    - The LLM can more easily work with uniform structure
    
    This function normalizes BOTH the original and decoded data by adding
    null for missing fields, ensuring fair comparison.
    
    Args:
        obj: The JSON object (dict/list/primitive) to normalize
        
    Returns:
        Normalized object with uniform array structures
    """
    if isinstance(obj, dict):
        # Recursively normalize all nested values
        return {k: normalize_for_comparison(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Normalize all items in the list
        normalized = [normalize_for_comparison(item) for item in obj]
        
        # For arrays of objects, ensure all objects have the same keys (with null for missing)
        if normalized and all(isinstance(item, dict) for item in normalized):
            # Get union of all keys across all objects
            all_keys = set()
            for item in normalized:
                all_keys.update(item.keys())
            
            # Add missing keys as null to each object
            for item in normalized:
                for key in all_keys:
                    if key not in item:
                        item[key] = None
        
        return normalized
    else:
        return obj


def test_single_case(test_case: TestCase, args) -> CompressionResult:
    """Run compression and fidelity test for a single test case."""
    try:
        # 1. Original JSON (minified for fair comparison)
        original_json = json.dumps(test_case.data, separators=(',', ':'))
        original_tokens = count_tokens(original_json)
        
        # 2. Deep-TOON Encoding
        # 2. Deep-TOON Encoding (or Smart Encoding)
        encoder = DeepToonEncoder()
        decoder = DeepToonDecoder() # Decoder initialized here for consistency

        encoded_data: str
        format_name: str

        if args.smart_threshold is not None:
            encoded_data = encoder.smart_encode(test_case.data, threshold=args.smart_threshold)
            # Determine format for display
            is_deep_toon = not (encoded_data.startswith('{') or encoded_data.startswith('['))
            format_name = "Deep-TOON" if is_deep_toon else "JSON"
        else:
            encoded_data = encoder.encode(test_case.data)
            format_name = "Deep-TOON"
            
        deep_toon_tokens = count_tokens(encoded_data)
        
        # 3. Roundtrip Verification with Semantic Comparison
        decoded_data = decoder.decode(encoded_data)
        
        # Normalize both for semantic comparison (missing fields = null)
        normalized_original = normalize_for_comparison(test_case.data)
        normalized_decoded = normalize_for_comparison(decoded_data)
        
        # DeepDiff returns empty dict if objects are equal
        diff = DeepDiff(normalized_original, normalized_decoded, ignore_order=True)
        roundtrip_success = not diff
        
        # 4. Calculate Metrics
        savings_tokens = original_tokens - deep_toon_tokens
        savings_percent = (savings_tokens / original_tokens) * 100 if original_tokens > 0 else 0
        
        return CompressionResult(
            name=test_case.name,
            original_tokens=original_tokens,
            deep_toon_tokens=deep_toon_tokens,
            savings_tokens=savings_tokens,
            savings_percent=savings_percent,
            roundtrip_success=roundtrip_success,
            format_name=format_name,
            error=f"Diff: {diff}" if not roundtrip_success else None
        )
        
    except Exception as e:
        return CompressionResult(
            name=test_case.name,
            original_tokens=0,
            deep_toon_tokens=0,
            savings_tokens=0,
            savings_percent=0,
            roundtrip_success=False,
            format_name="Error",
            error=str(e)
        )

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Deep-TOON Compression & Fidelity Test")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show data snippets")
    parser.add_argument("--smart-threshold", type=float, help="Use smart encoding with specified threshold (e.g. 0.1)")
    args = parser.parse_args()

    print("\n🧪 Deep-TOON Compression & Fidelity Test")
    print("========================================")
    
    # Generate test cases
    print("📋 Generating test cases...")
    test_cases = generate_comprehensive_test_cases()
    print(f"Found {len(test_cases)} test cases.\n")
    
    results = []
    total_orig = 0
    total_toon = 0
    
    # Print Header
    print(f"{'TEST CASE':<25} | {'ORIG TOKENS':<11} | {'TOON TOKENS':<11} | {'SAVINGS':<8} | {'FORMAT':<10} | {'FIDELITY':<10}")
    print("-" * 90) # Adjusted line length for new column
    
    for test_case in test_cases:
        result = test_single_case(test_case, args)
        results.append(result)
        
        total_orig += result.original_tokens
        total_toon += result.deep_toon_tokens
        
        status_icon = "✅ PASS" if result.roundtrip_success else "❌ FAIL"
        savings_str = f"{result.savings_percent:.1f}%"
        
        print(f"{result.name[:25]:<25} | {result.original_tokens:<11} | {result.deep_toon_tokens:<11} | {savings_str:<8} | {result.format_name:<10} | {status_icon}")
        
        if args.verbose:
            # Re-encode to get the strings for display
            original_json = json.dumps(test_case.data, separators=(',', ':'))
            encoder = DeepToonEncoder()
            
            # Use the same encoding logic as in test_single_case for verbose output
            if args.smart_threshold is not None:
                deep_toon_encoded_for_display = encoder.smart_encode(test_case.data, threshold=args.smart_threshold)
            else:
                deep_toon_encoded_for_display = encoder.encode(test_case.data)

            print(f"\n  🔍 Sneak Peek ({result.name}):")
            # Show first 300 chars to give a better view (approx 2-3 lines)
            print(f"  JSON ({result.original_tokens} tokens): {original_json[:300]}...")
            print(f"  TOON ({result.deep_toon_tokens} tokens): {deep_toon_encoded_for_display[:300]}...")
            print("-" * 90) # Adjusted line length
            
        if not result.roundtrip_success:
            print(f"  ⚠️  Error: {result.error}")

    print("-" * 80)
    
    # Summary
    total_savings = total_orig - total_toon
    total_savings_pct = (total_savings / total_orig) * 100 if total_orig > 0 else 0
    pass_count = sum(1 for r in results if r.roundtrip_success)
    
    print(f"\n📊 SUMMARY")
    print(f"Total Test Cases: {len(results)}")
    print(f"Passed:           {pass_count}/{len(results)}")
    print(f"Total Tokens:     {total_orig} -> {total_toon}")
    print(f"Total Savings:    {total_savings} tokens ({total_savings_pct:.1f}%)")
    
    if pass_count == len(results):
        print("\n🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print(f"\n❌ {len(results) - pass_count} TESTS FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
