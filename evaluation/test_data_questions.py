#!/usr/bin/env python3
"""
Fixed Test Data and Questions - Loads data from external JSON files
Includes programmatic Ground Truth verification logic.
"""

import json
import os
import math
from typing import Dict, List, Any, Tuple, Optional, Callable, Union
from dataclasses import dataclass


@dataclass
class Question:
    text: str
    expected_type: str
    # Function that takes the dataset and returns the exact expected answer
    ground_truth_func: Callable[[Dict], Any]


@dataclass
class TestCase:
    name: str
    data: Any
    questions: List[Question]
    complexity_level: str = 'simple'


def load_test_data(filename: str) -> Dict:
    """Load test data from JSON file."""
    filepath = os.path.join(os.path.dirname(__file__), 'test_data', filename)
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️  Warning: Test data file not found: {filename}")
        return {}
    except json.JSONDecodeError:
        print(f"⚠️  Warning: Invalid JSON in file: {filename}")
        return {}


def generate_comprehensive_test_cases() -> List[TestCase]:
    """
    Generate test cases by loading data from JSON files.
    Includes Ground Truth logic for every question.
    """
    test_cases = []
    
    # ============================================================================
    # ORIGINAL WORKING SIMPLE TESTS
    # ============================================================================
    
    # 1. User Demographics
    users_data = load_test_data('user_demographics.json')
    if users_data:
        test_cases.append(TestCase(
            name="User Demographics",
            data=users_data,
            questions=[
                Question(
                    text="Find the user with id=3 and provide only their exact 'address.city' value.",
                    expected_type='text',
                    ground_truth_func=lambda d: next((u['address']['city'] for u in d['users'] if u['id'] == 3), "NOT_FOUND")
                ),
                Question(
                    text="Look at all users and find the one with the longest 'firstName' (most characters). If multiple users have the same longest length, provide the first one found. Provide only that person's exact 'firstName'.",
                    expected_type='text',
                    # Find first user with maximum firstName length (handles ties)
                    ground_truth_func=lambda d: next(u['firstName'] for u in d['users'] if len(u['firstName']) == max(len(u['firstName']) for u in d['users']))
                ),
                Question(
                    text="Find the user whose 'company.title' contains the word 'Manager' and provide only their exact 'lastName'. (If multiple, provide the first one found)",
                    expected_type='text',
                    ground_truth_func=lambda d: next((u['lastName'] for u in d['users'] if 'Manager' in u['company']['title']), "NOT_FOUND")
                ),
                Question(
                    text="Look at the 'userAgent' fields and find which one mentions 'Chrome'. Provide only the exact 'username' of that user. (If multiple, provide the first one found)",
                    expected_type='text',
                    ground_truth_func=lambda d: next((u['username'] for u in d['users'] if 'Chrome' in u['userAgent']), "NOT_FOUND")
                )
            ],
            complexity_level='simple'
        ))
    
    # 2. Product Catalog
    products_data = load_test_data('product_catalog.json')
    if products_data:
        test_cases.append(TestCase(
            name="Product Catalog",
            data=products_data,
            questions=[
                Question(
                    text="Find the product with id=7 and provide only its exact 'brand' value.",
                    expected_type='text',
                    ground_truth_func=lambda d: next((p.get('brand', 'N/A') for p in d['products'] if p['id'] == 7), "NOT_FOUND")
                ),
                Question(
                    text="Look at all product descriptions and find which one mentions the word 'whisking'. Provide only that product's exact 'title'. (If multiple, provide the first one found)",
                    expected_type='text',
                    ground_truth_func=lambda d: next((p['title'] for p in d['products'] if 'whisking' in p['description'].lower()), "NOT_FOUND")
                ),
                Question(
                    text="Find the product that has 'moisturizing' in its description and provide only its exact 'category' value. (If multiple, provide the first one found)",
                    expected_type='text',
                    ground_truth_func=lambda d: next((p['category'] for p in d['products'] if 'moisturizing' in p['description'].lower()), "NOT_FOUND")
                ),
                Question(
                    text="Look at the 'tags' arrays and find a product that has 'lipstick' as one of its tags. Provide only that product's exact 'brand' value. (If multiple, provide the first one found)",
                    expected_type='text',
                    ground_truth_func=lambda d: next((p.get('brand', 'N/A') for p in d['products'] if 'lipstick' in [t.lower() for t in p.get('tags', [])]), "NOT_FOUND")
                )
            ],
            complexity_level='simple'
        ))
    
    # 3. E-commerce Orders
    orders_data = load_test_data('e_commerce_orders.json')
    if orders_data:
        test_cases.append(TestCase(
            name="E-commerce Orders",
            data=orders_data,
            questions=[
                Question(
                    text="Find the order with id='ORD-002' and provide only the exact 'customer.name' value.",
                    expected_type='text',
                    ground_truth_func=lambda d: next((o['customer']['name'] for o in d['orders'] if o['id'] == 'ORD-002'), "NOT_FOUND")
                ),
                Question(
                    text="Look at all items across all orders and find one that has 'Book' as its product name. Provide only the exact order 'id' that contains this item. (If multiple, provide the first one found)",
                    expected_type='text',
                    ground_truth_func=lambda d: next((o['id'] for o in d['orders'] for i in o['items'] if i['product'] == 'Book'), "NOT_FOUND")
                ),
                Question(
                    text="Find the customer who lives in 'LA' and provide only their exact 'age' value. (If multiple, provide the first one found)",
                    expected_type='number',
                    ground_truth_func=lambda d: next((o['customer']['age'] for o in d['orders'] if o['customer']['city'] == 'LA'), "NOT_FOUND")
                ),
                Question(
                    text="Look at all orders and find which one has the highest total price. Provide only the exact 'id' of that order.",
                    expected_type='text',
                    ground_truth_func=lambda d: max(d['orders'], key=lambda o: o['totals']['total'])['id']
                )
            ],
            complexity_level='simple'
        ))
    
    # 4. Customer Reviews
    reviews_data = load_test_data('customer_reviews.json')
    if reviews_data:
        test_cases.append(TestCase(
            name="Customer Reviews",
            data=reviews_data,
            questions=[
                Question(
                    text="Read all review comments and find the one that sounds most negative (rating 1). Provide only the exact 'customer' name who wrote that review. (If multiple, provide the first one found)",
                    expected_type='text',
                    ground_truth_func=lambda d: next((r['customer'] for r in d['reviews'] if r['rating'] == 1), "NOT_FOUND")
                ),
                Question(
                    text="Find the review with id=3 and provide only the exact 'product' name that was reviewed.",
                    expected_type='text',
                    ground_truth_func=lambda d: next((r['product'] for r in d['reviews'] if r['id'] == 3), "NOT_FOUND")
                ),
                Question(
                    text="Look at all comments and find which one mentions the word 'quality'. Provide only the exact 'rating' value of that review. (If multiple, provide the first one found)",
                    expected_type='number',
                    ground_truth_func=lambda d: next((r['rating'] for r in d['reviews'] if 'quality' in r['comment'].lower()), "NOT_FOUND")
                ),
                Question(
                    text="Find the review with the highest rating. Provide only the exact 'product' name. (If multiple, provide the first one found)",
                    expected_type='text',
                    ground_truth_func=lambda d: max(d['reviews'], key=lambda r: r['rating'])['product']
                )
            ],
            complexity_level='simple'
        ))
    
    # ============================================================================
    # ADDITIONAL COMPLEX ANALYTICAL TESTS
    # ============================================================================
    
    # 5. Advanced Analytics Test Case
    advanced_data = load_test_data('advanced_analytics.json')
    if advanced_data:
        def avg_salary_engineering(d):
            sals = [e['salary'] for e in d['employees'] if e['department'] == 'Engineering']
            return round(sum(sals) / len(sals)) if sals else 0

        def avg_salary_female_25_35(d):
            sals = [e['salary'] for e in d['employees'] 
                   if e['gender'] == 'F' and 25 <= e['age'] <= 35]
            return round(sum(sals) / len(sals)) if sals else 0

        def count_by_dept(d):
            counts = {}
            for e in d['employees']:
                counts[e['department']] = counts.get(e['department'], 0) + 1
            # Return sorted list of strings for consistent comparison
            return sorted([f"{k}: {v}" for k, v in counts.items()])

        def dept_highest_avg_salary(d):
            dept_sals = {}
            for e in d['employees']:
                if e['department'] not in dept_sals:
                    dept_sals[e['department']] = []
                dept_sals[e['department']].append(e['salary'])
            
            avgs = {k: sum(v)/len(v) for k, v in dept_sals.items()}
            return max(avgs.items(), key=lambda x: x[1])[0]

        test_cases.append(TestCase(
            name="Advanced Analytics",
            data=advanced_data,
            questions=[
                Question(
                    text="Calculate the average salary of employees in the 'Engineering' department. Round to the nearest dollar and provide only the integer.",
                    expected_type='number',
                    ground_truth_func=avg_salary_engineering
                ),
                Question(
                    text="Find the average salary of female employees between ages 25-35. Round to the nearest dollar and provide only the integer.",
                    expected_type='number',
                    ground_truth_func=avg_salary_female_25_35
                ),
                Question(
                    text="Count how many employees work in each department. Format as 'Department: count' on separate lines.",
                    expected_type='list',
                    ground_truth_func=count_by_dept
                ),
                Question(
                    text="Find the department with the highest average salary. Provide only the exact department name.",
                    expected_type='text',
                    ground_truth_func=dept_highest_avg_salary
                )
            ],
            complexity_level='complex'
        ))
    
    # ============================================================================
    # ADDITIONAL DATASETS FROM COMPREHENSIVE TEST (for fidelity/compression only)
    # ============================================================================
    
    # 6. Flat Object Array
    flat_data = load_test_data('flat_object_array.json')
    if flat_data:
        test_cases.append(TestCase(
            name="Flat Object Array",
            data=flat_data,
            questions=[], complexity_level='simple'
        ))

    # 7. Single Level Nesting
    nested_data = load_test_data('single_level_nesting.json')
    if nested_data:
        test_cases.append(TestCase(
            name="Single Level Nesting",
            data=nested_data,
            questions=[], complexity_level='simple'
        ))

    # 8. Deep Nesting (3+ levels)
    deep_nested_data = load_test_data('deep_nesting.json')
    if deep_nested_data:
        test_cases.append(TestCase(
            name="Deep Nesting",
            data=deep_nested_data,
            questions=[], complexity_level='complex'
        ))

    # 9. Mixed Data Types
    mixed_data = load_test_data('mixed_data_types.json')
    if mixed_data:
        test_cases.append(TestCase(
            name="Mixed Data Types",
            data=mixed_data,
            questions=[], complexity_level='complex'
        ))

    # 10. Sparse Data (lots of nulls)
    sparse_data = load_test_data('sparse_data.json')
    if sparse_data:
        test_cases.append(TestCase(
            name="Sparse Data",
            data=sparse_data,
            questions=[], complexity_level='simple'
        ))

    # 11. Array of Primitives (Skipped generation, but might exist from before or be empty)
    primitives_data = load_test_data('array_of_primitives.json')
    if primitives_data:
        test_cases.append(TestCase(
            name="Array of Primitives",
            data=primitives_data,
            questions=[], complexity_level='simple'
        ))

    # 12. Deeply Nested Arrays
    matrix_data = load_test_data('deeply_nested_arrays.json')
    if matrix_data:
        test_cases.append(TestCase(
            name="Deeply Nested Arrays",
            data=matrix_data,
            questions=[], complexity_level='complex'
        ))
    
    return test_cases

if __name__ == "__main__":
    # Self-verification when run directly
    print("Verifying Ground Truth Logic...")
    cases = generate_comprehensive_test_cases()
    for case in cases:
        print(f"\nChecking {case.name}...")
        for q in case.questions:
            try:
                answer = q.ground_truth_func(case.data)
                print(f"  Q: {q.text[:50]}... -> Answer: {answer}")
            except Exception as e:
                print(f"  ❌ ERROR in ground truth: {e}")