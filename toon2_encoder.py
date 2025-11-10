#!/usr/bin/env python3
"""
TOON2 Clean Implementation - Final Specification

Based on design decision for Option A: Explicit Hierarchical Tuples
Target: 40-60% token reduction on real-world nested JSON
"""

import json
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
import re


@dataclass
class TupleGroup:
    """Represents a group of fields that can be compressed into a tuple."""
    name: str
    fields: List[str]
    nested_groups: List['TupleGroup']
    
    def to_schema_string(self) -> str:
        """Generate schema string like 'address{street,city,coords{lat,lng}}'"""
        if not self.fields and not self.nested_groups:
            return self.name
        
        parts = []
        parts.extend(self.fields)
        
        for nested in self.nested_groups:
            parts.append(nested.to_schema_string())
        
        if parts:
            return f"{self.name}{{{','.join(parts)}}}"
        return self.name


class Toon2Encoder:
    """Clean TOON2 encoder implementing Option A specification."""
    
    def __init__(self, delimiter: str = ","):
        self.delimiter = delimiter
    
    def encode(self, data: Any) -> str:
        """Main encoding entry point."""
        if isinstance(data, list) and self._can_compress_array(data):
            return self._encode_array(data)
        elif isinstance(data, dict):
            return self._encode_object(data, 0)
        else:
            return json.dumps(data)  # Fallback for primitives
    
    def _can_compress_array(self, arr: List[Any]) -> bool:
        """Check if array is suitable for TOON2 compression."""
        if not arr:
            return False
        
        # Must be array of objects (at least 1 item)
        if not all(isinstance(item, dict) for item in arr):
            return False
        
        # Single item arrays can be compressed too
        if len(arr) == 1:
            return True
        
        # Check schema consistency (60% field overlap for multiple items)
        first_keys = set(arr[0].keys())
        if not first_keys:  # Empty objects
            return False
            
        consistent_count = 0
        
        for obj in arr:
            obj_keys = set(obj.keys())
            if not obj_keys:  # Skip empty objects
                continue
            union_keys = first_keys | obj_keys
            if len(union_keys) == 0:
                continue
            overlap = len(first_keys & obj_keys) / len(union_keys)
            if overlap >= 0.6:  # Lower threshold for more flexibility
                consistent_count += 1
        
        return consistent_count >= len(arr) * 0.6
    
    def _encode_array(self, arr: List[Dict[str, Any]]) -> str:
        """Encode array using TOON2 format."""
        # Build optimal schema
        schema = self._build_optimal_schema(arr)
        
        # Generate header
        schema_parts = [group.to_schema_string() for group in schema]
        header = f"[{len(arr)},{self.delimiter}]{{{','.join(schema_parts)}}}:"
        
        # Generate data rows
        rows = []
        for obj in arr:
            row_values = []
            for group in schema:
                value = self._extract_group_value(obj, group)
                encoded = self._encode_value(value, group)
                row_values.append(encoded)
            
            rows.append(self.delimiter.join(row_values))
        
        # Combine header and rows
        result = header
        for row in rows:
            result += f"\n  {row}"
        
        return result
    
    def _build_optimal_schema(self, arr: List[Dict[str, Any]]) -> List[TupleGroup]:
        """Build optimal schema with hierarchical tuple groups."""
        if not arr:
            return []
        
        # Analyze field structure
        field_analysis = self._analyze_fields(arr)
        
        # Build tuple groups
        groups = []
        
        # Group fields by structure type
        primitive_fields = []
        nested_objects = {}
        
        for field_name, field_info in field_analysis.items():
            if field_info['type'] in ['primitive', 'array', 'mixed']:
                # Treat arrays and mixed types as primitive for now
                primitive_fields.append(field_name)
            elif field_info['type'] == 'object':
                nested_objects[field_name] = field_info
        
        # Add primitive fields as individual groups
        for field in primitive_fields:
            groups.append(TupleGroup(field, [], []))
        
        # Build nested object groups
        for obj_name, obj_info in nested_objects.items():
            nested_group = self._build_nested_group(obj_name, obj_info)
            groups.append(nested_group)
        
        return groups
    
    def _analyze_fields(self, arr: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Analyze field types and structures across all objects."""
        field_analysis = {}
        
        # Sample from all objects to understand field types
        for obj in arr:
            for key, value in obj.items():
                field_type = self._get_field_type(value)
                
                if key not in field_analysis:
                    field_analysis[key] = {
                        'type': field_type,
                        'structure': self._analyze_structure(value),
                        'count': 1,
                        'samples': [value]
                    }
                else:
                    field_analysis[key]['count'] += 1
                    # Keep track of different types seen for this field
                    if field_type != field_analysis[key]['type']:
                        field_analysis[key]['type'] = 'mixed'
                    # Keep some samples for analysis
                    if len(field_analysis[key]['samples']) < 3:
                        field_analysis[key]['samples'].append(value)
        
        return field_analysis
    
    def _get_field_type(self, value: Any) -> str:
        """Determine the type of a field value."""
        if isinstance(value, dict):
            return 'object'
        elif isinstance(value, list):
            return 'array'
        else:
            return 'primitive'
    
    def _analyze_structure(self, value: Any) -> Dict[str, Any]:
        """Analyze the internal structure of complex values."""
        if isinstance(value, dict):
            structure = {'fields': {}}
            for k, v in value.items():
                structure['fields'][k] = {
                    'type': self._get_field_type(v),
                    'structure': self._analyze_structure(v) if isinstance(v, (dict, list)) else None
                }
            return structure
        elif isinstance(value, list):
            return {'array_type': 'mixed'}  # Simplified for now
        else:
            return {}
    
    def _build_nested_group(self, name: str, info: Dict[str, Any]) -> TupleGroup:
        """Build a nested tuple group for complex objects."""
        fields = []
        nested_groups = []
        
        if 'structure' in info and 'fields' in info['structure']:
            for field_name, field_info in info['structure']['fields'].items():
                if field_info['type'] == 'primitive':
                    fields.append(field_name)
                elif field_info['type'] == 'object':
                    # Recursively build nested groups
                    nested_group = self._build_nested_group(field_name, field_info)
                    nested_groups.append(nested_group)
        
        return TupleGroup(name, fields, nested_groups)
    
    def _extract_group_value(self, obj: Dict[str, Any], group: TupleGroup) -> Any:
        """Extract value for a tuple group from an object."""
        if not group.fields and not group.nested_groups:
            # Simple field
            return obj.get(group.name)
        
        # Complex group - extract tuple components
        base_obj = obj.get(group.name, {})
        if not isinstance(base_obj, dict):
            return None
        
        tuple_values = []
        
        # Add primitive fields (handle missing as null)
        for field in group.fields:
            tuple_values.append(base_obj.get(field))
        
        # Add nested groups
        for nested_group in group.nested_groups:
            nested_value = self._extract_group_value(base_obj, nested_group)
            tuple_values.append(nested_value)
        
        return tuple(tuple_values) if tuple_values else None
    
    def _encode_value(self, value: Any, group: TupleGroup) -> str:
        """Encode a value according to its group type."""
        if value is None:
            return "null"
        
        if isinstance(value, tuple):
            # Encode tuple
            encoded_items = []
            for item in value:
                if isinstance(item, tuple):
                    # Nested tuple
                    nested_encoded = [self._encode_primitive(sub_item) for sub_item in item]
                    encoded_items.append(f"({','.join(nested_encoded)})")
                else:
                    encoded_items.append(self._encode_primitive(item))
            
            return f"({','.join(encoded_items)})"
        else:
            # Simple value
            return self._encode_primitive(value)
    
    def _encode_primitive(self, value: Any) -> str:
        """Encode a primitive value with proper quoting."""
        if value is None:
            return "null"
        elif isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, str):
            # Quote if necessary
            if self._needs_quoting(value):
                escaped = value.replace('\\', '\\\\').replace('"', '\\"')
                return f'"{escaped}"'
            return value
        elif isinstance(value, (list, dict)):
            # Use JSON encoding for arrays and objects
            return json.dumps(value)
        else:
            return str(value)
    
    def _needs_quoting(self, s: str) -> bool:
        """Check if string needs quoting."""
        if not s:
            return True
        
        # Contains delimiters or structural characters
        if any(char in s for char in [self.delimiter, '(', ')', '[', ']', '{', '}', ':', '-']):
            return True
        
        # Contains whitespace
        if any(char.isspace() for char in s):
            return True
        
        # Looks like reserved words
        if s.lower() in ['null', 'true', 'false']:
            return True
        
        # Looks like number
        if re.match(r'^-?\d+(\.\d+)?([eE][+-]?\d+)?$', s):
            return True
        
        return False
    
    def _encode_object(self, obj: Dict[str, Any], depth: int) -> str:
        """Encode standalone object (fallback to simple format)."""
        lines = []
        processed_keys = set()
        
        # First pass: handle arrays that can be compressed
        for key, value in obj.items():
            if isinstance(value, list) and self._can_compress_array(value):
                # Use TOON2 for arrays
                encoded_array = self._encode_array(value)
                lines.append(f"{key}{encoded_array}")
                processed_keys.add(key)
            elif isinstance(value, dict):
                # Nested object - encode recursively
                nested_encoded = self._encode_object(value, depth + 1)
                if nested_encoded.strip():
                    lines.append(f"{key}:")
                    for line in nested_encoded.split('\n'):
                        lines.append(f"  {line}")
                processed_keys.add(key)
        
        # Second pass: handle all other fields (primitives, non-compressible arrays)
        for key, value in obj.items():
            if key not in processed_keys:
                if isinstance(value, list):
                    # Non-compressible array - encode as JSON
                    encoded_val = json.dumps(value)
                else:
                    # Primitive value
                    encoded_val = self._encode_primitive(value)
                lines.append(f"{key}: {encoded_val}")
        
        return '\n'.join(lines) if lines else json.dumps(obj)


# Test function
def test_toon2_clean():
    """Test the clean TOON2 implementation."""
    
    # Test data similar to dummyjson structure
    test_data = [
        {
            "id": 1,
            "firstName": "Emily",
            "lastName": "Johnson", 
            "age": 28,
            "address": {
                "address": "626 Main Street",
                "city": "Phoenix",
                "state": "Mississippi",
                "coordinates": {"lat": -77.16213, "lng": -92.084824}
            },
            "bank": {
                "cardNumber": "9289760655481815",
                "cardType": "Elo"
            }
        },
        {
            "id": 2,
            "firstName": "Michael",
            "lastName": "Williams",
            "age": 35, 
            "address": {
                "address": "385 Fifth Street",
                "city": "Houston",
                "state": "Alabama",
                "coordinates": {"lat": 22.815468, "lng": 115.608581}
            },
            "bank": {
                "cardNumber": "6737807858721625", 
                "cardType": "Elo"
            }
        }
    ]
    
    encoder = Toon2Encoder()
    encoded = encoder.encode(test_data)
    
    print("🧪 TOON2 Clean Encoder Test")
    print("=" * 40)
    print("Encoded:")
    print(encoded)
    print()
    
    # Calculate rough compression
    original = json.dumps(test_data)
    print(f"Original length: {len(original)}")
    print(f"TOON2 length: {len(encoded)}")
    print(f"Compression: {(len(original) - len(encoded)) / len(original) * 100:.1f}%")


if __name__ == "__main__":
    test_toon2_clean()