#!/usr/bin/env python3
"""
TOON2 Clean Decoder - Final Specification

Decoder for the clean TOON2 implementation with hierarchical tuples.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple


class Toon2DecodeError(Exception):
    """Error during TOON2 decoding."""
    pass


class Toon2Decoder:
    """Clean TOON2 decoder for hierarchical tuple format."""
    
    def decode(self, toon2_str: str) -> Any:
        """Main decoding entry point."""
        lines = toon2_str.strip().split('\n')
        if not lines:
            return None
        
        first_line = lines[0].strip()
        
        # Check if it's TOON2 tabular format
        tabular_match = re.match(r'^(.*?)\[(\d+),([^]]*)\]\{(.*)\}:$', first_line)
        if tabular_match:
            key, length, delimiter, schema = tabular_match.groups()
            key = key.strip() if key else None
            delimiter = delimiter or ','
            
            # Parse schema and decode array
            schema_groups = self._parse_schema(schema)
            
            # Determine how many lines belong to the array
            array_lines_count = int(length)
            array_data = self._decode_tabular_array(lines[1:array_lines_count+1], int(length), delimiter, schema_groups)
            
            # Check if there are additional lines after the array
            remaining_lines = lines[array_lines_count+1:]
            
            if remaining_lines:
                # Parse remaining lines as simple format
                remaining_str = '\n'.join(remaining_lines)
                additional_data = self._parse_simple_format(remaining_str)
                
                if key and isinstance(additional_data, dict):
                    result = {key: array_data}
                    result.update(additional_data)
                    return result
                elif key:
                    return {key: array_data}
                else:
                    return array_data
            else:
                if key:
                    return {key: array_data}
                else:
                    return array_data
        
        # Fallback to simple parsing
        return self._parse_simple_format(toon2_str)
    
    def _parse_schema(self, schema_str: str) -> List[Dict[str, Any]]:
        """Parse schema string into structured format."""
        groups = []
        current_pos = 0
        
        while current_pos < len(schema_str):
            group, next_pos = self._parse_single_group(schema_str, current_pos)
            groups.append(group)
            current_pos = next_pos
            
            # Skip comma separator
            if current_pos < len(schema_str) and schema_str[current_pos] == ',':
                current_pos += 1
        
        return groups
    
    def _parse_single_group(self, schema_str: str, start_pos: int) -> Tuple[Dict[str, Any], int]:
        """Parse a single group from schema string."""
        pos = start_pos
        
        # Find the group name
        name_start = pos
        while pos < len(schema_str) and schema_str[pos] not in '{,':
            pos += 1
        
        name = schema_str[name_start:pos].strip()
        
        # Check if it has nested structure
        if pos < len(schema_str) and schema_str[pos] == '{':
            # Parse nested structure
            brace_count = 1
            content_start = pos + 1
            pos += 1
            
            while pos < len(schema_str) and brace_count > 0:
                if schema_str[pos] == '{':
                    brace_count += 1
                elif schema_str[pos] == '}':
                    brace_count -= 1
                pos += 1
            
            content = schema_str[content_start:pos-1]
            nested_groups = self._parse_schema(content)
            
            return {
                'name': name,
                'type': 'tuple',
                'nested': nested_groups
            }, pos
        else:
            # Simple field
            return {
                'name': name,
                'type': 'simple'
            }, pos
    
    def _decode_tabular_array(self, data_lines: List[str], length: int, delimiter: str, schema_groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Decode tabular array data."""
        objects = []
        
        for i in range(min(length, len(data_lines))):
            line = data_lines[i].strip()
            if line.startswith('  '):
                line = line[2:]  # Remove indentation
            
            # Split by delimiter while respecting parentheses and quotes
            values = self._smart_split(line, delimiter)
            
            if len(values) != len(schema_groups):
                raise Toon2DecodeError(f"Row {i+1} has {len(values)} values but expected {len(schema_groups)}")
            
            # Decode each value according to its schema
            obj = {}
            for value, group in zip(values, schema_groups):
                decoded_value = self._decode_group_value(value.strip(), group)
                obj[group['name']] = decoded_value
            
            objects.append(obj)
        
        return objects
    
    def _smart_split(self, line: str, delimiter: str) -> List[str]:
        """Split line by delimiter while respecting parentheses, brackets, braces and quotes."""
        parts = []
        current = ""
        in_quotes = False
        paren_depth = 0
        bracket_depth = 0  # For JSON arrays [...]
        brace_depth = 0   # For JSON objects {...}
        
        i = 0
        while i < len(line):
            char = line[i]
            
            if char == '"' and (i == 0 or line[i-1] != '\\'):
                in_quotes = not in_quotes
                current += char
            elif not in_quotes:
                if char == '(':
                    paren_depth += 1
                    current += char
                elif char == ')':
                    paren_depth -= 1
                    current += char
                elif char == '[':
                    bracket_depth += 1
                    current += char
                elif char == ']':
                    bracket_depth -= 1
                    current += char
                elif char == '{':
                    brace_depth += 1
                    current += char
                elif char == '}':
                    brace_depth -= 1
                    current += char
                elif char == delimiter and paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
                    parts.append(current)
                    current = ""
                else:
                    current += char
            else:
                current += char
            
            i += 1
        
        if current:
            parts.append(current)
        
        return parts
    
    def _decode_group_value(self, value_str: str, group: Dict[str, Any]) -> Any:
        """Decode value according to group schema."""
        if group['type'] == 'simple':
            return self._decode_primitive(value_str)
        elif group['type'] == 'tuple':
            return self._decode_tuple_value(value_str, group['nested'])
        else:
            return self._decode_primitive(value_str)
    
    def _decode_tuple_value(self, value_str: str, nested_groups: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Decode tuple value into object."""
        if value_str == "null" or not value_str.strip():
            return {}
        
        # Remove outer parentheses
        if value_str.startswith('(') and value_str.endswith(')'):
            content = value_str[1:-1]
        else:
            content = value_str
        
        # Split tuple content
        values = self._smart_split(content, ',')
        
        # Build object from values and nested groups
        obj = {}
        for i, (value, group) in enumerate(zip(values, nested_groups)):
            if i < len(values):
                decoded_value = self._decode_group_value(value.strip(), group)
                obj[group['name']] = decoded_value
        
        return obj
    
    def _decode_primitive(self, value_str: str) -> Any:
        """Decode primitive value."""
        value_str = value_str.strip()
        
        if not value_str or value_str == 'null':
            return None
        elif value_str == 'true':
            return True
        elif value_str == 'false':
            return False
        
        # Handle JSON arrays and objects
        if (value_str.startswith('[') and value_str.endswith(']')) or \
           (value_str.startswith('{') and value_str.endswith('}')):
            try:
                return json.loads(value_str)
            except json.JSONDecodeError:
                pass  # Fall through to string handling
        
        # Handle quoted strings (both single and double quotes)
        if (value_str.startswith('"') and value_str.endswith('"')) or \
           (value_str.startswith("'") and value_str.endswith("'")):
            try:
                # Try JSON parsing first (for double quotes)
                return json.loads(value_str)
            except json.JSONDecodeError:
                # Fallback: remove quotes (works for both single and double)
                return value_str[1:-1]
        
        # Try to parse as number
        try:
            if '.' in value_str or 'e' in value_str.lower():
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            pass
        
        return value_str
    
    def _parse_simple_format(self, toon2_str: str) -> Any:
        """Fallback parser for non-tabular formats."""
        lines = toon2_str.strip().split('\n')
        result = {}
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Check if this line starts an embedded TOON2 array
            tabular_pattern = r'^(.*?)\[(\d+),([^]]*)\]\{(.*)\}:$'
            tabular_match = re.match(tabular_pattern, line)
            
            if tabular_match:
                key, length, delimiter, schema = tabular_match.groups()
                key = key.strip() if key else None
                delimiter = delimiter or ','
                
                # Collect all subsequent indented lines for this array
                array_lines = [line]
                i += 1
                while i < len(lines) and lines[i].startswith('  '):
                    array_lines.append(lines[i])
                    i += 1
                
                # Parse schema and decode array directly (avoid recursive call)
                schema_groups = self._parse_schema(schema)
                array_data = self._decode_tabular_array(array_lines[1:], int(length), delimiter, schema_groups)
                
                if key:
                    result[key] = array_data
                else:
                    result = array_data
                continue
            
            # Handle simple key: value pairs
            elif ':' in line and not line.endswith(':'):
                # Make sure this isn't a TOON2 tabular format line
                if not re.match(tabular_pattern, line):
                    key, value = line.split(':', 1)
                    value = value.strip()
                    
                    # Check if value looks like JSON array or object
                    if value.startswith('[') or value.startswith('{'):
                        try:
                            result[key.strip()] = json.loads(value)
                        except json.JSONDecodeError:
                            result[key.strip()] = self._decode_primitive(value)
                    else:
                        result[key.strip()] = self._decode_primitive(value)
            
            i += 1
        
        return result if result else toon2_str


# Test function  
def test_roundtrip():
    """Test complete roundtrip with your notebook example."""
    import requests
    from toon2_encoder import Toon2Encoder
    import tiktoken
    from deepdiff import DeepDiff
    
    def fetch_dummyjson_users(limit=3):
        response = requests.get(f'https://dummyjson.com/users?limit={limit}')
        return response.json()
    
    def count_tokens(text):
        encoding = tiktoken.encoding_for_model("gpt-4")
        return len(encoding.encode(text))
    
    print("🧪 COMPLETE ROUNDTRIP TEST")
    print("=" * 50)
    
    # Get data
    data = fetch_dummyjson_users(3)
    original_tokens = count_tokens(json.dumps(data))
    
    # Encode  
    encoder = Toon2Encoder()
    encoded = encoder.encode(data)
    toon_tokens = count_tokens(encoded)
    
    print(f"📥 Original tokens: {original_tokens}")
    print(f"📤 TOON2 tokens: {toon_tokens}")
    print(f"💰 Reduction: {(original_tokens - toon_tokens)/original_tokens*100:.1f}%")
    print()
    
    # Decode
    decoder = Toon2Decoder()
    decoded = decoder.decode(encoded)
    
    print(f"🔄 Roundtrip check:")
    print(f"  Original type: {type(data)}")
    print(f"  Decoded type: {type(decoded)}")
    
    # Compare with DeepDiff
    diff = DeepDiff(data, decoded, ignore_order=True)
    
    if not diff:
        print("  ✅ PERFECT ROUNDTRIP!")
        return True
    else:
        print("  ❌ Differences found:")
        print(f"  Types: {list(diff.keys())}")
        return False


if __name__ == "__main__":
    test_roundtrip()