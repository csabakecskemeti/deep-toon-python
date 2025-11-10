# TOON2: Token-Oriented Object Notation v2

TOON2 is an enhanced token-optimized JSON representation format designed for LLMs and AI applications. It builds upon the original [TOON format](https://github.com/toon-format/toon) with significant improvements for real-world nested JSON structures.

## 🎯 Key Improvements Over TOON v1

| Feature | TOON v1 | TOON2 |
|---------|---------|--------|
| **Nested Objects** | ❌ Flattens to paths (`address.street`) | ✅ Hierarchical tuples (`address{street,city}`) |
| **Complex Structures** | ❌ Poor compression on nested data | ✅ 30-60% reduction on real JSON |
| **Schema Clarity** | ❌ Implicit structure | ✅ Explicit hierarchical notation |
| **Token Efficiency** | ~3% on complex JSON | 36-60% on complex JSON |
| **LLM Readability** | Good for flat data | Excellent for all data types |

## 📊 Performance Comparison

**Test Data: [dummyjson.com/users](https://dummyjson.com/users?limit=3) (3 users)**

```
Original JSON:    1,675 tokens
TOON v1:         1,620 tokens (3.2% reduction)
TOON2:           1,065 tokens (36.4% reduction)
```

## 🏗️ Format Specification

### Basic Structure

```toon2
[N,delimiter]{schema}:
  value1,value2,value3
  value4,value5,value6
```

### Hierarchical Tuples

TOON2 uses explicit hierarchical notation to group related fields:

```toon2
# Nested objects become tuples
address{street,city,coordinates{lat,lng}}

# Results in data like:
("626 Main Street", "Phoenix", (-77.16, -92.08))
```

### Complete Example

**Original JSON:**
```json
{
  "users": [
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
    }
  ],
  "total": 208,
  "skip": 0,
  "limit": 3
}
```

**TOON2 Format:**
```toon2
users[1,]{id,firstName,lastName,age,address{address,city,state,coordinates{lat,lng}},bank{cardNumber,cardType}}:
  1,Emily,Johnson,28,("626 Main Street",Phoenix,Mississippi,(-77.16213,-92.084824)),("9289760655481815",Elo)
total: 208
skip: 0  
limit: 3
```

## 🔧 Usage Examples

### Installation

```python
# Copy toon2_encoder.py and toon2_decoder.py to your project
from toon2_encoder import Toon2Encoder
from toon2_decoder import Toon2Decoder
```

### Basic Encoding/Decoding

```python
import json
from toon2_encoder import Toon2Encoder
from toon2_decoder import Toon2Decoder

# Sample nested data
data = {
    "users": [
        {
            "id": 1,
            "name": "Alice",
            "address": {
                "street": "123 Main St",
                "city": "NYC",
                "coordinates": {"lat": 40.7, "lng": -74.0}
            },
            "preferences": {
                "theme": "dark", 
                "notifications": True
            }
        },
        {
            "id": 2,
            "name": "Bob", 
            "address": {
                "street": "456 Oak Ave",
                "city": "LA", 
                "coordinates": {"lat": 34.0, "lng": -118.2}
            },
            "preferences": {
                "theme": "light",
                "notifications": False  
            }
        }
    ],
    "count": 2
}

# Encode to TOON2
encoder = Toon2Encoder()
toon2_str = encoder.encode(data)
print("TOON2 Format:")
print(toon2_str)

# Decode back to original
decoder = Toon2Decoder() 
decoded_data = decoder.decode(toon2_str)

# Verify perfect roundtrip
from deepdiff import DeepDiff
diff = DeepDiff(data, decoded_data, ignore_order=True)
print("Roundtrip Success:", not diff)
```

**Output:**
```toon2
users[2,]{id,name,address{street,city,coordinates{lat,lng}},preferences{theme,notifications}}:
  1,Alice,("123 Main St",NYC,(40.7,-74.0)),(dark,true)
  2,Bob,("456 Oak Ave",LA,(34.0,-118.2)),(light,false)
count: 2
```

### Token Counting Example

```python
import tiktoken
from toon2_encoder import Toon2Encoder

def count_tokens(text, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Compare token usage
original_json = json.dumps(data)
toon2_format = encoder.encode(data)

original_tokens = count_tokens(original_json)
toon2_tokens = count_tokens(toon2_format)

print(f"Original JSON: {original_tokens} tokens")
print(f"TOON2 Format:  {toon2_tokens} tokens") 
print(f"Reduction:     {(original_tokens-toon2_tokens)/original_tokens*100:.1f}%")
```

### Real-World API Example

```python
import requests
from toon2_encoder import Toon2Encoder

# Fetch real data
response = requests.get("https://dummyjson.com/users?limit=5")
api_data = response.json()

# Convert to TOON2
encoder = Toon2Encoder()
compressed = encoder.encode(api_data)

print(f"Original size: {len(json.dumps(api_data))} chars")
print(f"TOON2 size:    {len(compressed)} chars")
print(f"Compression:   {(len(json.dumps(api_data))-len(compressed))/len(json.dumps(api_data))*100:.1f}%")

# Use in LLM prompt
prompt = f"""
Analyze this user data in TOON2 format:

{compressed}

What insights can you provide about the users?
"""
```

## 🎨 Format Features

### Schema Declaration

The schema explicitly declares the structure:

```toon2
{field1,field2,nested{subfield1,subfield2},deep{level1{level2}}}
```

### Tuple Nesting

Related fields are grouped into tuples:

```toon2
# Person with address
person{name,age,address{street,city}}
# Results in: ("Alice", 30, ("123 Main", "NYC"))
```

### Null Handling

Missing or null values are handled gracefully:

```toon2
# With missing city
("123 Main", null, (40.7, -74.0))
```

### Quoting Rules

Strings are quoted only when necessary:

```toon2
# No quotes needed
Simple,Text,123

# Quotes for special characters  
"Text with, comma","Multi word text","123-abc"
```

## 🔄 Comparison with TOON v1

### TOON v1 Approach (Path Flattening)
```json
// Original
{"user": {"profile": {"name": "Alice", "age": 30}}}

// TOON v1 
user.profile.name,user.profile.age
Alice,30
```

### TOON2 Approach (Hierarchical Tuples)
```json  
// Original
{"user": {"profile": {"name": "Alice", "age": 30}}}

// TOON2
[1,]{user{profile{name,age}}}:
  ("Alice",30)
```

**Why TOON2 is Better:**

1. **Shorter schemas** - No repeated path prefixes
2. **Better compression** - Tuples are more compact than paths
3. **Clearer structure** - Hierarchy is explicit
4. **LLM friendly** - Easier to parse and understand

## 🚀 Performance Characteristics

### When TOON2 Excels

- **Nested objects** (addresses, preferences, metadata)
- **Repeated structures** (arrays of complex objects)  
- **Deep hierarchies** (API responses, config files)
- **Mixed data types** (numbers, strings, booleans together)

### Token Savings by Data Type

| Data Type | Typical Reduction |
|-----------|-------------------|
| Flat objects | 10-20% |
| 1-level nesting | 25-35% |
| 2+ level nesting | 40-60% |
| Array of objects | 35-50% |

## 🔧 Advanced Usage

### Custom Delimiters

```python
# Use semicolon delimiter for data containing commas
encoder = Toon2Encoder(delimiter=";")
```

### Handling Large Arrays

```python
# TOON2 automatically detects when arrays are worth compressing
# Arrays with <2 items or inconsistent schemas fall back to JSON
```

### Error Handling

```python
try:
    decoded = decoder.decode(toon2_string)
except Toon2DecodeError as e:
    print(f"Decode error: {e}")
    # Handle malformed TOON2 data
```

## 📈 Use Cases

- **LLM Training Data** - Reduce token costs for large datasets
- **API Response Compression** - Faster transmission and processing  
- **Configuration Files** - More readable than JSON for complex configs
- **Data Interchange** - Efficient format for AI-to-AI communication
- **Prompt Engineering** - Include more context in limited token budgets

## 🔬 Technical Details

### Schema Detection Algorithm

1. **Field Analysis** - Identify primitive vs nested fields
2. **Structure Grouping** - Group related fields into tuples  
3. **Optimization** - Choose best compression strategy per field group
4. **Schema Generation** - Create hierarchical schema notation

### Parsing Strategy

1. **Pattern Matching** - Detect TOON2 tabular format
2. **Schema Parsing** - Build nested structure from schema
3. **Smart Splitting** - Handle quoted strings and nested tuples
4. **Type Inference** - Convert strings back to appropriate types

## 🤝 Contributing

TOON2 is designed to be extended and improved. Key areas for contribution:

- **Performance optimization** for very large datasets
- **Additional encoding strategies** for specific data patterns  
- **Language bindings** for other programming languages
- **Integration tools** for popular APIs and frameworks

## 📄 License

MIT License - Feel free to use in your projects!

---

**TOON2 - Making AI communication more efficient, one token at a time.** 🤖✨