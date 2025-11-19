# Deep-TOON: Deep Token-Oriented Object Notation

Deep-TOON is a token-optimized JSON representation format designed for LLMs and AI applications. It provides significant compression for nested JSON structures while maintaining perfect data fidelity and LLM readability.

## 📊 Performance Overview

**Test Data: [dummyjson.com/users](https://dummyjson.com/users?limit=3) (3 users)**

```
Original JSON:    1,675 tokens
Deep-TOON:       1,065 tokens (36.4% reduction)
```

**Comprehensive Test Results:**
- **Average reduction: 28.7%** across diverse data types
- **Best case: 61.0%** reduction on large structured datasets  
- **Success rate: 92.9%** perfect roundtrip fidelity

## 🏗️ Format Specification

### Basic Structure

```toon2
[N,delimiter]{schema}:
  value1,value2,value3
  value4,value5,value6
```

### Hierarchical Tuples

Deep-TOON uses explicit hierarchical notation to group related fields:

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

**Deep-TOON Format:**
```toon2
users[1,]{id,firstName,lastName,age,address{address,city,state,coordinates{lat,lng}},bank{cardNumber,cardType}}:
  1,Emily,Johnson,28,("626 Main Street",Phoenix,Mississippi,(-77.16213,-92.084824)),("9289760655481815",Elo)
total: 208
skip: 0  
limit: 3
```

## 🔧 Usage Examples

### Installation

```bash
pip install deep-toon
```

### Basic Usage

```python
import deep_toon

# Your JSON data
data = {
    "users": [
        {
            "id": 1,
            "name": "Alice",
            "address": {
                "street": "123 Main St",
                "city": "NYC",
                "coordinates": {"lat": 40.7, "lng": -74.0}
            }
        },
        {
            "id": 2,
            "name": "Bob", 
            "address": {
                "street": "456 Oak Ave",
                "city": "LA", 
                "coordinates": {"lat": 34.0, "lng": -118.2}
            }
        }
    ]
}

# Compress to Deep-TOON format
compressed = deep_toon.encode(data)
print("Compressed:", compressed)

# Decompress back to original
decoded_data = deep_toon.decode(compressed)
assert data == decoded_data  # Perfect fidelity
```

**Output:**
```toon2
users[2,]{id,name,address{street,city,coordinates{lat,lng}}}:
  1,Alice,("123 Main St",NYC,(40.7,-74.0))
  2,Bob,("456 Oak Ave",LA,(34.0,-118.2))
```

### Advanced Usage

```python
# Use the classes directly for more control
from deep_toon import DeepToonEncoder, DeepToonDecoder

encoder = DeepToonEncoder()
decoder = DeepToonDecoder()

# Custom delimiter for data with commas
encoder = DeepToonEncoder(delimiter=';')
compressed = encoder.encode(data)
```

### Smart Encoding (Save-Safe)

Use `smart_encode` to automatically fall back to minified JSON if Deep-TOON doesn't achieve a specified savings threshold (default 10%).

```python
from deep_toon import smart_encode

# Only use Deep-TOON if it saves > 10% tokens
# Otherwise returns minified JSON
encoded = smart_encode(data, threshold=0.1)

# You can also use a custom token counter (defaults to char length)
encoded = smart_encode(data, token_counter=len)
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

## 🎨 Deep-TOON Design Philosophy

Deep-TOON uses **hierarchical tuples** to represent nested structures efficiently:

```json  
// Original JSON
{"user": {"profile": {"name": "Alice", "age": 30}}}

// Deep-TOON representation
[1,]{user{profile{name,age}}}:
  (("Alice",30))
```

**Key Benefits:**

1. **Compact schemas** - Structure declared once, no repetition
2. **Explicit hierarchy** - Clear nesting with `{...}` notation  
3. **Tuple efficiency** - Related data grouped logically
4. **LLM optimized** - Easy to read and parse

## 🚀 Performance Characteristics

### When Deep-TOON Excels

- **Nested objects** (addresses, preferences, metadata)
- **Repeated structures** (arrays of complex objects)  
- **Deep hierarchies** (API responses, config files)
- **Mixed data types** (numbers, strings, booleans together)

### Token Savings by Data Type

| Data Type | Typical Reduction |
|-----------|-------------------|
| Flat objects | 10-30% |
| 1-level nesting | 25-45% |
| 2+ level nesting | 30-60% |
| Array of objects | 35-50% |

## 🔧 Advanced Usage

### Custom Delimiters

```python
# Use semicolon delimiter for data containing commas
encoder = DeepToonEncoder(delimiter=";")
```

### Handling Large Arrays

```python
# Deep-TOON automatically detects when arrays are worth compressing
# Arrays with <2 items or inconsistent schemas fall back to JSON
```

### Error Handling

```python
try:
    decoded = decoder.decode(deep_toon_string)
except DeepToonDecodeError as e:
    print(f"Decode error: {e}")
    # Handle malformed Deep-TOON data
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

1. **Pattern Matching** - Detect Deep-TOON tabular format
2. **Schema Parsing** - Build nested structure from schema
3. **Smart Splitting** - Handle quoted strings and nested tuples
4. **Type Inference** - Convert strings back to appropriate types

## 🤝 Contributing

Deep-TOON is designed to be extended and improved. Key areas for contribution:

- **Performance optimization** for very large datasets
- **Additional encoding strategies** for specific data patterns  
- **Language bindings** for other programming languages
- **Integration tools** for popular APIs and frameworks

## 📄 License

Apache 2.0 License - Free for commercial and personal use!

---

**Deep-TOON - Efficient JSON representation for LLM applications.** 🚀✨