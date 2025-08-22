#pragma once

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <stdexcept>

// Simple JSON parser for configuration loading
// Note: This is a minimal parser for our specific use case

namespace config {

struct Value {
    enum Type { STRING, INT, FLOAT, BOOL, ARRAY, OBJECT, NULL_VAL };
    Type type = NULL_VAL;
    std::string string_val;
    int int_val = 0;
    float float_val = 0.0f;
    bool bool_val = false;
    std::vector<Value> array_val;
    std::map<std::string, Value> object_val;
    
    Value() = default;
    Value(const std::string& s) : type(STRING), string_val(s) {}
    Value(int i) : type(INT), int_val(i) {}
    Value(float f) : type(FLOAT), float_val(f) {}
    Value(bool b) : type(BOOL), bool_val(b) {}
    
    bool is_string() const { return type == STRING; }
    bool is_int() const { return type == INT; }
    bool is_float() const { return type == FLOAT; }
    bool is_bool() const { return type == BOOL; }
    bool is_array() const { return type == ARRAY; }
    bool is_object() const { return type == OBJECT; }
    bool is_null() const { return type == NULL_VAL; }
    
    std::string as_string() const {
        if (type == STRING) return string_val;
        throw std::runtime_error("Value is not a string");
    }
    
    int as_int() const {
        if (type == INT) return int_val;
        if (type == FLOAT) return static_cast<int>(float_val);
        throw std::runtime_error("Value is not an int");
    }
    
    float as_float() const {
        if (type == FLOAT) return float_val;
        if (type == INT) return static_cast<float>(int_val);
        throw std::runtime_error("Value is not a float");
    }
    
    bool as_bool() const {
        if (type == BOOL) return bool_val;
        throw std::runtime_error("Value is not a bool");
    }
    
    const std::vector<Value>& as_array() const {
        if (type == ARRAY) return array_val;
        throw std::runtime_error("Value is not an array");
    }
    
    const std::map<std::string, Value>& as_object() const {
        if (type == OBJECT) return object_val;
        throw std::runtime_error("Value is not an object");
    }
    
    const Value& operator[](const std::string& key) const {
        if (type != OBJECT) throw std::runtime_error("Value is not an object");
        auto it = object_val.find(key);
        if (it == object_val.end()) throw std::runtime_error("Key not found: " + key);
        return it->second;
    }
    
    const Value& operator[](size_t index) const {
        if (type != ARRAY) throw std::runtime_error("Value is not an array");
        if (index >= array_val.size()) throw std::runtime_error("Array index out of bounds");
        return array_val[index];
    }
    
    bool has(const std::string& key) const {
        if (type != OBJECT) return false;
        return object_val.find(key) != object_val.end();
    }
    
    size_t size() const {
        if (type == ARRAY) return array_val.size();
        if (type == OBJECT) return object_val.size();
        return 0;
    }
};

// Simple JSON parser
class Parser {
private:
    std::string text;
    size_t pos = 0;
    
    void skip_whitespace() {
        while (pos < text.size() && std::isspace(text[pos])) pos++;
    }
    
    char peek() {
        skip_whitespace();
        return pos < text.size() ? text[pos] : '\0';
    }
    
    char consume() {
        skip_whitespace();
        return pos < text.size() ? text[pos++] : '\0';
    }
    
    std::string parse_string() {
        if (consume() != '"') throw std::runtime_error("Expected '\"'");
        std::string result;
        while (pos < text.size() && text[pos] != '"') {
            if (text[pos] == '\\' && pos + 1 < text.size()) {
                pos++;
                switch (text[pos]) {
                    case 'n': result += '\n'; break;
                    case 't': result += '\t'; break;
                    case 'r': result += '\r'; break;
                    case '"': result += '"'; break;
                    case '\\': result += '\\'; break;
                    default: result += text[pos]; break;
                }
            } else {
                result += text[pos];
            }
            pos++;
        }
        if (pos >= text.size()) throw std::runtime_error("Unterminated string");
        pos++; // consume closing '"'
        return result;
    }
    
    Value parse_number() {
        std::string num;
        bool is_float = false;
        while (pos < text.size() && (std::isdigit(text[pos]) || text[pos] == '.' || text[pos] == '-' || text[pos] == 'e' || text[pos] == 'E')) {
            if (text[pos] == '.') is_float = true;
            num += text[pos++];
        }
        if (is_float) {
            return Value(std::stof(num));
        } else {
            return Value(std::stoi(num));
        }
    }
    
    Value parse_value() {
        char c = peek();
        if (c == '"') {
            return Value(parse_string());
        } else if (c == '{') {
            return parse_object();
        } else if (c == '[') {
            return parse_array();
        } else if (c == 't' || c == 'f') {
            return parse_bool();
        } else if (c == 'n') {
            return parse_null();
        } else if (std::isdigit(c) || c == '-') {
            return parse_number();
        } else {
            throw std::runtime_error("Unexpected character: " + std::string(1, c));
        }
    }
    
    Value parse_object() {
        if (consume() != '{') throw std::runtime_error("Expected '{'");
        Value obj;
        obj.type = Value::OBJECT;
        
        if (peek() == '}') {
            consume();
            return obj;
        }
        
        while (true) {
            std::string key = parse_string();
            if (consume() != ':') throw std::runtime_error("Expected ':'");
            Value value = parse_value();
            obj.object_val[key] = value;
            
            char c = peek();
            if (c == '}') {
                consume();
                break;
            } else if (c == ',') {
                consume();
            } else {
                throw std::runtime_error("Expected ',' or '}'");
            }
        }
        return obj;
    }
    
    Value parse_array() {
        if (consume() != '[') throw std::runtime_error("Expected '['");
        Value arr;
        arr.type = Value::ARRAY;
        
        if (peek() == ']') {
            consume();
            return arr;
        }
        
        while (true) {
            arr.array_val.push_back(parse_value());
            
            char c = peek();
            if (c == ']') {
                consume();
                break;
            } else if (c == ',') {
                consume();
            } else {
                throw std::runtime_error("Expected ',' or ']'");
            }
        }
        return arr;
    }
    
    Value parse_bool() {
        if (text.substr(pos, 4) == "true") {
            pos += 4;
            return Value(true);
        } else if (text.substr(pos, 5) == "false") {
            pos += 5;
            return Value(false);
        } else {
            throw std::runtime_error("Expected 'true' or 'false'");
        }
    }
    
    Value parse_null() {
        if (text.substr(pos, 4) == "null") {
            pos += 4;
            Value null_val;
            null_val.type = Value::NULL_VAL;
            return null_val;
        } else {
            throw std::runtime_error("Expected 'null'");
        }
    }
    
public:
    Value parse(const std::string& json) {
        text = json;
        pos = 0;
        return parse_value();
    }
};

Value load_json(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    
    Parser parser;
    return parser.parse(buffer.str());
}

} // namespace config