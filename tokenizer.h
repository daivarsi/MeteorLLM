#pragma once
#include <string>
#include <vector>
#include <cstdint>
namespace vitaqwen {
class Tokenizer {
public:
 void load(const std::string&path);
 std::vector<int> encode(const std::string&text)const;
 std::string decode(int token)const;
private:
 void bpe(const std::string&text,size_t start,size_t end,std::vector<int>&out)const;
 int special_at(const std::string&,size_t,size_t&)const;
 std::vector<uint8_t>data;
 uint32_t count=0,merge_count=0,special_count=0,offset_base=0,blob_base=0,blob_size=0,byte_base=0,special_base=0,merge_base=0;
};
}
