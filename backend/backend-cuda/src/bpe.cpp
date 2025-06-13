// bpe.cpp
#include "bpe.hpp"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <regex>
#include <iterator>
#include <cstring> // For strchr

namespace bpe
{

    // --- Helper functions (internal linkage) ---
    namespace
    {

        // Self-contained Base64 decoder
        std::string base64_decode(const std::string &in)
        {
            std::string out;
            std::vector<int> T(256, -1);
            const std::string B64_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            for (int i = 0; i < 64; i++)
                T[B64_CHARS[i]] = i;

            int val = 0, valb = -8;
            for (unsigned char c : in)
            {
                if (T[c] == -1)
                    break;
                val = (val << 6) + T[c];
                valb += 6;
                if (valb >= 0)
                {
                    out.push_back(char((val >> valb) & 0xFF));
                    valb -= 8;
                }
            }
            return out;
        }

        std::vector<std::pair<size_t, Rank>> byte_pair_merge(const std::vector<uint8_t> &piece, const std::map<std::vector<uint8_t>, Rank> &ranks)
        {
            std::vector<std::pair<size_t, Rank>> parts;
            if (piece.empty())
            {
                return parts;
            }
            for (size_t i = 0; i <= piece.size(); ++i)
            {
                parts.emplace_back(i, std::numeric_limits<Rank>::max());
            }

            auto get_rank_for_pair = [&](size_t start, size_t end) -> Rank
            {
                if (start + 1 >= end)
                    return std::numeric_limits<Rank>::max();
                std::vector<uint8_t> sub(piece.begin() + start, piece.begin() + end);
                auto it = ranks.find(sub);
                return (it != ranks.end()) ? it->second : std::numeric_limits<Rank>::max();
            };

            for (size_t i = 0; i < parts.size() - 2; ++i)
            {
                parts[i].second = get_rank_for_pair(parts[i].first, parts[i + 1].first);
            }

            while (parts.size() > 1)
            {
                Rank min_rank = std::numeric_limits<Rank>::max();
                size_t min_rank_idx = -1;
                for (size_t i = 0; i < parts.size() - 1; ++i)
                {
                    if (parts[i].second < min_rank)
                    {
                        min_rank = parts[i].second;
                        min_rank_idx = i;
                    }
                }
                if (min_rank == std::numeric_limits<Rank>::max())
                {
                    break;
                }

                // Merge the best pair
                parts.erase(parts.begin() + min_rank_idx + 1);

                // Update ranks around the merge point
                if (min_rank_idx > 0)
                {
                    parts[min_rank_idx - 1].second = get_rank_for_pair(parts[min_rank_idx - 1].first, parts[min_rank_idx].first);
                }
                if (min_rank_idx < parts.size() - 1)
                {
                    parts[min_rank_idx].second = get_rank_for_pair(parts[min_rank_idx].first, parts[min_rank_idx + 1].first);
                }
            }
            return parts;
        }

        std::vector<Rank> byte_pair_encode(const std::vector<uint8_t> &piece, const std::map<std::vector<uint8_t>, Rank> &ranks)
        {
            if (piece.size() == 1)
            {
                auto it = ranks.find(piece);
                if (it == ranks.end())
                    throw std::runtime_error("Single byte token not found in ranks.");
                return {it->second};
            }
            auto merged_parts = byte_pair_merge(piece, ranks);
            std::vector<Rank> tokens;
            tokens.reserve(merged_parts.size());
            for (size_t i = 0; i < merged_parts.size() - 1; ++i)
            {
                std::vector<uint8_t> sub_piece(piece.begin() + merged_parts[i].first, piece.begin() + merged_parts[i + 1].first);
                auto it = ranks.find(sub_piece);
                if (it == ranks.end())
                    throw std::runtime_error("Token not found in ranks after merge.");
                tokens.push_back(it->second);
            }
            return tokens;
        }
    } // namespace

    // --- PImpl Definition ---
    class BytePairEncoder::BPEImpl
    {
    public:
        std::map<std::vector<uint8_t>, Rank> encoder_;
        std::map<std::string, Rank> special_tokens_encoder_;
        std::map<Rank, std::vector<uint8_t>> decoder_;
        std::map<Rank, std::vector<uint8_t>> special_tokens_decoder_;
        std::regex regex_;
        std::regex special_regex_;

        BPEImpl(const std::map<std::vector<uint8_t>, Rank> &encoder,
                const std::map<std::string, Rank> &special_tokens_encoder,
                const std::string &pattern)
            : encoder_(encoder), special_tokens_encoder_(special_tokens_encoder)
        {

            try
            {
                // The icase flag makes the entire regex case-insensitive
                regex_ = std::regex(pattern, std::regex::ECMAScript | std::regex::icase);
            }
            catch (const std::regex_error &e)
            {
                throw std::runtime_error(std::string("Regex error for main pattern: ") + e.what());
            }

            for (const auto &[key, value] : encoder_)
            {
                decoder_[value] = key;
            }

            if (!special_tokens_encoder_.empty())
            {
                std::string special_regex_pattern;
                for (const auto &[key, value] : special_tokens_encoder_)
                {
                    special_tokens_decoder_[value] = std::vector<uint8_t>(key.begin(), key.end());
                    if (!special_regex_pattern.empty())
                    {
                        special_regex_pattern += "|";
                    }
                    // Escape special regex characters in the token string
                    std::string escaped_key;
                    for (char c : key)
                    {
                        if (strchr(".+*?^${}()[]|\\", c))
                        {
                            escaped_key += '\\';
                        }
                        escaped_key += c;
                    }
                    special_regex_pattern += escaped_key;
                }
                try
                {
                    special_regex_ = std::regex(special_regex_pattern);
                }
                catch (const std::regex_error &e)
                {
                    throw std::runtime_error(std::string("Regex error for special tokens: ") + e.what());
                }
            }
        }
    };

    // --- BytePairEncoder Method Implementations ---

    BytePairEncoder::BytePairEncoder(
        const std::map<std::vector<uint8_t>, Rank> &encoder,
        const std::map<std::string, Rank> &special_tokens_encoder,
        const std::string &pattern) : pimpl_(std::make_unique<BPEImpl>(encoder, special_tokens_encoder, pattern)) {}

    BytePairEncoder::~BytePairEncoder() = default;
    BytePairEncoder::BytePairEncoder(const BytePairEncoder &other) : pimpl_(std::make_unique<BPEImpl>(*other.pimpl_)) {}
    BytePairEncoder &BytePairEncoder::operator=(const BytePairEncoder &other)
    {
        if (this != &other)
        {
            pimpl_ = std::make_unique<BPEImpl>(*other.pimpl_);
        }
        return *this;
    }
    BytePairEncoder::BytePairEncoder(BytePairEncoder &&other) noexcept = default;
    BytePairEncoder &BytePairEncoder::operator=(BytePairEncoder &&other) noexcept = default;

    std::vector<Rank> BytePairEncoder::encode(const std::string &text, const std::set<std::string> &allowed_special) const
    {
        std::vector<Rank> tokens;

        auto process_chunk = [&](const std::string::const_iterator &begin, const std::string::const_iterator &end)
        {
            if (begin >= end)
                return;
            auto words_begin = std::sregex_iterator(begin, end, pimpl_->regex_);
            auto words_end = std::sregex_iterator();
            for (auto it = words_begin; it != words_end; ++it)
            {
                std::string match_str = it->str();
                std::vector<uint8_t> piece(match_str.begin(), match_str.end());
                auto enc_it = pimpl_->encoder_.find(piece);
                if (enc_it != pimpl_->encoder_.end())
                {
                    tokens.push_back(enc_it->second);
                }
                else
                {
                    auto piece_tokens = byte_pair_encode(piece, pimpl_->encoder_);
                    tokens.insert(tokens.end(), piece_tokens.begin(), piece_tokens.end());
                }
            }
        };

        if (pimpl_->special_tokens_encoder_.empty() || allowed_special.empty())
        {
            process_chunk(text.cbegin(), text.cend());
            return tokens;
        }

        auto special_begin = std::sregex_iterator(text.cbegin(), text.cend(), pimpl_->special_regex_);
        auto special_end = std::sregex_iterator();

        auto last_pos = text.cbegin();
        for (auto it = special_begin; it != special_end; ++it)
        {
            std::string special_token = it->str();

            if (allowed_special.count(special_token))
            {
                // *** THE FIX IS HERE ***
                // Process the text chunk before this special token.
                // The end of the prefix is the beginning of the current match.
                process_chunk(last_pos, it->prefix().second);

                // Add the special token
                tokens.push_back(pimpl_->special_tokens_encoder_.at(special_token));

                // Move the cursor past this special token
                last_pos = it->suffix().first;
            }
        }

        // Process the final chunk of text after the last special token
        process_chunk(last_pos, text.cend());

        return tokens;
    }

    std::vector<Rank> BytePairEncoder::encode_with_special_tokens(const std::string &text) const
    {
        std::set<std::string> allowed_special;
        for (const auto &[key, value] : pimpl_->special_tokens_encoder_)
        {
            allowed_special.insert(key);
        }
        return encode(text, allowed_special);
    }

    std::string BytePairEncoder::decode(const std::vector<Rank> &tokens) const
    {
        std::vector<uint8_t> decoded_bytes;
        for (const auto &token : tokens)
        {
            auto dec_it = pimpl_->decoder_.find(token);
            if (dec_it != pimpl_->decoder_.end())
            {
                const auto &bytes = dec_it->second;
                decoded_bytes.insert(decoded_bytes.end(), bytes.begin(), bytes.end());
            }
            else
            {
                auto sdec_it = pimpl_->special_tokens_decoder_.find(token);
                if (sdec_it != pimpl_->special_tokens_decoder_.end())
                {
                    const auto &bytes = sdec_it->second;
                    decoded_bytes.insert(decoded_bytes.end(), bytes.begin(), bytes.end());
                }
                else
                {
                    throw std::runtime_error("Invalid token for decoding: " + std::to_string(token));
                }
            }
        }
        return std::string(decoded_bytes.begin(), decoded_bytes.end());
    }

    // --- Standalone Function Implementations ---
    std::map<std::vector<uint8_t>, Rank> load_merge_rules(const std::string &path)
    {
        std::ifstream file(path);
        if (!file.is_open())
        {
            throw std::runtime_error("Cannot open merge rules file: " + path);
        }
        std::string line;
        std::map<std::vector<uint8_t>, Rank> ranks;
        int line_num = 0;
        while (std::getline(file, line))
        {
            line_num++;
            if (line.empty() || line[0] == '#')
                continue;

            std::string b64_token;
            Rank rank;
            std::stringstream ss(line);
            if (!(ss >> b64_token >> rank))
            {
                throw std::runtime_error("Error parsing line " + std::to_string(line_num));
            }

            std::string decoded_str = base64_decode(b64_token);
            ranks[{decoded_str.begin(), decoded_str.end()}] = rank;
        }
        return ranks;
    }

    BytePairEncoder llama3_tokenizer(const std::string &path)
    {
        auto mergeable_ranks = load_merge_rules(path);
        std::vector<std::string> special_tokens = {
            "<|begin_of_text|>", "<|end_of_text|>", "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>", "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>", "<|start_header_id|>", "<|end_header_id|>",
            "<|reserved_special_token_4|>", "<|eot_id|>"};

        // Using portable ASCII-based regex pattern
        std::string pattern = R"('s|'t|'re|'ve|'m|'ll|'d|[^\r\na-z0-9]?[a-z]+|[0-9]{1,3}| ?[^\sa-z0-9]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

        Rank num_base_tokens = 128000;
        std::map<std::string, Rank> special_tokens_encoder;
        for (size_t i = 0; i < special_tokens.size(); ++i)
        {
            special_tokens_encoder[special_tokens[i]] = num_base_tokens + i;
        }

        return BytePairEncoder(mergeable_ranks, special_tokens_encoder, pattern);
    }

} // namespace bpe