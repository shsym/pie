// bpe.hpp
#ifndef BPE_HPP
#define BPE_HPP

#include <string>
#include <vector>
#include <map>
#include <set>
#include <memory>

namespace bpe
{

    using Rank = uint32_t;

    class BytePairEncoder
    {
    public:
        // Constructor
        BytePairEncoder(
            const std::map<std::vector<uint8_t>, Rank> &encoder,
            const std::map<std::string, Rank> &special_tokens_encoder,
            const std::string &pattern);

        // Destructor
        ~BytePairEncoder();

        // Copy semantics
        BytePairEncoder(const BytePairEncoder &other);
        BytePairEncoder &operator=(const BytePairEncoder &other);

        // Move semantics
        BytePairEncoder(BytePairEncoder &&other) noexcept;
        BytePairEncoder &operator=(BytePairEncoder &&other) noexcept;

        /**
         * @brief Encodes a string into a sequence of token IDs.
         * @param text The input string to encode.
         * @param allowed_special A set of special tokens that should be encoded.
         * @return A vector of token IDs.
         */
        std::vector<Rank> encode(const std::string &text, const std::set<std::string> &allowed_special) const;

        /**
         * @brief Encodes a string into a sequence of token IDs, allowing all special tokens.
         * @param text The input string to encode.
         * @return A vector of token IDs.
         */
        std::vector<Rank> encode_with_special_tokens(const std::string &text) const;

        /**
         * @brief Decodes a sequence of token IDs back into a string.
         * @param tokens A vector of token IDs.
         * @return The decoded string.
         */
        std::string decode(const std::vector<Rank> &tokens) const;

    private:
        class BPEImpl;
        std::unique_ptr<BPEImpl> pimpl_;
    };

    /**
     * @brief Loads merge rules from a file.
     * @param path The path to the file containing merge rules.
     * @return A map from byte sequences to their ranks.
     */
    std::map<std::vector<uint8_t>, Rank> load_merge_rules(const std::string &path);

    /**
     * @brief Creates a BytePairEncoder configured for Llama3.
     * @param path The path to the Llama3 tokenizer model file.
     * @return A configured BytePairEncoder instance.
     */
    BytePairEncoder llama3_tokenizer(const std::string &path);

} // namespace bpe

#endif // BPE_HPP