// Exact next-position option logits. JSONL in/out; no sampling or text decoding.
// Build against the same llama.cpp version used to convert the GGUF.
#include "llama.h"
#include "ggml-backend.h"
#include "json.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using json = nlohmann::json;

int main(int argc, char **argv) {
    if (argc != 2) { std::cerr << "usage: gguf_logits model.gguf\n"; return 2; }
    ggml_backend_load_all();
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 99;
    auto *model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;
    auto cp = llama_context_default_params();
    cp.n_ctx = 2048; cp.n_batch = 1024; cp.n_ubatch = 1024; cp.n_seq_max = 1;
    cp.n_threads = 8; cp.n_threads_batch = 8;
    auto *ctx = llama_init_from_model(model, cp);
    if (!ctx) { llama_model_free(model); return 4; }
    auto *vocab = llama_model_get_vocab(model);
    std::vector<llama_token> labels;
    for (char c = 'A'; c <= 'Z'; ++c) {
        std::string label = std::string(" ") + c;
        llama_token token[4];
        int n = llama_tokenize(vocab, label.data(), label.size(), token, 4, false, false);
        if (n != 1) { std::cerr << "option is not a single token\n"; return 5; }
        labels.push_back(token[0]);
    }
    std::string line;
    while (std::getline(std::cin, line)) {
        try {
            if (line.size() > 200000) throw std::runtime_error("request too large");
            auto in = json::parse(line);
            std::string prompt = in.at("prompt").get<std::string>();
            int k = in.at("n_options").get<int>();
            if (k < 2 || k > 26) throw std::runtime_error("need 2-26 options");
            std::vector<llama_token> tokens(1025);
            int n = llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), tokens.size(), false, true);
            if (n <= 0 || n > 1024) throw std::runtime_error("prompt exceeds the validated context budget");
            tokens.resize(n);
            llama_memory_clear(llama_get_memory(ctx), true);
            auto batch = llama_batch_get_one(tokens.data(), n);
            if (llama_decode(ctx, batch) != 0) throw std::runtime_error("prefill failed");
            llama_synchronize(ctx);
            float *all = llama_get_logits_ith(ctx, -1);
            if (!all) throw std::runtime_error("missing logits");
            std::vector<float> logits;
            for (int i = 0; i < k; ++i) {
                if (!std::isfinite(all[labels[i]])) throw std::runtime_error("nonfinite option logit");
                logits.push_back(all[labels[i]]);
            }
            std::cout << json({{"logits", logits}, {"prompt_tokens", n}}).dump() << std::endl;
        } catch (const std::exception &e) {
            std::cout << json({{"error", e.what()}}).dump() << std::endl;
        }
    }
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
