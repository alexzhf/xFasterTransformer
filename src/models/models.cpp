// Copyright (c) 2023 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================
#include "models.h"

#include <string.h>

#include <stdexcept>

#include "INIReader.h"
#include "baichuan.h"
#include "chatglm.h"
#include "chatglm2.h"
#include "chatglm3.h"
#include "hybrid_model.h"
#include "llama.h"
#include "opt_decoder.h"
#include "searcher.h"

namespace xft {
enum class GenerationMode { GREEDY_SEARCH, BEAM_SEARCH, SAMPLE };

GenerationMode getGenerationMode(SearcherConfig &config_) {
    if (config_.numBeams == 1) {
        if (config_.doSample) {
            return GenerationMode::SAMPLE;
        } else {
            return GenerationMode::GREEDY_SEARCH;
        }
    } else if (config_.numBeams > 1) {
        return GenerationMode::BEAM_SEARCH;
    } else {
        printf("numBeams should greater than or equal to 1.\n");
        exit(-1);
    }
}

Model::Model() : decoder(nullptr), searcher(nullptr), isNewInput(true) {
    Env::setVerbose();
}

Model::~Model() {
    exitSlaves();
    if (decoder != nullptr) { delete decoder; }
    if (searcher != nullptr) { delete searcher; }
}

void Model::exitSlaves() {
    if (decoder->getRank() == 0) {
        configuration.numBeams = 0;
        Messenger &messenger = decoder->getMessenger();
        messenger.broadcast((int *)&configuration, sizeof(SearcherConfig) / sizeof(int));
    }
}

void Model::input(std::vector<int32_t> &inputIds_, int batchSize_) {
    isNewInput = true;
    Messenger &messenger = decoder->getMessenger();
    int dims[2];
    if (decoder->getRank() == 0) {
        dims[0] = batchSize_;
        dims[1] = inputIds_.size();
    }
    messenger.broadcast(dims, 2);
    batchSize = dims[0];
    seqLen = dims[1] / batchSize;

    inputIds.resize(dims[1]);
    if (decoder->getRank() == 0) { inputIds = inputIds_; }
    messenger.broadcast(inputIds.data(), dims[1]);
}

void Model::config(int maxLen_, int numBeams_, int numBeamHypsToKeep_, float lenPenalty_, bool doEarlyStopping_,
        int eosTokenId_, int padTokenId_, bool doSample_, float temperature_, int topK_, float topP_) {
    isNewInput = true;
    if (decoder->getRank() == 0) {
        configuration.maxLen = maxLen_;
        configuration.numBeams = numBeams_;
        configuration.numBeamHypsToKeep = numBeamHypsToKeep_;
        configuration.lenPenalty = lenPenalty_;
        configuration.doEarlyStopping = doEarlyStopping_;
        configuration.eosTokenId = eosTokenId_;
        configuration.padTokenId = padTokenId_;
        configuration.doSample = doSample_;
        configuration.temperature = temperature_;
        configuration.topK = topK_;
        configuration.topP = topP_;
    }
    Messenger &messenger = decoder->getMessenger();
    messenger.broadcast((int *)&configuration, sizeof(SearcherConfig) / sizeof(int));

    // Slaves get exit flags and exit directly
    if (decoder->getRank() > 0 && configuration.numBeams == 0) { exit(0); }

    createSearcher(configuration);
}

void Model::config(SearcherConfig &config_) {
    isNewInput = true;
    if (decoder->getRank() == 0) { configuration = config_; }
    Messenger &messenger = decoder->getMessenger();
    messenger.broadcast((int *)&configuration, sizeof(SearcherConfig) / sizeof(int));

    // Slaves get exit flags and exit directly
    if (decoder->getRank() > 0 && configuration.numBeams == 0) { exit(0); }

    createSearcher(configuration);
}

bool Model::isDone() {
    if (searcher == nullptr || inputIds.empty()) {
        printf("Please set input and config first.\n");
        exit(-1);
    }
    return !isNewInput && searcher->isDone();
}

std::vector<int32_t> Model::generate() {
    if (inputIds.empty()) {
        printf("Please set input tokens by model.input().\n");
        exit(-1);
    }
    if (searcher == nullptr) {
        printf("Please set generation config by model.config().\n");
        exit(-1);
    }

    if (isNewInput) {
        isNewInput = false;
        return searcher->getNextToken(inputIds.data(), batchSize, inputIds.size() / batchSize);
    } else {
        return searcher->getNextToken();
    }
}

void Model::createSearcher(SearcherConfig &config_) {
    if (searcher != nullptr) { delete searcher; }

    GenerationMode genMode = getGenerationMode(config_);
    if (genMode == GenerationMode::GREEDY_SEARCH) {
        searcher = new GreedySearch(*decoder, config_);
    } else if (genMode == GenerationMode::BEAM_SEARCH) {
        searcher = new BeamSearch(*decoder, config_);
    } else if (genMode == GenerationMode::SAMPLE) {
        searcher = new SampleSearch(*decoder, config_);
    }
}

int Model::getRank() {
    return decoder->getRank();
}

void Model::setDecoder(AbstractDecoder *dec) {
    decoder = dec;
}

void Model::setPrefix(std::vector<int32_t> &prefixIDs_) {
    Messenger &messenger = decoder->getMessenger();
    int prefixSeqLen = prefixIDs_.size();

    messenger.broadcast(&prefixSeqLen, 1);

    std::vector<int32_t> perfixIDs;
    perfixIDs.resize(prefixSeqLen);
    if (decoder->getRank() == 0) { perfixIDs = prefixIDs_; }
    messenger.broadcast(perfixIDs.data(), prefixSeqLen);

    decoder->setPrefix(perfixIDs.data(), prefixSeqLen);
}

void Model::unsetPrefix() {
    decoder->unsetPrefix();
}

AutoModel::AutoModel(std::string modelPath, xft::DataType datatype) : Model() {
    std::string configPath = modelPath + "/config.ini";
    INIReader reader = INIReader(configPath);

    if (reader.ParseError() < 0) {
        printf("Could not load model config.ini.\n");
        exit(-1);
    }
    std::string modeltype = *reader.Sections().begin();

    if (modeltype == "gpt") {
        switch (datatype) {
            case xft::DataType::fp16: setDecoder(new OptDecoder<float16_t>(modelPath)); break;
            case xft::DataType::bf16: setDecoder(new OptDecoder<bfloat16_t>(modelPath)); break;
            case xft::DataType::int8: setDecoder(new OptDecoder<int8_t>(modelPath)); break;
            case xft::DataType::w8a8: setDecoder(new OptDecoder<w8a8_t>(modelPath)); break;
            case xft::DataType::int4: setDecoder(new OptDecoder<uint4x2_t>(modelPath)); break;
            case xft::DataType::nf4: setDecoder(new OptDecoder<nf4x2_t>(modelPath)); break;
            case xft::DataType::bf16_fp16:
                setDecoder(new HybridModel<OptDecoder, bfloat16_t, float16_t>(modelPath));
                break;
            case xft::DataType::bf16_int8:
                setDecoder(new HybridModel<OptDecoder, bfloat16_t, int8_t>(modelPath));
                break;
            case xft::DataType::bf16_w8a8:
                setDecoder(new HybridModel<OptDecoder, bfloat16_t, w8a8_t>(modelPath));
                break;
            case xft::DataType::bf16_int4:
                setDecoder(new HybridModel<OptDecoder, bfloat16_t, uint4x2_t>(modelPath));
                break;
            case xft::DataType::bf16_nf4:
                setDecoder(new HybridModel<OptDecoder, bfloat16_t, nf4x2_t>(modelPath));
                break;
            case xft::DataType::w8a8_int8: setDecoder(new HybridModel<OptDecoder, w8a8_t, int8_t>(modelPath)); break;
            case xft::DataType::w8a8_int4: setDecoder(new HybridModel<OptDecoder, w8a8_t, uint4x2_t>(modelPath)); break;
            case xft::DataType::w8a8_nf4: setDecoder(new HybridModel<OptDecoder, w8a8_t, nf4x2_t>(modelPath)); break;
            default: printf("Unsupported data type.\n"); exit(-1);
        }
    } else if (modeltype == "llama") {
        switch (datatype) {
            case xft::DataType::fp16: setDecoder(new LlamaLLM<float16_t>(modelPath)); break;
            case xft::DataType::bf16: setDecoder(new LlamaLLM<bfloat16_t>(modelPath)); break;
            case xft::DataType::int8: setDecoder(new LlamaLLM<int8_t>(modelPath)); break;
            case xft::DataType::w8a8: setDecoder(new LlamaLLM<w8a8_t>(modelPath)); break;
            case xft::DataType::int4: setDecoder(new LlamaLLM<uint4x2_t>(modelPath)); break;
            case xft::DataType::nf4: setDecoder(new LlamaLLM<nf4x2_t>(modelPath)); break;
            case xft::DataType::bf16_fp16:
                setDecoder(new HybridModel<LlamaLLM, bfloat16_t, float16_t>(modelPath));
                break;
            case xft::DataType::bf16_int8: setDecoder(new HybridModel<LlamaLLM, bfloat16_t, int8_t>(modelPath)); break;
            case xft::DataType::bf16_w8a8: setDecoder(new HybridModel<LlamaLLM, bfloat16_t, w8a8_t>(modelPath)); break;
            case xft::DataType::bf16_int4:
                setDecoder(new HybridModel<LlamaLLM, bfloat16_t, uint4x2_t>(modelPath));
                break;
            case xft::DataType::bf16_nf4: setDecoder(new HybridModel<LlamaLLM, bfloat16_t, nf4x2_t>(modelPath)); break;
            case xft::DataType::w8a8_int8: setDecoder(new HybridModel<LlamaLLM, w8a8_t, int8_t>(modelPath)); break;
            case xft::DataType::w8a8_int4: setDecoder(new HybridModel<LlamaLLM, w8a8_t, uint4x2_t>(modelPath)); break;
            case xft::DataType::w8a8_nf4: setDecoder(new HybridModel<LlamaLLM, w8a8_t, nf4x2_t>(modelPath)); break;
            default: printf("Unsupported data type.\n"); exit(-1);
        }
    } else if (modeltype == "baichuan") {
        switch (datatype) {
            case xft::DataType::fp16: setDecoder(new Baichuan<float16_t>(modelPath)); break;
            case xft::DataType::bf16: setDecoder(new Baichuan<bfloat16_t>(modelPath)); break;
            case xft::DataType::int8: setDecoder(new Baichuan<int8_t>(modelPath)); break;
            case xft::DataType::w8a8: setDecoder(new Baichuan<w8a8_t>(modelPath)); break;
            case xft::DataType::int4: setDecoder(new Baichuan<uint4x2_t>(modelPath)); break;
            case xft::DataType::nf4: setDecoder(new Baichuan<nf4x2_t>(modelPath)); break;
            case xft::DataType::bf16_fp16:
                setDecoder(new HybridModel<Baichuan, bfloat16_t, float16_t>(modelPath));
                break;
            case xft::DataType::bf16_int8: setDecoder(new HybridModel<Baichuan, bfloat16_t, int8_t>(modelPath)); break;
            case xft::DataType::bf16_w8a8: setDecoder(new HybridModel<Baichuan, bfloat16_t, w8a8_t>(modelPath)); break;
            case xft::DataType::bf16_int4:
                setDecoder(new HybridModel<Baichuan, bfloat16_t, uint4x2_t>(modelPath));
                break;
            case xft::DataType::bf16_nf4: setDecoder(new HybridModel<Baichuan, bfloat16_t, nf4x2_t>(modelPath)); break;
            case xft::DataType::w8a8_int8: setDecoder(new HybridModel<Baichuan, w8a8_t, int8_t>(modelPath)); break;
            case xft::DataType::w8a8_int4: setDecoder(new HybridModel<Baichuan, w8a8_t, uint4x2_t>(modelPath)); break;
            case xft::DataType::w8a8_nf4: setDecoder(new HybridModel<Baichuan, w8a8_t, nf4x2_t>(modelPath)); break;
            default: printf("Unsupported data type.\n"); exit(-1);
        }
    } else if (modeltype == "chatglm") {
        switch (datatype) {
            case xft::DataType::fp16: setDecoder(new ChatGLM<float16_t>(modelPath)); break;
            case xft::DataType::bf16: setDecoder(new ChatGLM<bfloat16_t>(modelPath)); break;
            case xft::DataType::int8: setDecoder(new ChatGLM<int8_t>(modelPath)); break;
            case xft::DataType::w8a8: setDecoder(new ChatGLM<w8a8_t>(modelPath)); break;
            case xft::DataType::int4: setDecoder(new ChatGLM<uint4x2_t>(modelPath)); break;
            case xft::DataType::nf4: setDecoder(new ChatGLM<nf4x2_t>(modelPath)); break;
            case xft::DataType::bf16_fp16:
                setDecoder(new HybridModel<ChatGLM, bfloat16_t, float16_t>(modelPath));
                break;
            case xft::DataType::bf16_int8: setDecoder(new HybridModel<ChatGLM, bfloat16_t, int8_t>(modelPath)); break;
            case xft::DataType::bf16_w8a8: setDecoder(new HybridModel<ChatGLM, bfloat16_t, w8a8_t>(modelPath)); break;
            case xft::DataType::bf16_int4:
                setDecoder(new HybridModel<ChatGLM, bfloat16_t, uint4x2_t>(modelPath));
                break;
            case xft::DataType::bf16_nf4: setDecoder(new HybridModel<ChatGLM, bfloat16_t, nf4x2_t>(modelPath)); break;
            case xft::DataType::w8a8_int8: setDecoder(new HybridModel<ChatGLM, w8a8_t, int8_t>(modelPath)); break;
            case xft::DataType::w8a8_int4: setDecoder(new HybridModel<ChatGLM, w8a8_t, uint4x2_t>(modelPath)); break;
            case xft::DataType::w8a8_nf4: setDecoder(new HybridModel<ChatGLM, w8a8_t, nf4x2_t>(modelPath)); break;
            default: printf("Unsupported data type.\n"); exit(-1);
        }
    } else if (modeltype == "chatglm2") {
        switch (datatype) {
            case xft::DataType::fp16: setDecoder(new ChatGLM2<float16_t, RmsNorm>(modelPath)); break;
            case xft::DataType::bf16: setDecoder(new ChatGLM2<bfloat16_t, RmsNorm>(modelPath)); break;
            case xft::DataType::int8: setDecoder(new ChatGLM2<int8_t, RmsNorm>(modelPath)); break;
            case xft::DataType::w8a8: setDecoder(new ChatGLM2<w8a8_t, RmsNorm>(modelPath)); break;
            case xft::DataType::int4: setDecoder(new ChatGLM2<uint4x2_t, RmsNorm>(modelPath)); break;
            case xft::DataType::nf4: setDecoder(new ChatGLM2<nf4x2_t, RmsNorm>(modelPath)); break;
            case xft::DataType::bf16_fp16:
                setDecoder(new HybridModel<ChatGLM2, bfloat16_t, float16_t>(modelPath));
                break;
            case xft::DataType::bf16_int8: setDecoder(new HybridModel<ChatGLM2, bfloat16_t, int8_t>(modelPath)); break;
            case xft::DataType::bf16_w8a8: setDecoder(new HybridModel<ChatGLM2, bfloat16_t, w8a8_t>(modelPath)); break;
            case xft::DataType::bf16_int4:
                setDecoder(new HybridModel<ChatGLM2, bfloat16_t, uint4x2_t>(modelPath));
                break;
            case xft::DataType::bf16_nf4: setDecoder(new HybridModel<ChatGLM2, bfloat16_t, nf4x2_t>(modelPath)); break;
            case xft::DataType::w8a8_int8: setDecoder(new HybridModel<ChatGLM2, w8a8_t, int8_t>(modelPath)); break;
            case xft::DataType::w8a8_int4: setDecoder(new HybridModel<ChatGLM2, w8a8_t, uint4x2_t>(modelPath)); break;
            case xft::DataType::w8a8_nf4: setDecoder(new HybridModel<ChatGLM2, w8a8_t, nf4x2_t>(modelPath)); break;
            default: printf("Unsupported data type.\n"); exit(-1);
        }
    } else if (modeltype == "chatglm3") {
        switch (datatype) {
            case xft::DataType::fp16: setDecoder(new ChatGLM3<float16_t, RmsNorm>(modelPath)); break;
            case xft::DataType::bf16: setDecoder(new ChatGLM3<bfloat16_t, RmsNorm>(modelPath)); break;
            case xft::DataType::int8: setDecoder(new ChatGLM3<int8_t, RmsNorm>(modelPath)); break;
            case xft::DataType::w8a8: setDecoder(new ChatGLM3<w8a8_t, RmsNorm>(modelPath)); break;
            case xft::DataType::int4: setDecoder(new ChatGLM3<uint4x2_t, RmsNorm>(modelPath)); break;
            case xft::DataType::nf4: setDecoder(new ChatGLM3<nf4x2_t, RmsNorm>(modelPath)); break;
            case xft::DataType::bf16_fp16:
                setDecoder(new HybridModel<ChatGLM3, bfloat16_t, float16_t>(modelPath));
                break;
            case xft::DataType::bf16_int8: setDecoder(new HybridModel<ChatGLM3, bfloat16_t, int8_t>(modelPath)); break;
            case xft::DataType::bf16_w8a8: setDecoder(new HybridModel<ChatGLM3, bfloat16_t, w8a8_t>(modelPath)); break;
            case xft::DataType::bf16_int4:
                setDecoder(new HybridModel<ChatGLM3, bfloat16_t, uint4x2_t>(modelPath));
                break;
            case xft::DataType::bf16_nf4: setDecoder(new HybridModel<ChatGLM3, bfloat16_t, nf4x2_t>(modelPath)); break;
            case xft::DataType::w8a8_int8: setDecoder(new HybridModel<ChatGLM3, w8a8_t, int8_t>(modelPath)); break;
            case xft::DataType::w8a8_int4: setDecoder(new HybridModel<ChatGLM3, w8a8_t, uint4x2_t>(modelPath)); break;
            case xft::DataType::w8a8_nf4: setDecoder(new HybridModel<ChatGLM3, w8a8_t, nf4x2_t>(modelPath)); break;
            default: printf("Unsupported data type.\n"); exit(-1);
        }
    } else {
        printf("Unsupported data type.\n");
        exit(-1);
    }
}
} // namespace xft
