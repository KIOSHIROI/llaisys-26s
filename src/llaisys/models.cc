#include "llaisys/models/qwen2.h"

#include "../models/qwen2/qwen2_model.hpp"

__C {
    struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
        int device_id = 0;
        if (device_ids != nullptr && ndevice > 0) {
            device_id = device_ids[0];
        }
        return new LlaisysQwen2Model(*meta, device, device_id);
    }

    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
        delete model;
    }

    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
        return model->weights();
    }

    int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
        return model->infer(token_ids, ntoken);
    }
}
