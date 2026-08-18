package com.marketmind.services.impl

import com.marketmind.services.InferenceProvider
import org.springframework.stereotype.Service

@Service
class GrpcInferenceProvider : InferenceProvider {
    override fun performInference(input: String): String = "Inference result from gRPC"
}
