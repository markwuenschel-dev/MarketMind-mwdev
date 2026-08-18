package com.marketmind.services

interface InferenceProvider {
    fun performInference(input: String): String
}
