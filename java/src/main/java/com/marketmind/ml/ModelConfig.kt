package com.marketmind.ml

import org.springframework.beans.factory.annotation.Value
import org.springframework.context.annotation.Configuration

/**
 * JAVA → KOTLIN changes:
 *
 * 1. `@Value` constructor injection — cleaner than field injection.
 *    `val` fields (immutable) because model config properties shouldn't change after startup.
 *
 * 2. No getters needed — Kotlin auto-generates `getModelVersion()` and `getLearningRate()`
 *    for Java interop. Kotlin callers access them as `config.modelVersion` directly.
 */
@Configuration
class ModelConfig(
    @Value("\${app.model.version}") val modelVersion: String,
    @Value("\${app.learning-rate}") val learningRate: Double
)
