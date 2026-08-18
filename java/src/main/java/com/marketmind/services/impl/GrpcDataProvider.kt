package com.marketmind.services.impl

import com.marketmind.services.DataProvider
import org.springframework.beans.factory.annotation.Value
import org.springframework.stereotype.Service

/**
 * JAVA → KOTLIN changes:
 *
 * 1. `@Value` on constructor parameter — idiomatic Kotlin/Spring: inject directly into the
 *    primary constructor rather than into a field. The effect is identical at runtime.
 *
 * 2. `override fun` — Kotlin requires the `override` keyword explicitly.
 *    Java's `@Override` annotation is optional; Kotlin's `override` modifier is mandatory.
 *    This prevents accidental overrides (compile error if the parent doesn't have the method).
 *
 * 3. String template — `"Data from gRPC at $endpoint"` replaces concatenation.
 */
@Service
class GrpcDataProvider(
    @Value("\${api.ib.endpoint}") private val endpoint: String
) : DataProvider {

    override fun fetchData(query: String): String = "Data from gRPC at $endpoint"
}
