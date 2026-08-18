package com.marketmind.config

import com.marketmind.features.dashboard.DashboardViewModel
import com.marketmind.services.DataFetchService
import com.marketmind.utils.LogUtils
import io.github.resilience4j.circuitbreaker.CircuitBreakerConfig
import io.github.resilience4j.circuitbreaker.CircuitBreakerRegistry
import io.grpc.StatusRuntimeException
import io.micrometer.core.instrument.MeterRegistry
import io.micrometer.core.instrument.simple.SimpleMeterRegistry
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import java.time.Duration
import java.util.*
import java.util.function.Supplier

/**
 * JAVA → KOTLIN changes:
 *
 * 1. Anonymous object — `object : Supplier<String> { ... }` replaces Java's
 *    `new Supplier<String>() { ... }`. Kotlin uses `object : InterfaceName { ... }` syntax.
 *    The anonymous object body works the same as an anonymous inner class in Java.
 *
 * 2. `@Synchronized` annotation — Kotlin has no `synchronized` keyword.
 *    `@Synchronized` on a function is equivalent to Java's `synchronized(this) { ... }` block
 *    applied to the entire function body. For field-level locking, use `ReentrantLock` or
 *    `@Synchronized` on a function that wraps the critical section.
 *    Here we use an explicit `synchronized(this)` lambda to match the original exactly.
 *
 * 3. `?: throw` / `?:` — Elvis used for guard expressions.
 *
 * 4. String template — `"Token refresh failed: $e"` replaces concatenation.
 *
 * 5. `apply { }` on CircuitBreakerConfig.custom() — configures the builder inline
 *    without a temporary variable. Each `.X()` call returns the builder, which Kotlin can
 *    chain; `apply` is used when the builder is not fluent (doesn't return `this`).
 *    Here the builder IS fluent, so chaining is fine without `apply`.
 */
@Configuration
open class AppConfig {

    @Bean
    open fun tokenSupplier(): Supplier<String> {
        // Anonymous object — replaces Java anonymous class `new Supplier<String>() { ... }`
        return object : Supplier<String> {
            private var token: String? = null
            private var expiryTime: Long = 0L

            override fun get(): String {
                // `synchronized(this)` takes a lambda in Kotlin — replaces synchronized block
                synchronized(this) {
                    if (token == null || System.currentTimeMillis() >= expiryTime) {
                        try {
                            token = fetchNewToken()
                            expiryTime = System.currentTimeMillis() + 3_600_000L
                        } catch (e: Exception) {
                            LogUtils.getLogger(AppConfig::class.java.name)
                                .error("Failed to refresh token", e)
                            throw RuntimeException("Token refresh failed", e)
                        }
                    }
                    // `!!` = non-null assertion: tells Kotlin "I know this is non-null here"
                    // Use only when you're certain; throws NullPointerException if wrong
                    return token!!
                }
            }

            private fun fetchNewToken(): String = "new-jwt-token"
        }
    }

    @Bean
    open fun resourceBundle(): ResourceBundle = ResourceBundle.getBundle("labels", Locale.getDefault())

    @Bean
    open fun meterRegistry(): MeterRegistry = SimpleMeterRegistry()

    @Bean
    open fun circuitBreakerRegistry(): CircuitBreakerRegistry {
        val config = CircuitBreakerConfig.custom()
            .failureRateThreshold(50f)
            .waitDurationInOpenState(Duration.ofSeconds(10))
            .permittedNumberOfCallsInHalfOpenState(3)
            .slidingWindowSize(10)
            .recordExceptions(StatusRuntimeException::class.java)  // `::class.java` = Java Class<T>
            .build()
        return CircuitBreakerRegistry.of(config)
    }

    @Bean
    open fun dashboardViewModel(dataFetchService: DataFetchService): DashboardViewModel =
        DashboardViewModel(dataFetchService)
}
