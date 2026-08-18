package com.marketmind.config

import com.github.benmanes.caffeine.cache.Caffeine
import org.springframework.cache.CacheManager
import org.springframework.cache.annotation.EnableCaching
import org.springframework.cache.caffeine.CaffeineCacheManager
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import java.util.concurrent.TimeUnit

/**
 * JAVA → KOTLIN changes:
 *
 * Minimal changes here — Spring @Configuration classes map 1:1.
 *
 * 1. `open` on the class and `@Bean` functions — required for Spring CGLIB proxy generation.
 *    Kotlin classes are final by default; Spring needs to subclass them to create proxies.
 *    The `kotlin-spring` Gradle/Maven plugin adds `open` automatically to @Configuration
 *    classes. Without the plugin, you must add it manually.
 *
 * 2. Method reference `::caffeineCacheBuilder` — Kotlin method references use `::`.
 *    Note: `setCaffeine` expects a `Caffeine<Any, Any>`, which we return from the private function.
 *
 * 3. Function return type inferred — `private fun caffeineCacheBuilder()` returns
 *    `Caffeine<Any, Any>` inferred from the expression. Kotlin infers return types for
 *    single-expression functions; you can always write it explicitly for clarity.
 */
@Configuration
@EnableCaching
open class CacheConfig {

    @Bean
    open fun cacheManager(): CacheManager =
        CaffeineCacheManager("marketData").apply {
            // `apply { }` — configures the object and returns it; replaces multiple setter calls
            setCaffeine(caffeineCacheBuilder())
        }

    private fun caffeineCacheBuilder(): Caffeine<Any, Any> =
        Caffeine.newBuilder()
            .expireAfterWrite(1, TimeUnit.MINUTES)
            .maximumSize(1000)
            .recordStats()
}
