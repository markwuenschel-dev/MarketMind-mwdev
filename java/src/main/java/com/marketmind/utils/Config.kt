package com.marketmind.utils

import jakarta.annotation.PostConstruct
import org.springframework.beans.factory.annotation.Value
import org.springframework.stereotype.Component
import java.util.Properties

/**
 * JAVA → KOTLIN changes:
 *
 * 1. Spring `@Value` on constructor parameters — Kotlin idiomatic Spring style injects
 *    directly into the primary constructor. No `@Autowired` needed (Spring infers it
 *    when there's a single constructor).
 *    In Java: field-level injection. In Kotlin: constructor-level is preferred.
 *
 * 2. `var` with a backing `@Value` — `@Value("\${app.name}") val appName: String` reads
 *    the property at startup. Note: `$` must be escaped as `\$` inside Kotlin string literals
 *    because `$` starts a string template.
 *
 * 3. `open` keyword — Spring requires `open` (non-final) classes and methods to create
 *    CGLIB proxies. Java classes are open by default; Kotlin classes are `final` by default.
 *    The `kotlin-spring` plugin auto-applies `open` to @Component/@Service etc., but we
 *    add it explicitly here on the overridable getters for clarity.
 *
 * 4. `checkNotNull` / `check()` — Kotlin stdlib for postcondition checks.
 *    `check(condition) { "message" }` throws IllegalStateException if false.
 *
 * 5. `Properties` field is `private val` — immutable reference, mutable contents.
 */
@Component
open class Config(
    @Value("\${app.name}") val appName: String,
    @Value("\${app.version}") val appVersion: String,
    @Value("\${api.ib.endpoint}") val ibEndpoint: String,
    @Value("\${api.fred.key}") val fredApiKey: String
) {
    private val properties = Properties()

    // `var` with custom getter/setter — Kotlin property; Java interop generates getTheme()/setTheme()
    var theme: String = "Light"
    var refreshRate: Double = 10.0
    var notificationsEnabled: Boolean = true

    // `@PostConstruct` works the same as Java
    @PostConstruct
    open fun validate() {
        // `check()` throws IllegalStateException — the right exception for invalid state
        check(ibEndpoint.isNotEmpty()) { "Interactive Brokers API endpoint is missing" }
        check(fredApiKey.isNotEmpty()) { "FRED API key is missing" }
    }

    open fun save() { /* Persist to file or database */ }

    fun setProperties(props: Properties) {
        properties.clear()
        properties.putAll(props)
    }

    // `?: defaultValue` is the Elvis operator — returns the right side if left is null
    fun getProperty(key: String, defaultValue: String?): String? =
        properties.getProperty(key, defaultValue)
}
