package com.marketmind.utils

import org.slf4j.Logger
import org.slf4j.LoggerFactory
import org.slf4j.MDC
import java.nio.file.Files
import java.nio.file.Path
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ConcurrentHashMap

/**
 * JAVA → KOTLIN changes:
 *
 * 1. `object` — replaces `final class` with private constructor.
 *
 * 2. Static initializer block `static { ... }` → `init { }` inside the object.
 *    In a Kotlin `object`, `init` runs when the object is first accessed (same semantics).
 *
 * 3. `arrayOf(...)` — replaces `new String[]{ ... }` for array literals.
 *
 * 4. `any { }` on collections — replaces manual for-loop with boolean accumulator.
 *    `SENSITIVE_KEYS.any { it.equals(key, ignoreCase = true) }` reads like English.
 *    Named parameter `ignoreCase = true` replaces `String.equalsIgnoreCase()`.
 *
 * 5. String template in `buildString { }` — replaces StringBuilder append chains.
 *    `buildString` is a Kotlin stdlib function that builds a String via a StringBuilder DSL.
 *    `appendLine("text")` = append + newline; cleaner than multiple append() calls.
 *
 * 6. `for (elem in array)` — Kotlin for-in; no `.iterator()` needed.
 *
 * 7. `ignored` in catch clause — Kotlin requires naming the exception variable;
 *    use `_` or a descriptive name if you're ignoring it.
 */
object LogUtils {

    private val loggerCache = ConcurrentHashMap<String, Logger>()
    private const val LOG_DIR = "logs"

    private val SENSITIVE_KEYS = arrayOf(
        "password", "secret", "api_key", "token", "auth", "credentials"
    )

    // `init` in an object = Java's static initializer block
    init {
        try {
            Files.createDirectories(Path.of(LOG_DIR))
        } catch (_: Exception) {
            // Ignored — same as original
        }
    }

    fun getLogger(name: String): Logger =
        loggerCache.computeIfAbsent(name, LoggerFactory::getLogger)

    fun bindContext(key: String, value: String) {
        MDC.put(key, if (isSensitive(key)) "***REDACTED***" else value)
    }

    fun unbindContext(key: String) = MDC.remove(key)

    fun redactSensitive(input: MutableMap<String, Any>): MutableMap<String, Any> {
        for (key in SENSITIVE_KEYS) {
            if (input.containsKey(key)) input[key] = "***REDACTED***"
        }
        return input
    }

    // `any { }` — returns true if at least one element matches the predicate
    // `ignoreCase = true` is a named argument — clearer than positional booleans
    private fun isSensitive(key: String): Boolean =
        SENSITIVE_KEYS.any { it.equals(key, ignoreCase = true) }

    fun formatException(t: Throwable): String =
        // `buildString { }` — Kotlin DSL for building strings via StringBuilder
        buildString {
            append(SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS").format(Date()))
            append(" ERROR: ").append(t.toString()).append("\n")
            for (elem in t.stackTrace) {
                append("\tat ").append(elem.toString()).append("\n")
            }
        }
}
