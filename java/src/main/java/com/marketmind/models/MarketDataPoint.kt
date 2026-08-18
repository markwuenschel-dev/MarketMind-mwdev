package com.marketmind.models

import java.time.Instant

/**
 * JAVA → KOTLIN changes:
 *
 * 1. `data class` — Kotlin generates equals(), hashCode(), toString(), copy() automatically.
 *    We override equals/hashCode here to preserve the original's timestamp-only comparison
 *    (the Java version only compared by timestamp, not all fields).
 *
 * 2. Constructor parameters become properties directly — no need to declare fields and then
 *    assign them in the constructor body. `val` = immutable (Java's `private final`).
 *
 * 3. `init {}` block — runs after the primary constructor. Replaces the constructor body.
 *
 * 4. `require(condition) { "message" }` — Kotlin stdlib function that throws
 *    IllegalArgumentException if false. Replaces the verbose Java if/throw pattern.
 *
 * 5. No getters needed — `point.timestamp` works directly; Kotlin generates
 *    Java-compatible getters (getTimestamp(), etc.) for interop automatically.
 *
 * 6. `compareTo` uses single-expression function syntax (`=` instead of `{ return ... }`).
 *
 * 7. `other !is MarketDataPoint` — Kotlin's `is` / `!is` replace Java's `instanceof`.
 *    After an `is` check, Kotlin smart-casts `other` — no explicit cast needed.
 */
data class MarketDataPoint(
    val timestamp: Instant,
    val open: Double,
    val high: Double,
    val low: Double,
    val close: Double,
    val volume: Long
) : Comparable<MarketDataPoint> {

    init {
        // require() throws IllegalArgumentException with the given message if false
        require(open > 0 && high > 0 && low > 0 && close > 0) { "Prices must be positive" }
        require(volume >= 0) { "Volume cannot be negative" }
    }

    // Single-expression function: body is just an expression after `=`
    override fun compareTo(other: MarketDataPoint): Int =
        timestamp.compareTo(other.timestamp)

    // Override because data class default equals() compares ALL fields,
    // but the original Java only compared by timestamp.
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is MarketDataPoint) return false  // smart-cast: `other` is now MarketDataPoint below
        return timestamp == other.timestamp
    }

    override fun hashCode(): Int = timestamp.hashCode()
}
