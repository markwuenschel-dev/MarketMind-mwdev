package com.marketmind.models

/**
 * JAVA → KOTLIN changes:
 *
 * 1. `data class` with default values — replaces the Java POJO with 6 fields, 6 getters,
 *    6 setters, plus any constructors. The entire original file collapses to ~10 lines.
 *
 * 2. Default parameter values — `var sp500: Double = 0.0` means you can construct with
 *    `DashboardSummary()` or `DashboardSummary(sp500 = 4500.0)`.
 *    Named arguments (`sp500 = 4500.0`) make call sites self-documenting.
 *
 * 3. `var` fields — mutable like the original Java setters. If these were never mutated
 *    after construction you'd use `val` instead.
 *
 * 4. `List<DoubleArray>` — idiomatic Kotlin prefers `List` (read-only view) over `MutableList`
 *    at API boundaries; `emptyList()` is the null-safe default.
 */
data class DashboardSummary(
    var sp500: Double = 0.0,
    var dowJones: Double = 0.0,
    var nasdaq: Double = 0.0,
    var portfolioValue: Double = 0.0,
    var notificationCount: Int = 0,
    var sp500History: List<DoubleArray> = emptyList()  // Each element is [timestamp, value]
)
