package com.marketmind.utils

import java.text.DecimalFormat
import java.text.NumberFormat
import java.util.*

/**
 * JAVA → KOTLIN changes:
 *
 * 1. `object` declaration — same as DateUtils, replaces a utility class with all-static methods.
 *
 * 2. ThreadLocal with `by lazy` alternative — the Java code uses ThreadLocal for thread-safe
 *    formatter instances. Kotlin keeps `ThreadLocal` the same way; the lambda syntax is cleaner.
 *
 * 3. `when` expression — replaces Java `if/else if/else` chains.
 *    Unlike Java's switch, `when` is an expression (returns a value), works with any type,
 *    and uses `->` instead of `:` + `break`.
 *
 * 4. `Math.abs(value)` → `kotlin.math.abs(value)` — Kotlin has its own math package.
 *    Both work, but the Kotlin import is idiomatic.
 *
 * 5. `also { }` scope function — runs a block on the receiver and returns the receiver.
 *    Used here to configure NumberFormat after creation without a temp variable.
 *    Java equivalent: create, configure, return.
 */
object NumberFormatter {

    private val DEFAULT_LOCALE = Locale.US

    // ThreadLocal lambdas — same thread-safety concept as Java, cleaner syntax
    private val INTEGER_FORMAT = ThreadLocal.withInitial<NumberFormat> {
        NumberFormat.getIntegerInstance(DEFAULT_LOCALE)
    }
    private val DECIMAL_FORMAT = ThreadLocal.withInitial<DecimalFormat> {
        DecimalFormat("#,##0.00")
    }
    private val CURRENCY_FORMAT = ThreadLocal.withInitial<NumberFormat> {
        // `also { }` — configure the object and return it; replaces local variable pattern
        NumberFormat.getCurrencyInstance(DEFAULT_LOCALE).also {
            it.minimumFractionDigits = 2
            it.maximumFractionDigits = 2
        }
    }
    private val PERCENT_FORMAT = ThreadLocal.withInitial<NumberFormat> {
        NumberFormat.getPercentInstance(DEFAULT_LOCALE).also {
            it.minimumFractionDigits = 2
            it.maximumFractionDigits = 2
        }
    }

    fun formatInt(value: Long): String = INTEGER_FORMAT.get().format(value)

    fun formatDecimal(value: Double): String = DECIMAL_FORMAT.get().format(value)

    fun formatCurrency(value: Double): String = CURRENCY_FORMAT.get().format(value)

    fun formatPercent(value: Double): String = PERCENT_FORMAT.get().format(value)

    fun formatCurrency(value: Double, currencyCode: String): String =
        NumberFormat.getCurrencyInstance(DEFAULT_LOCALE).also {
            it.currency = Currency.getInstance(currencyCode)
            it.minimumFractionDigits = 2
            it.maximumFractionDigits = 2
        }.format(value)

    fun formatAbbreviated(value: Double): String {
        val abs = Math.abs(value)
        // `when` expression — returns a value, replaces if/else if/else chain
        // Each branch is `condition -> result`; no fall-through, no break needed
        return when {
            abs >= 1_000_000_000 -> formatDecimal(value / 1_000_000_000) + "B"
            abs >= 1_000_000     -> formatDecimal(value / 1_000_000) + "M"
            abs >= 1_000         -> formatDecimal(value / 1_000) + "K"
            else                 -> formatDecimal(value)
        }
    }
}
