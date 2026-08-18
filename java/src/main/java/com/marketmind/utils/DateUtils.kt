package com.marketmind.utils

import java.time.*
import java.time.format.DateTimeFormatter
import java.time.temporal.ChronoUnit

/**
 * JAVA → KOTLIN changes:
 *
 * 1. `object` declaration — replaces a Java `final class` with a private constructor and
 *    all-static methods. In Kotlin, `object` creates a true singleton. You call its members
 *    the same way: `DateUtils.parseIsoDate(...)`.
 *
 * 2. No `private constructor()` needed — object declarations cannot be instantiated at all.
 *
 * 3. Constants (`val`) are declared directly in the object body.
 *    `const val` = compile-time constant (like Java `static final` for primitives/Strings).
 *    Regular `val` = runtime constant (for complex types like DateTimeFormatter).
 *
 * 4. Single-expression functions — `fun foo(): T = expr` replaces `{ return expr }`.
 *    Used throughout since most functions are one-liners.
 *
 * 5. No `@JvmStatic` needed here because all members are already on the object (singleton),
 *    but you'd add `@JvmStatic` inside a `companion object` for Java call-site compatibility.
 */
object DateUtils {

    /** ISO-8601 date formatter (e.g., 2024-06-01) */
    val ISO_DATE: DateTimeFormatter = DateTimeFormatter.ISO_LOCAL_DATE

    /** ISO-8601 datetime formatter (e.g., 2024-06-01T14:52:00) */
    val ISO_DATE_TIME: DateTimeFormatter = DateTimeFormatter.ISO_LOCAL_DATE_TIME

    /** User-friendly display date (e.g., Jun 1, 2024) */
    val DISPLAY_DATE: DateTimeFormatter = DateTimeFormatter.ofPattern("MMM d, yyyy")

    /** Display datetime with time (e.g., Jun 1, 2024 14:52) */
    val DISPLAY_DATE_TIME: DateTimeFormatter = DateTimeFormatter.ofPattern("MMM d, yyyy HH:mm")

    // Single-expression functions — `=` instead of `{ return ... }`
    fun parseIsoDate(date: String): LocalDate = LocalDate.parse(date, ISO_DATE)

    fun parseIsoDateTime(dateTime: String): LocalDateTime =
        LocalDateTime.parse(dateTime, ISO_DATE_TIME)

    fun formatIsoDate(date: LocalDate): String = date.format(ISO_DATE)

    fun formatIsoDateTime(dateTime: LocalDateTime): String = dateTime.format(ISO_DATE_TIME)

    fun formatDisplayDate(date: LocalDate): String = date.format(DISPLAY_DATE)

    fun formatDisplayDateTime(dateTime: LocalDateTime): String = dateTime.format(DISPLAY_DATE_TIME)

    fun toEpochMillis(dateTime: LocalDateTime): Long =
        dateTime.toInstant(ZoneOffset.UTC).toEpochMilli()

    fun fromEpochMillis(epochMillis: Long): LocalDateTime =
        Instant.ofEpochMilli(epochMillis).atZone(ZoneOffset.UTC).toLocalDateTime()

    fun daysBetween(start: LocalDate, end: LocalDate): Long =
        ChronoUnit.DAYS.between(start, end)

    fun minutesBetween(start: LocalDateTime, end: LocalDateTime): Long =
        ChronoUnit.MINUTES.between(start, end)

    fun toZonedDateTime(localDateTime: LocalDateTime, zoneId: ZoneId): ZonedDateTime =
        localDateTime.atZone(zoneId)

    fun formatZonedDateTime(zonedDateTime: ZonedDateTime, formatter: DateTimeFormatter): String =
        zonedDateTime.format(formatter)

    fun nowInZone(zoneId: ZoneId): ZonedDateTime = ZonedDateTime.now(zoneId)
}
