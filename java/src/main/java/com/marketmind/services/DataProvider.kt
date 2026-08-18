package com.marketmind.services

/**
 * JAVA → KOTLIN changes:
 *
 * Interfaces in Kotlin are identical in concept to Java interfaces.
 * Differences:
 * - No `public` needed — interface members are public by default.
 * - No `abstract` needed — interface methods are abstract by default.
 * - Kotlin interfaces CAN have default implementations (like Java 8+) using a regular body.
 * - No semicolons.
 */
interface DataProvider {
    fun fetchData(query: String): String
}
