package com.marketmind.features.dashboard

import com.marketmind.models.DashboardSummary
import org.springframework.stereotype.Service

/**
 * JAVA → KOTLIN changes:
 *
 * 1. `DashboardSummary(sp500 = 4500.0, ...)` — named arguments make construction readable.
 *    Replaces Java's mutable object + setters pattern since DashboardSummary is now a data class.
 *    No need to create the object and then call six setters.
 *
 * 2. `buildList { }` — Kotlin stdlib for building an immutable list inline.
 *    `repeat(10) { i -> ... }` replaces `for (int i = 0; i < 10; i++)`.
 *    Inside `buildList`, `add(...)` appends to the list being built.
 *
 * 3. `doubleArrayOf(...)` — Kotlin's array literal for `double[]`.
 *    Replaces `new double[]{ i, 4500 + i * 10 }`.
 */
@Service
class DashboardService {

    fun fetchDashboardSummary(): DashboardSummary {
        // Named arguments + data class: replace the create-then-set-fields pattern
        val history = buildList {
            // `repeat(n) { i -> }` replaces for (int i = 0; i < n; i++)
            repeat(10) { i ->
                add(doubleArrayOf(i.toDouble(), 4500.0 + i * 10))  // doubleArrayOf = new double[]{}
            }
        }
        return DashboardSummary(
            sp500 = 4500.0,
            dowJones = 35000.0,
            nasdaq = 15000.0,
            portfolioValue = 100000.0,
            notificationCount = 5,
            sp500History = history
        )
    }
}
