package com.marketmind.services

import com.marketmind.models.MarketData
import com.marketmind.models.MarketDataPoint
import com.marketmind.utils.LogUtils
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.cache.annotation.Cacheable
import org.springframework.stereotype.Service
import java.time.Instant
import java.util.SortedSet

/**
 * JAVA → KOTLIN changes:
 *
 * 1. `@Service` class with primary constructor — Kotlin's idiomatic Spring style.
 *    No `@Autowired` annotation on the constructor is needed when there's only one constructor;
 *    Spring infers it. Adding `@Autowired` explicitly is fine but redundant here.
 *
 * 2. `companion object` — holds the logger. `MarketData::class.java` is how Kotlin refers
 *    to the Java Class object. `::class` gives KClass; `.java` converts it to Class<T>.
 *
 * 3. `require(ticker.isNotBlank())` — Kotlin extension on String; replaces the null-or-empty check.
 *    Note: in Kotlin, String parameters can't be null unless declared `String?`, so we only
 *    need the blank check (the null check is handled by the type system).
 *
 * 4. No `throws IOException` declarations — Kotlin has no checked exceptions.
 *    All exceptions are unchecked; you never declare them on function signatures.
 */
@Service
class DataFetchService @Autowired constructor(
    private val marketData: MarketData
) {
    companion object {
        private val logger = LogUtils.getLogger(DataFetchService::class.java.name)
    }

    @Cacheable(value = ["marketData"], key = "#ticker")
    fun getMarketData(ticker: String): SortedSet<MarketDataPoint> {
        require(ticker.isNotBlank()) { "Ticker cannot be null or empty" }

        LogUtils.bindContext("ticker", ticker)
        return try {
            logger.info("Fetching market data for ticker {}", ticker)
            val point = fetchFromGrpc(ticker)
            marketData.addData(ticker, point)
            val dataPoints = marketData.getData(ticker)
            logger.debug("Retrieved {} data points for ticker {}", dataPoints.size, ticker)
            dataPoints
        } finally {
            LogUtils.unbindContext("ticker")
        }
    }

    private fun fetchFromGrpc(ticker: String): MarketDataPoint {
        val price = 150.00 + Math.random() * 10
        val volume = (1000 + Math.random() * 1000).toLong()
        return MarketDataPoint(Instant.now(), price, price, price, price, volume)
    }
}
