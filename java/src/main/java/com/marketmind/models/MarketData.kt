package com.marketmind.models

import com.fasterxml.jackson.core.type.TypeReference
import com.fasterxml.jackson.databind.ObjectMapper
import com.marketmind.utils.LogUtils
import io.micrometer.core.instrument.Counter
import io.micrometer.core.instrument.MeterRegistry
import io.micrometer.core.instrument.Timer
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.stereotype.Component
import java.io.File
import java.time.Instant
import java.util.*
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ConcurrentSkipListSet
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

/**
 * JAVA → KOTLIN changes:
 *
 * 1. `companion object` — Kotlin has no static members. Class-level declarations go in a
 *    `companion object`, which is a singleton object attached to the class.
 *    `companion object { val logger = ... }` replaces `private static final Logger logger = ...`
 *
 * 2. Primary constructor with `@Autowired` — `class MarketData @Autowired constructor(...)`.
 *    Fields declared in the class body use `private val` / `private var` directly.
 *
 * 3. `lateinit var` — for non-null properties that can't be initialized at construction time
 *    (e.g., assigned in `initMetrics()`). Accessing before init throws an exception.
 *
 * 4. `init {}` block — runs after the primary constructor; used to call initMetrics().
 *
 * 5. `require(symbol.isNotBlank())` — Kotlin extension function on String; cleaner than
 *    `symbol == null || symbol.trim().isEmpty()`.
 *
 * 6. String templates — `"Duplicate timestamp for symbol $symbol"` replaces concatenation.
 *
 * 7. `for ((symbol, set) in data)` — destructuring declaration in a for-loop.
 *    Replaces `for (Map.Entry<...> entry : data.entrySet())` + entry.getKey/getValue().
 *
 * 8. `data.mapValues { (_, v) -> v.toList() }` — functional transformation of a map.
 *    The `_` discards the key we don't need.
 *
 * 9. `set.lastOrNull()` — Kotlin stdlib extension; replaces isEmpty() + last() pattern.
 *    Returns null instead of throwing if empty (null-safe).
 *
 * 10. `fun interface MarketDataListener` — Kotlin SAM (Single Abstract Method) interface.
 *     Allows callers to pass a lambda: `data.addListener { sym, pt -> ... }`.
 *
 * 11. `compareBy(MarketDataPoint::timestamp)` — Kotlin stdlib comparator builder.
 *     Replaces `Comparator.comparing(MarketDataPoint::getTimestamp)`.
 *     `::timestamp` is a property reference (Kotlin properties have no separate getter reference).
 */
@Component
class MarketData @Autowired constructor(
    private val meterRegistry: MeterRegistry
) {
    private val data = ConcurrentHashMap<String, ConcurrentSkipListSet<MarketDataPoint>>()
    private val listeners = CopyOnWriteArrayList<MarketDataListener>()
    private val symbolCount = AtomicInteger(0)
    private val totalDataPoints = AtomicLong(0)

    private lateinit var dataPointsAdded: Counter
    private lateinit var addDataTimer: Timer
    private lateinit var getDataTimer: Timer

    companion object {
        private val logger = LogUtils.getLogger(MarketData::class.java.name)
    }

    init {
        initMetrics()
    }

    private fun initMetrics() {
        dataPointsAdded = meterRegistry.counter("marketdata.datapoints.added")
        addDataTimer = meterRegistry.timer("marketdata.adddata.time")
        getDataTimer = meterRegistry.timer("marketdata.getdata.time")
        meterRegistry.gauge("marketdata.symbols.count", symbolCount, AtomicInteger::get)
        meterRegistry.gauge("marketdata.datapoints.total", totalDataPoints, AtomicLong::get)
    }

    fun addListener(listener: MarketDataListener) { listeners.add(listener) }
    fun removeListener(listener: MarketDataListener) { listeners.remove(listener) }

    fun addData(symbol: String, point: MarketDataPoint) {
        require(symbol.isNotBlank()) { "Symbol cannot be null or empty" }
        val sample = Timer.start(meterRegistry)
        try {
            val key = symbol.trim()
            val set = data.computeIfAbsent(key) {
                // Lambda passed to computeIfAbsent — replaces anonymous inner class
                symbolCount.incrementAndGet()
                ConcurrentSkipListSet(compareBy(MarketDataPoint::timestamp))
            }
            if (!set.add(point)) {
                logger.warn("Attempted to add duplicate timestamp for symbol {}", symbol)
                throw IllegalArgumentException("Duplicate timestamp for symbol $symbol")
            }
            totalDataPoints.incrementAndGet()
            dataPointsAdded.increment()
            listeners.forEach { it.onDataAdded(symbol, point) }  // forEach with lambda
            logger.debug("Successfully added data point for symbol {}", symbol)
        } finally {
            sample.stop(addDataTimer)
        }
    }

    fun getData(symbol: String): SortedSet<MarketDataPoint> {
        require(symbol.isNotBlank()) { "Symbol cannot be null or empty" }
        val sample = Timer.start(meterRegistry)
        return try {
            // `data[key]` is idiomatic Kotlin for map.get(key) — returns null if absent
            val set = data[symbol.trim()]
            if (set == null) {
                logger.debug("No data found for symbol {}", symbol)
                Collections.emptySortedSet()
            } else {
                Collections.unmodifiableSortedSet(set)
            }
        } finally {
            sample.stop(getDataTimer)
        }
    }

    // Return type `MarketDataPoint?` — the `?` means nullable (may return null)
    // `lastOrNull()` replaces the Java isEmpty() + last() two-step pattern
    fun getLatestDataPoint(symbol: String): MarketDataPoint? =
        getData(symbol).lastOrNull()

    fun getDataSince(symbol: String, since: Instant): SortedSet<MarketDataPoint> {
        val set = getData(symbol)
        if (set.isEmpty()) return Collections.emptySortedSet()
        val from = MarketDataPoint(since, 0.001, 0.001, 0.001, 0.001, 0)
        return Collections.unmodifiableSortedSet(TreeSet(set.tailSet(from)))
    }

    fun pruneOldData(threshold: Instant) {
        // Destructuring declaration: `(symbol, set)` unpacks Map.Entry key and value
        for ((symbol, set) in data) {
            val to = MarketDataPoint(threshold, 0.001, 0.001, 0.001, 0.001, 0)
            val toRemove = set.headSet(to)
            val removed = toRemove.size.toLong()
            set.removeAll(toRemove)
            totalDataPoints.addAndGet(-removed)
            logger.info("Pruned {} old data points for symbol {}", removed, symbol)
        }
    }

    fun saveToFile(filePath: String) {
        val mapper = ObjectMapper()
        // mapValues transforms each value; `(_, v)` destructures the entry, `_` = unused key
        val serializableData = data.mapValues { (_, v) -> v.toList() }
        mapper.writeValue(File(filePath), serializableData)
        logger.info("Saved market data to {}", filePath)
    }

    fun loadFromFile(filePath: String) {
        val mapper = ObjectMapper()
        val loadedData: Map<String, List<MarketDataPoint>> = mapper.readValue(
            File(filePath),
            object : TypeReference<Map<String, List<MarketDataPoint>>>() {}
        )
        data.clear()
        symbolCount.set(0)
        totalDataPoints.set(0)
        for ((symbol, points) in loadedData) {
            val set = ConcurrentSkipListSet(compareBy(MarketDataPoint::timestamp))
            set.addAll(points)
            data[symbol] = set
            totalDataPoints.addAndGet(points.size.toLong())
        }
        symbolCount.set(data.size)
    }

    // `fun interface` = SAM interface — a single-method interface that accepts lambdas
    fun interface MarketDataListener {
        fun onDataAdded(symbol: String, point: MarketDataPoint)
    }
}
