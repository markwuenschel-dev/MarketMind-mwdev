package com.marketmind.services

import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper
import com.marketmind.utils.PythonRunner
import org.slf4j.LoggerFactory
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.stereotype.Service
import java.util.concurrent.CompletableFuture

/**
 * JAVA → KOTLIN changes:
 *
 * 1. Java `record` → Kotlin `data class` (BacktestResult).
 *    Java records are immutable and generate equals/hashCode/toString. Kotlin data classes
 *    do the same. The only difference: Kotlin data class fields can be `var` (mutable),
 *    while record components are always immutable. We use `val` here to match the record.
 *
 * 2. Factory method `failure()` → `companion object` function.
 *    Java records can have static factory methods. In Kotlin, `companion object` holds
 *    class-level functions. `@JvmStatic` would make it callable from Java as BacktestResult.failure().
 *
 * 3. `?.` safe call and `?:` Elvis — for nullable JSON node access.
 *    `root.get("success")?.asBoolean() ?: false` means: if get() returns null, use false.
 *    Contrast with Java where you'd get a NullPointerException without explicit null checks.
 *
 * 4. `lines()` — Kotlin extension on String; returns a Sequence<String> without splitting.
 *    Replaces `output.split("\n")`.
 *
 * 5. `firstOrNull { predicate }` — returns the first matching element or null;
 *    replaces a manual for-loop with a return inside.
 *
 * 6. `companion object { private val mapper = ObjectMapper() }` — holds the shared
 *    ObjectMapper at the class level (equivalent to Java's `private static final`).
 */
@Service
class BacktestService @Autowired constructor(
    private val pythonRunner: PythonRunner
) {
    companion object {
        private val logger = LoggerFactory.getLogger(BacktestService::class.java)
        private val mapper = ObjectMapper()
    }

    /**
     * Java record → Kotlin data class.
     * `val` fields = immutable. Factory method lives in `companion object`.
     */
    data class BacktestResult(
        val success: Boolean,
        val error: String?,
        val bundlePath: String?,
        val validationStatus: String?,
        val totalReturn: Double,
        val sharpeRatio: Double,
        val maxDrawdown: Double,
        val winRate: Double,
        val numTrades: Int
    ) {
        companion object {
            // Factory method — replaces Java's static method on the record
            fun failure(error: String) = BacktestResult(
                success = false, error = error,
                bundlePath = null, validationStatus = null,
                totalReturn = 0.0, sharpeRatio = 0.0,
                maxDrawdown = 0.0, winRate = 0.0, numTrades = 0
            )
        }
    }

    fun runBacktest(dataFile: String, fastSma: Int, slowSma: Int): BacktestResult {
        return try {
            val result = pythonRunner.runModule(
                "py.bridge.run_pipeline",
                listOf(dataFile, "--fast-sma", fastSma.toString(), "--slow-sma", slowSma.toString())
            )

            if (!result.success()) {
                logger.error("Python process failed: {}", result.stderr)
                return BacktestResult.failure("Python process failed: ${result.stderr}")
            }

            val jsonStr = extractJson(result.stdout)
                ?: run {
                    // `run { }` — executes a block and returns its result; used here for multi-line else
                    logger.error("No JSON found in output: {}", result.stdout)
                    return BacktestResult.failure("No JSON in output")
                }

            val root: JsonNode = mapper.readTree(jsonStr)

            // `?.asBoolean() ?: false` — safe call + Elvis: null-safe JSON access
            if (root.get("success")?.asBoolean() != true) {
                return BacktestResult.failure(root.get("error")?.asText() ?: "unknown error")
            }

            val backtest = root.get("backtest")
            BacktestResult(
                success = true,
                error = null,
                bundlePath = root.get("bundle_path")?.asText(),
                validationStatus = root.get("validation")?.get("status")?.asText(),
                totalReturn = backtest?.get("total_return")?.asDouble() ?: 0.0,
                sharpeRatio = backtest?.get("sharpe_ratio")?.asDouble() ?: 0.0,
                maxDrawdown = backtest?.get("max_drawdown")?.asDouble() ?: 0.0,
                winRate = backtest?.get("win_rate")?.asDouble() ?: 0.0,
                numTrades = backtest?.get("num_trades")?.asInt() ?: 0
            )

        } catch (e: Exception) {
            logger.error("Failed to parse backtest result", e)
            BacktestResult.failure(e.message ?: "unknown error")
        }
    }

    private fun extractJson(output: String): String? =
        // `lines()` Kotlin extension; `firstOrNull { }` replaces manual for-loop
        output.lines()
            .map { it.trim() }
            .firstOrNull { it.startsWith("{") && it.endsWith("}") }

    fun runBacktestAsync(dataFile: String, fastSma: Int, slowSma: Int): CompletableFuture<BacktestResult> =
        CompletableFuture.supplyAsync { runBacktest(dataFile, fastSma, slowSma) }
}
