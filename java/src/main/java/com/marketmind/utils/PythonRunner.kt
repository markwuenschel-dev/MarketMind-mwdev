package com.marketmind.utils

import org.slf4j.LoggerFactory
import org.springframework.stereotype.Component
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader
import java.util.concurrent.CompletableFuture
import java.util.concurrent.TimeUnit

/**
 * JAVA → KOTLIN changes:
 *
 * 1. Java `record` → Kotlin `data class` — Java 16+ records map directly to Kotlin data classes.
 *    `ProcessResult` was a record in Java; it becomes a data class here.
 *    Both give you equals, hashCode, toString, and component functions automatically.
 *
 * 2. `listOf(...)` — Kotlin read-only list literal; replaces `List.of(...)`.
 *
 * 3. Thread lambda — `Thread { ... }` replaces `new Thread(() -> { ... })`.
 *    Kotlin's SAM conversion means any single-method Java interface accepts a lambda.
 *
 * 4. `use { }` scope function — Kotlin's replacement for try-with-resources.
 *    Any `Closeable` can use `.use { reader -> ... }`. The resource is closed automatically.
 *
 * 5. `trimEnd()` — Kotlin String extension; replaces `.trim()` (trims both ends in Java,
 *    but we only want trailing whitespace removed from stdout/stderr here). We use `trim()`
 *    to match the original behavior exactly.
 *
 * 6. `buildList { }` — Kotlin stdlib function for building a list; replaces ArrayList+add pattern.
 *    Or we can just use `mutableListOf` + `add` which is more direct here.
 *
 * 7. `StringBuilder` — same as Java; `.append()` works identically.
 *
 * 8. `companion object` — holds the logger (class-level, not instance-level).
 */
@Component
class PythonRunner {

    private val pythonCommand: String

    companion object {
        private val logger = LoggerFactory.getLogger(PythonRunner::class.java)
        private const val TIMEOUT_SECONDS = 60L
    }

    init {
        pythonCommand = detectPythonCommand()
        logger.info("Using Python command: {}", pythonCommand)
    }

    private fun detectPythonCommand(): String {
        for (cmd in listOf("python3", "python")) {
            try {
                val p = ProcessBuilder(cmd, "--version")
                    .redirectErrorStream(true)
                    .start()
                if (p.waitFor(5, TimeUnit.SECONDS) && p.exitValue() == 0) return cmd
            } catch (_: Exception) {
                // try next
            }
        }
        logger.warn("No Python found, defaulting to 'python'")
        return "python"
    }

    fun runModule(module: String, args: List<String>): ProcessResult {
        val command = mutableListOf(pythonCommand, "-m", module) + args
        logger.info("Running: {} -m {} {}", pythonCommand, module, args.joinToString(" "))

        return try {
            val pb = ProcessBuilder(command).apply {
                // `apply { }` — configures the receiver and returns it; replaces pb.X(); pb.Y()
                environment()["PYTHONUNBUFFERED"] = "1"
            }
            val process = pb.start()

            val stdout = StringBuilder()
            val stderr = StringBuilder()

            // `Thread { }` — SAM conversion; Kotlin lambda passed as Runnable
            val stdoutThread = Thread {
                // `.use { }` — replaces try-with-resources; closes the reader automatically
                BufferedReader(InputStreamReader(process.inputStream)).use { reader ->
                    reader.lineSequence().forEach { stdout.append(it).append("\n") }
                }
            }
            val stderrThread = Thread {
                BufferedReader(InputStreamReader(process.errorStream)).use { reader ->
                    reader.lineSequence().forEach { stderr.append(it).append("\n") }
                }
            }

            stdoutThread.start()
            stderrThread.start()

            if (!process.waitFor(TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
                process.destroyForcibly()
                return ProcessResult("", "Process timed out", -1)
            }

            stdoutThread.join(1000)
            stderrThread.join(1000)

            val exitCode = process.exitValue()
            if (stderr.isNotEmpty()) {
                logger.warn("Python stderr: {}", stderr.toString().trim())
            }

            ProcessResult(stdout.toString().trim(), stderr.toString().trim(), exitCode)

        } catch (e: Exception) {
            logger.error("Failed to execute Python module", e)
            ProcessResult("", e.message ?: "unknown error", -1)
        }
    }

    fun runModuleAsync(module: String, args: List<String>): CompletableFuture<ProcessResult> =
        CompletableFuture.supplyAsync { runModule(module, args) }

    fun executeScript(scriptPath: String): String = runModule(scriptPath, emptyList()).stdout

    /**
     * Java `record` → Kotlin `data class`.
     * `stdout`, `stderr`, `exitCode` are `val` (immutable, like record components).
     * Method `success()` becomes a regular function in the data class body.
     */
    data class ProcessResult(val stdout: String, val stderr: String, val exitCode: Int) {
        fun success(): Boolean = exitCode == 0 || exitCode == 1
    }
}
