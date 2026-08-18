package com.marketmind.features.dashboard

import com.marketmind.services.BacktestService
import com.marketmind.services.BacktestService.BacktestResult
import com.marketmind.utils.SceneManager
import javafx.application.Platform
import javafx.fxml.FXML
import javafx.scene.chart.LineChart
import javafx.scene.chart.XYChart
import javafx.scene.control.Button
import javafx.scene.control.Label
import javafx.scene.control.TextField
import javafx.scene.layout.VBox
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Component

/**
 * JAVA → KOTLIN changes:
 *
 * 1. `@FXML` fields use `lateinit var` — because JavaFX injects them after construction
 *    (they're null at construction, non-null after `initialize()`).
 *    In Java, FXML fields are simply null until injected; in Kotlin, we must declare them
 *    nullable (`Label?`) or use `lateinit var` to avoid null checks on every use.
 *    `lateinit var` is the idiomatic choice for FXML-injected fields.
 *
 * 2. `thenAccept { result -> Platform.runLater { ... } }` — Kotlin lambda in CompletableFuture.
 *    The `->` inside `{ result -> ... }` separates the parameter from the body.
 *    When there's only one parameter you can also use `it`: `thenAccept { Platform.runLater { ... } }`.
 *
 * 3. `exceptionally { e -> ... ; null }` — returns null to satisfy the `T?` return type.
 *    In Kotlin, the last expression in a lambda is the return value.
 *
 * 4. `setAll(...)` on JavaFX list — same API, called the same way.
 *
 * 5. `String.format(...)` — same as Java; Kotlin also has string templates for simple cases.
 *    `"%.2f%%".format(value)` is idiomatic Kotlin (String extension function).
 *
 * 6. `for (point in summary.sp500History)` — Kotlin for-in replaces Java's enhanced for.
 */
@Component
class DashboardController(
    private val dashboardViewModel: DashboardViewModel,
    private val dashboardService: DashboardService,
    private val backtestService: BacktestService,
    private val sceneManager: SceneManager
) {
    companion object {
        private val logger = LoggerFactory.getLogger(DashboardController::class.java)
    }

    // `lateinit var` — FXML-injected; non-null after JavaFX calls initialize()
    @FXML private lateinit var statusLabel: Label
    @FXML private lateinit var sp500Label: Label
    @FXML private lateinit var dowJonesLabel: Label
    @FXML private lateinit var nasdaqLabel: Label
    @FXML private lateinit var portfolioValueLabel: Label
    @FXML private lateinit var notificationCountLabel: Label
    @FXML private lateinit var sp500Chart: LineChart<String, Number>
    @FXML private lateinit var dataFileField: TextField
    @FXML private lateinit var fastSmaField: TextField
    @FXML private lateinit var slowSmaField: TextField
    @FXML private lateinit var runBacktestButton: Button
    @FXML private lateinit var resultsBox: VBox
    @FXML private lateinit var validationStatusLabel: Label
    @FXML private lateinit var totalReturnLabel: Label
    @FXML private lateinit var sharpeRatioLabel: Label
    @FXML private lateinit var maxDrawdownLabel: Label
    @FXML private lateinit var winRateLabel: Label
    @FXML private lateinit var numTradesLabel: Label

    @FXML
    private fun initialize() {
        statusLabel.textProperty().bind(dashboardViewModel.statusMessageProperty())
        sp500Label.textProperty().bind(dashboardViewModel.sp500Property().asString())
        dowJonesLabel.textProperty().bind(dashboardViewModel.dowJonesProperty().asString())
        nasdaqLabel.textProperty().bind(dashboardViewModel.nasdaqProperty().asString())
        portfolioValueLabel.textProperty().bind(dashboardViewModel.portfolioValueProperty().asString())
        notificationCountLabel.textProperty().bind(dashboardViewModel.notificationCountProperty().asString())
        sp500Chart.data.add(dashboardViewModel.getSp500Series())
        refreshDashboard()
    }

    @FXML
    private fun onRunBacktestClicked() {
        val dataFile = dataFileField.text.trim()
        val fastSma = fastSmaField.text.trim().toIntOrNull()
        val slowSma = slowSmaField.text.trim().toIntOrNull()

        // `toIntOrNull()` — returns null if parsing fails, replaces try/catch NumberFormatException
        if (fastSma == null || slowSma == null) {
            statusLabel.textProperty().unbind()
            statusLabel.text = "Error: Invalid SMA values"
            return
        }

        runBacktestButton.isDisable = true  // `isDisable = true` replaces setDisable(true)
        statusLabel.textProperty().unbind()
        statusLabel.text = "Running backtest..."

        backtestService.runBacktestAsync(dataFile, fastSma, slowSma)
            .thenAccept { result ->
                // Lambda parameter `result` replaces `(BacktestResult result) ->`
                Platform.runLater { displayBacktestResult(result) }
            }
            .exceptionally { e ->
                Platform.runLater {
                    statusLabel.text = "Error: ${e.message}"  // String template
                    runBacktestButton.isDisable = false
                }
                null  // Last expression = return value for exceptionally()
            }
    }

    private fun displayBacktestResult(result: BacktestResult) {
        runBacktestButton.isDisable = false
        if (!result.success) {
            statusLabel.text = "Backtest failed: ${result.error}"
            resultsBox.isVisible = false; resultsBox.isManaged = false
            return
        }

        statusLabel.text = "Backtest complete!"
        resultsBox.isVisible = true; resultsBox.isManaged = true

        validationStatusLabel.text = result.validationStatus
        validationStatusLabel.style = if ("PASS" == result.validationStatus)
            "-fx-text-fill: green; -fx-font-weight: bold;"
        else
            "-fx-text-fill: red; -fx-font-weight: bold;"

        // `"%.2f%%".format(value)` — Kotlin String extension, same as String.format("%.2f%%", value)
        totalReturnLabel.text = "%.2f%%".format(result.totalReturn * 100)
        sharpeRatioLabel.text = "%.2f".format(result.sharpeRatio)
        maxDrawdownLabel.text = "%.2f%%".format(result.maxDrawdown * 100)
        winRateLabel.text = "%.2f%%".format(result.winRate * 100)
        numTradesLabel.text = result.numTrades.toString()

        logger.info("Backtest result displayed: {}", result)
    }

    @FXML private fun onRefreshClicked() = refreshDashboard()

    private fun refreshDashboard() {
        val summary = dashboardService.fetchDashboardSummary()
        dashboardViewModel.setSp500(summary.sp500)
        dashboardViewModel.setDowJones(summary.dowJones)
        dashboardViewModel.setNasdaq(summary.nasdaq)
        dashboardViewModel.setPortfolioValue(summary.portfolioValue)
        dashboardViewModel.setNotificationCount(summary.notificationCount)

        // `map { }` transforms each element; replaces for-loop + chartData.add(...)
        val chartData = summary.sp500History.map { point ->
            XYChart.Data<String, Number>(point[0].toString(), point[1])
        }
        dashboardViewModel.getSp500Series().data.setAll(chartData)
        dashboardViewModel.setStatusMessage("Dashboard updated at ${System.currentTimeMillis()}")
    }

    @FXML private fun onSettingsClicked() = sceneManager.showView("settings")
    @FXML private fun onLoginClicked() = sceneManager.showView("login")
}
