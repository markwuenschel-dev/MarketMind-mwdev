package com.marketmind.features.realtime

import com.marketmind.websocket.MarketDataWebSocketClient
import javafx.application.Platform
import javafx.fxml.FXML
import javafx.scene.chart.LineChart
import javafx.scene.chart.XYChart
import javafx.scene.control.Label
import org.slf4j.LoggerFactory
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.stereotype.Component
import java.net.URI

/**
 * JAVA → KOTLIN changes:
 *
 * 1. `lateinit var webSocketClient` — the WebSocket client is initialized in `initialize()`,
 *    not at construction, so `lateinit var` is appropriate.
 *    The original Java left it as null (untyped); Kotlin requires us to declare it nullable
 *    (`MarketDataWebSocketClient?`) or use `lateinit var`.
 *    `lateinit var` is used here since we check `::webSocketClient.isInitialized` in shutdown().
 *
 * 2. Listener lambda — `webSocketClient.priceProperty().addListener { _, _, newPrice -> ... }`.
 *    The three lambda parameters correspond to `(observable, oldValue, newValue)`.
 *    We use `_` for parameters we don't need — idiomatic Kotlin.
 *
 * 3. `::webSocketClient.isInitialized` — Kotlin property reference; used to check if a
 *    `lateinit var` has been assigned before accessing it in `shutdown()`.
 *
 * 4. `logger.error(...)` replaces `System.err.println(...)` — logging best practice.
 */
@Component
class RealTimeController {

    companion object {
        private val logger = LoggerFactory.getLogger(RealTimeController::class.java)
    }

    @FXML private lateinit var realTimeLabel: Label
    @FXML private lateinit var tickerLabel: Label
    @FXML private lateinit var priceLabel: Label
    @FXML private lateinit var priceChart: LineChart<String, Number>

    private lateinit var webSocketClient: MarketDataWebSocketClient
    private val priceSeries = XYChart.Series<String, Number>()

    @Autowired
    private lateinit var viewModel: RealTimeViewModel

    @FXML
    fun initialize() {
        priceSeries.name = "Price"
        priceChart.data.add(priceSeries)

        try {
            webSocketClient = MarketDataWebSocketClient(URI("ws://localhost:8080/market-data"))
            webSocketClient.connect()

            tickerLabel.textProperty().bind(webSocketClient.tickerProperty())
            priceLabel.textProperty().bind(webSocketClient.priceProperty().asString("%.2f"))

            // Lambda with `_` for unused parameters — `(obs, oldPrice, newPrice)` → `(_, _, newPrice)`
            webSocketClient.priceProperty().addListener { _, _, newPrice ->
                Platform.runLater {
                    val timestamp = System.currentTimeMillis().toString()
                    priceSeries.data.add(XYChart.Data(timestamp, newPrice.toDouble()))
                    if (priceSeries.data.size > 10) priceSeries.data.removeAt(0)
                }
            }
        } catch (e: Exception) {
            logger.error("Error connecting to WebSocket: {}", e.message)
        }
    }

    fun shutdown() {
        // `::webSocketClient.isInitialized` — checks if lateinit var has been set
        if (::webSocketClient.isInitialized) {
            webSocketClient.close()
        }
    }
}
