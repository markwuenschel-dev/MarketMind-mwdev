package com.marketmind.websocket

import com.fasterxml.jackson.databind.ObjectMapper
import javafx.application.Platform
import javafx.beans.property.DoubleProperty
import javafx.beans.property.SimpleDoubleProperty
import javafx.beans.property.SimpleStringProperty
import javafx.beans.property.StringProperty
import org.java_websocket.client.WebSocketClient
import org.java_websocket.handshake.ServerHandshake
import org.slf4j.LoggerFactory
import java.net.URI

/**
 * JAVA → KOTLIN changes:
 *
 * 1. `class ... (serverUri: URI) : WebSocketClient(serverUri)` — primary constructor
 *    delegates to the parent constructor with `: WebSocketClient(serverUri)`.
 *    Java equivalent: `public MarketDataWebSocketClient(URI uri) { super(uri); }`.
 *
 * 2. `override fun` — mandatory `override` keyword for overriding methods.
 *
 * 3. `try { ... } catch (e: Exception)` — all exceptions are unchecked in Kotlin;
 *    no need to declare `throws Exception` on function signatures.
 *
 * 4. `marketData.ticker` / `marketData.price` — property access if the deserialized type
 *    is a data class. Here we reference `MarketData` (the data class from the handler file).
 *    If it were a plain Java class with getters, it would work the same way (Kotlin
 *    exposes Java getters as properties).
 *
 * 5. Logger replaces `System.out.println` / `System.err.println`.
 */
class MarketDataWebSocketClient(serverUri: URI) : WebSocketClient(serverUri) {

    companion object {
        private val logger = LoggerFactory.getLogger(MarketDataWebSocketClient::class.java)
    }

    private val objectMapper = ObjectMapper()
    private val _tickerProperty = SimpleStringProperty()
    private val _priceProperty = SimpleDoubleProperty()

    override fun onOpen(handshakedata: ServerHandshake) {
        logger.info("WebSocket client connected to server")
    }

    override fun onMessage(message: String) {
        try {
            val marketData = objectMapper.readValue(message, MarketData::class.java)
            Platform.runLater {
                _tickerProperty.set(marketData.ticker)
                _priceProperty.set(marketData.price)
            }
        } catch (e: Exception) {
            logger.error("Error processing WebSocket message: {}", e.message)
        }
    }

    override fun onClose(code: Int, reason: String, remote: Boolean) {
        logger.info("WebSocket client closed: {}", reason)
    }

    override fun onError(ex: Exception) {
        logger.error("WebSocket client error: {}", ex.message)
    }

    fun tickerProperty(): StringProperty = _tickerProperty
    fun priceProperty(): DoubleProperty = _priceProperty
}
