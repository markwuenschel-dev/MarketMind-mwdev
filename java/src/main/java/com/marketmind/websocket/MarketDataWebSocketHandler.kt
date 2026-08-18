package com.marketmind.websocket

import com.fasterxml.jackson.databind.ObjectMapper
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Component
import org.springframework.web.socket.CloseStatus
import org.springframework.web.socket.TextMessage
import org.springframework.web.socket.WebSocketSession
import org.springframework.web.socket.handler.TextWebSocketHandler
import java.util.concurrent.ConcurrentHashMap

/**
 * NOTE: The `MarketData` inner class was defined inside the Java file.
 * In Kotlin it's declared at the top level of this file (before the handler class).
 * Top-level declarations are common in Kotlin — no need to nest unrelated classes.
 *
 * JAVA → KOTLIN changes:
 *
 * 1. `data class MarketData` — replaces the simple Java POJO with getters/setters.
 *    `var` fields because the original had setters (mutable).
 *
 * 2. `sessions[session.id]` — Kotlin map access uses `[]`; `.id` = property (was `getId()`).
 *
 * 3. Logger replaces `System.out.println` / `System.err.println`.
 *
 * 4. `Thread { }.start()` — SAM conversion; lambda passed as Runnable.
 *    `also { it.start() }` could be used for a more functional style.
 *
 * 5. `session.isOpen` — property access (was `session.isOpen()`).
 *    Note: in Java `isOpen()` is a method; Kotlin exposes it as property `isOpen`.
 */

// Top-level class — no need to nest it in the handler file in Kotlin
data class MarketData(
    var ticker: String,
    var price: Double,
    var timestamp: Long
)

@Component
class MarketDataWebSocketHandler : TextWebSocketHandler() {

    companion object {
        private val logger = LoggerFactory.getLogger(MarketDataWebSocketHandler::class.java)
    }

    private val sessions = ConcurrentHashMap<String, WebSocketSession>()
    private val objectMapper = ObjectMapper()

    override fun afterConnectionEstablished(session: WebSocketSession) {
        sessions[session.id] = session  // `session.id` = property access (was getId())
        logger.info("WebSocket connection established: {}", session.id)
        session.sendMessage(TextMessage("Connected to MarketMind WebSocket"))
    }

    override fun afterConnectionClosed(session: WebSocketSession, status: CloseStatus) {
        sessions.remove(session.id)
        logger.info("WebSocket connection closed: {}", session.id)
    }

    fun broadcastMarketData(marketData: MarketData) {
        try {
            val message = TextMessage(objectMapper.writeValueAsString(marketData))
            // `values` property (was `values()` method on Map); `filter` for open sessions
            sessions.values
                .filter { it.isOpen }
                .forEach { it.sendMessage(message) }
        } catch (e: Exception) {
            logger.error("Error broadcasting market data: {}", e.message)
        }
    }

    fun simulateMarketDataUpdate() {
        // Thread with SAM lambda — replaces `new Thread(() -> { ... }).start()`
        Thread {
            try {
                while (true) {
                    broadcastMarketData(
                        MarketData("AAPL", 150.00 + Math.random() * 10, System.currentTimeMillis())
                    )
                    Thread.sleep(5000)
                }
            } catch (e: InterruptedException) {
                logger.warn("Market data simulation interrupted: {}", e.message)
            }
        }.start()
    }
}
