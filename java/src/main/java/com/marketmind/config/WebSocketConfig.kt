package com.marketmind.config

import com.marketmind.websocket.MarketDataWebSocketHandler
import jakarta.annotation.PostConstruct
import org.springframework.context.annotation.Configuration
import org.springframework.web.socket.config.annotation.EnableWebSocket
import org.springframework.web.socket.config.annotation.WebSocketConfigurer
import org.springframework.web.socket.config.annotation.WebSocketHandlerRegistry

/**
 * JAVA → KOTLIN changes:
 *
 * 1. `class ... (private val handler: MarketDataWebSocketHandler) : WebSocketConfigurer`
 *    — Primary constructor with constructor injection. No `@Autowired` needed (single constructor).
 *    The `: WebSocketConfigurer` after the class declaration implements the interface.
 *    Compare with Java: `class X implements Y { @Autowired X(Y y) { this.y = y; } }`.
 *
 * 2. `override fun` — mandatory `override` keyword (same concept, enforced at compile time).
 *
 * 3. `@PostConstruct` — works identically to Java.
 *
 * Note: `open class` would be needed if Spring creates a CGLIB proxy. Since this is a
 * @Configuration with no proxied @Bean methods, `open` may not be strictly required,
 * but it's safer to add it for Spring compatibility without the kotlin-spring plugin.
 */
@Configuration
@EnableWebSocket
open class WebSocketConfig(
    private val marketDataHandler: MarketDataWebSocketHandler
) : WebSocketConfigurer {

    override fun registerWebSocketHandlers(registry: WebSocketHandlerRegistry) {
        registry.addHandler(marketDataHandler, "/market-data").setAllowedOrigins("*")
    }

    @PostConstruct
    fun startMarketDataSimulation() {
        marketDataHandler.simulateMarketDataUpdate()
    }
}
