package com.marketmind.features.dashboard

import com.marketmind.services.DataFetchService
import javafx.beans.property.*
import javafx.scene.chart.XYChart
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.stereotype.Component

/**
 * JAVA → KOTLIN changes:
 *
 * 1. Constructor injection with `@Autowired` — same pattern as other Spring beans.
 *    The Java version had `@Autowired` on the constructor; Kotlin keeps it explicit here
 *    since `AppConfig` also creates a `DashboardViewModel` bean (the annotation helps
 *    Spring disambiguate).
 *
 * 2. Property initializers — `val statusMessage = SimpleStringProperty(...)` declared at
 *    the class body level (not in a constructor body). Kotlin initializes them in declaration order.
 *
 * 3. JavaFX property pattern — the Java code exposes three methods per field:
 *    `fooProperty()`, `getFoo()`, `setFoo()`. In Kotlin we keep this pattern for JavaFX
 *    compatibility, but notice `getFoo()` / `setFoo()` could be replaced with a Kotlin
 *    computed property:
 *       `var sp500: Double get() = _sp500.get() set(v) { _sp500.set(v) }`
 *    We keep the explicit function style here since JavaFX FXML binding expects `fooProperty()`.
 */
@Component
class DashboardViewModel @Autowired constructor(
    private val dataFetchService: DataFetchService
) {
    private val statusMessage = SimpleStringProperty("Welcome to MarketMind Dashboard")
    private val sp500 = SimpleDoubleProperty()
    private val dowJones = SimpleDoubleProperty()
    private val nasdaq = SimpleDoubleProperty()
    private val portfolioValue = SimpleDoubleProperty()
    private val notificationCount = SimpleIntegerProperty()
    private val sp500Series = XYChart.Series<String, Number>()

    fun statusMessageProperty(): StringProperty = statusMessage
    fun getStatusMessage(): String = statusMessage.get()
    fun setStatusMessage(message: String) = statusMessage.set(message)

    fun sp500Property(): DoubleProperty = sp500
    fun getSp500(): Double = sp500.get()
    fun setSp500(value: Double) = sp500.set(value)

    fun dowJonesProperty(): DoubleProperty = dowJones
    fun getDowJones(): Double = dowJones.get()
    fun setDowJones(value: Double) = dowJones.set(value)

    fun nasdaqProperty(): DoubleProperty = nasdaq
    fun getNasdaq(): Double = nasdaq.get()
    fun setNasdaq(value: Double) = nasdaq.set(value)

    fun portfolioValueProperty(): DoubleProperty = portfolioValue
    fun getPortfolioValue(): Double = portfolioValue.get()
    fun setPortfolioValue(value: Double) = portfolioValue.set(value)

    fun notificationCountProperty(): IntegerProperty = notificationCount
    fun getNotificationCount(): Int = notificationCount.get()
    fun setNotificationCount(count: Int) = notificationCount.set(count)

    fun getSp500Series(): XYChart.Series<String, Number> = sp500Series
}
