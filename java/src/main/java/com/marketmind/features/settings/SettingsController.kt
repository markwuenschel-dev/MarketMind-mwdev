package com.marketmind.features.settings

import com.marketmind.MainApp
import com.marketmind.utils.Config
import com.marketmind.utils.LogUtils
import javafx.fxml.FXML
import javafx.scene.control.*
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.stereotype.Component

/**
 * JAVA → KOTLIN changes:
 *
 * 1. `@Autowired lateinit var config: Config` — field injection for Config bean.
 *
 * 2. `themeComboBox.items.addAll(...)` — Kotlin property access (`items` vs `getItems()`).
 *    JavaFX generates Java getters; Kotlin exposes them as properties automatically.
 *    `getItems()` → `.items`, `getValue()` → `.value`, `isSelected()` → `.isSelected`.
 *
 * 3. `config.theme` — Kotlin property access; the Java `getTheme()` call becomes `config.theme`.
 *    Similarly, `config.theme = selectedTheme` replaces `config.setTheme(selectedTheme)`.
 *    This works because Kotlin automatically exposes Java getters/setters as properties.
 *
 * 4. `with(alert) { ... }` scope function — configures an object without repeating its name.
 *    Replaces `alert.setTitle(...); alert.setHeaderText(...); alert.setContentText(...)`.
 *    `with` is used when you don't need to return the receiver (unlike `apply` which does).
 */
@Component
class SettingsController {

    companion object {
        private val logger = LogUtils.getLogger(SettingsController::class.java.name)
    }

    private var mainApp: MainApp? = null

    @Autowired
    private lateinit var config: Config

    @FXML private lateinit var themeComboBox: ComboBox<String>
    @FXML private lateinit var refreshRateSlider: Slider
    @FXML private lateinit var notificationsCheckBox: CheckBox
    @FXML private lateinit var saveButton: Button

    @FXML
    private fun initialize() {
        logger.info("Initializing SettingsController")
        themeComboBox.items.addAll("Light", "Dark", "Solarized")
        // Kotlin property access — `config.theme` calls getTheme() under the hood
        themeComboBox.value = config.theme
        refreshRateSlider.value = config.refreshRate
        notificationsCheckBox.isSelected = config.notificationsEnabled
    }

    fun setMainApp(app: MainApp) { mainApp = app }

    @FXML
    private fun handleSave() {
        val selectedTheme = themeComboBox.value
        val refreshRate = refreshRateSlider.value
        val notificationsEnabled = notificationsCheckBox.isSelected

        LogUtils.bindContext("theme", selectedTheme)
        try {
            // Property assignment — `config.theme = x` calls setTheme(x) under the hood
            config.theme = selectedTheme
            config.refreshRate = refreshRate
            config.notificationsEnabled = notificationsEnabled
            config.save()
            logger.info("Settings saved: theme={}, refreshRate={}, notifications={}",
                selectedTheme, refreshRate, notificationsEnabled)
            showAlert("Settings saved successfully.")
        } catch (e: Exception) {
            logger.error("Failed to save settings", e)
            showAlert("Failed to save settings. Please try again.")
        } finally {
            LogUtils.unbindContext("theme")
        }
    }

    private fun showAlert(message: String) {
        // `with(obj) { }` — runs block with obj as `this`; no need to prefix each call with `alert.`
        with(Alert(Alert.AlertType.INFORMATION)) {
            title = "Settings"
            headerText = null
            contentText = message
            showAndWait()
        }
    }
}
