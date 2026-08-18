package com.marketmind.features.login

import com.marketmind.MainApp
import com.marketmind.services.UserAuthService
import com.marketmind.utils.LogUtils
import javafx.fxml.FXML
import javafx.scene.control.Button
import javafx.scene.control.Label
import javafx.scene.control.PasswordField
import javafx.scene.control.TextField
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.stereotype.Component

/**
 * JAVA → KOTLIN changes:
 *
 * 1. `@Autowired` field injection with `lateinit var` — matches the Java field injection style.
 *    `userAuthService` is injected by Spring after construction.
 *
 * 2. `var mainApp: MainApp? = null` — nullable because setMainApp() sets it lazily.
 *    We use `?.` safe call when accessing it: `mainApp?.showDashboard()`.
 *
 * 3. `isNullOrBlank()` — Kotlin stdlib extension on String?; replaces
 *    `username == null || username.trim().isEmpty()`. Works on nullable String.
 *
 * 4. `?: return` pattern — the Elvis operator with early return:
 *    `val x = y ?: return` means "if y is null, return from the function".
 */
@Component
class LoginController {

    companion object {
        private val logger = LogUtils.getLogger(LoginController::class.java.name)
    }

    private var mainApp: MainApp? = null  // Nullable — set via setMainApp()

    @Autowired
    private lateinit var userAuthService: UserAuthService

    @FXML private lateinit var usernameField: TextField
    @FXML private lateinit var passwordField: PasswordField
    @FXML private lateinit var loginButton: Button
    @FXML private lateinit var errorLabel: Label

    @FXML
    private fun initialize() {
        logger.info("Initializing LoginController")
        errorLabel.isVisible = false
    }

    fun setMainApp(app: MainApp) { mainApp = app }

    @FXML
    private fun handleLogin() {
        val username = usernameField.text
        val password = passwordField.text

        // `isNullOrBlank()` — works on String?; handles null AND blank in one check
        if (username.isNullOrBlank() || password.isNullOrEmpty()) {
            showError("Please enter both username and password.")
            return
        }

        LogUtils.bindContext("username", username)
        try {
            if (userAuthService.authenticate(username, password)) {
                logger.info("User {} logged in successfully", username)
                mainApp?.showDashboard()  // `?.` safe call — does nothing if mainApp is null
            } else {
                showError("Invalid username or password.")
            }
        } catch (e: Exception) {
            logger.error("Authentication failed for user {}", username, e)
            showError("An error occurred during login. Please try again.")
        } finally {
            LogUtils.unbindContext("username")
        }
    }

    private fun showError(message: String) {
        errorLabel.text = message
        errorLabel.isVisible = true
    }
}
