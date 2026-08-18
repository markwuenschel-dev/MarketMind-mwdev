package com.marketmind.features

import com.marketmind.MainApp
import com.marketmind.services.DataFetchService
import com.marketmind.utils.LogUtils
import com.marketmind.utils.SceneManager
import javafx.fxml.FXML
import javafx.fxml.FXMLLoader
import javafx.scene.layout.BorderPane
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.context.ApplicationContext
import org.springframework.stereotype.Component

/**
 * JAVA → KOTLIN changes:
 *
 * 1. `@Autowired lateinit var` — Spring field injection. Each field is declared non-null
 *    (Kotlin guarantees non-null after Spring injects them).
 *
 * 2. `@FXML lateinit var rootPane` — FXML-injected BorderPane.
 *
 * 3. `var sceneManager: SceneManager? = null` — nullable because it's set via a setter method,
 *    not injected. Alternatively use `lateinit var` if always set before use.
 *
 * 4. Function bodies as single expressions where possible:
 *    `@FXML private fun showDashboard() = loadView("/fxml/Dashboard.fxml")`
 */
@Component
class RootLayoutController {

    companion object {
        private val logger = LogUtils.getLogger(RootLayoutController::class.java.name)
    }

    private var mainApp: MainApp? = null
    private var sceneManager: SceneManager? = null

    @Autowired private lateinit var dataFetchService: DataFetchService
    @Autowired private lateinit var applicationContext: ApplicationContext

    @FXML private lateinit var rootPane: BorderPane

    @FXML
    private fun initialize() {
        logger.info("Initializing RootLayoutController")
        loadView("/fxml/Dashboard.fxml")
    }

    fun setMainApp(app: MainApp) { mainApp = app }
    fun setSceneManager(sm: SceneManager) { sceneManager = sm }

    fun loadView(fxmlPath: String) {
        try {
            LogUtils.bindContext("fxml", fxmlPath)
            logger.debug("Loading view: {}", fxmlPath)
            val loader = FXMLLoader(javaClass.getResource(fxmlPath)).apply {
                setControllerFactory(applicationContext::getBean)
            }
            rootPane.center = loader.load()
        } catch (e: Exception) {
            logger.error("Failed to load view: {}", fxmlPath, e)
            throw RuntimeException("Failed to load view: $fxmlPath", e)
        } finally {
            LogUtils.unbindContext("fxml")
        }
    }

    // Single-expression FXML handlers
    @FXML private fun showDashboard() = loadView("/fxml/Dashboard.fxml")
    @FXML private fun showLogin()     = loadView("/fxml/Login.fxml")
    @FXML private fun showSettings()  = loadView("/fxml/Settings.fxml")
    @FXML private fun showRealTime()  = loadView("/fxml/RealTime.fxml")
    @FXML private fun showModel()     = loadView("/fxml/Model.fxml")
}
