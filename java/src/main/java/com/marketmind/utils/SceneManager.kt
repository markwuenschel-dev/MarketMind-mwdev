package com.marketmind.utils

import com.marketmind.features.RootLayoutController
import javafx.fxml.FXMLLoader
import javafx.scene.Node
import javafx.scene.Parent
import javafx.scene.Scene
import javafx.scene.layout.BorderPane
import javafx.stage.Stage
import org.slf4j.LoggerFactory
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.context.ConfigurableApplicationContext
import org.springframework.stereotype.Component
import java.io.IOException
import java.util.Properties

/**
 * JAVA → KOTLIN changes:
 *
 * 1. `@Autowired` field injection vs constructor injection — the original Java uses field
 *    injection (`@Autowired private ...`). In Kotlin, field injection requires `@Autowired`
 *    plus `lateinit var` (because the field can't be initialized at construction time,
 *    but we declare it as non-null). We keep field injection here to match the original.
 *    Constructor injection is preferred for testability, but this preserves the structure.
 *
 * 2. `lateinit var` — tells Kotlin "I promise this will be set before first access".
 *    Accessing it before assignment throws `UninitializedPropertyAccessException`.
 *    It can only be used with `var` (mutable) and non-nullable reference types.
 *
 * 3. `getOrDefault(key, null)` → `viewPaths[key]` — Kotlin map access returns null for
 *    missing keys, so `getOrDefault(key, null)` is just `map[key]`.
 *
 * 4. Null check: `?: throw` — Elvis operator: if the left side is null, throw/return the right.
 *    Replaces `if (x == null) throw ...`.
 *
 * 5. `var rootLayout: Parent? = null` — nullable type. The `?` means this can be null.
 *    We check `rootLayout == null` explicitly before use, and Kotlin's smart cast makes
 *    it non-null inside the `if (rootLayout != null)` branch.
 *
 * 6. `try { } catch (e: IOException)` — same structure as Java, no checked exceptions in Kotlin.
 *    All exceptions are unchecked in Kotlin (you never have to declare `throws IOException`).
 */
@Component
open class SceneManager {

    private var rootLayout: Parent? = null

    companion object {
        private val logger = LoggerFactory.getLogger(SceneManager::class.java)
    }

    @Autowired
    private lateinit var springContext: ConfigurableApplicationContext

    @Autowired
    private lateinit var config: Config

    private val viewCache = mutableMapOf<String, Node>()      // Kotlin mutableMapOf() = HashMap
    private val viewPaths = mutableMapOf<String, String>()

    init {
        loadViewPaths()
    }

    private fun loadViewPaths() {
        // `use { }` on InputStream replaces try-with-resources
        try {
            javaClass.getResourceAsStream("/config/views.properties")?.use { stream ->
                val props = Properties()
                props.load(stream)
                props.stringPropertyNames().forEach { key ->
                    viewPaths[key] = props.getProperty(key)
                }
            }
        } catch (e: IOException) {
            logger.error("Failed to load view paths", e)
            viewPaths["dashboard"] = "/fxml/Dashboard.fxml"
            viewPaths["login"]     = "/fxml/Login.fxml"
            viewPaths["settings"]  = "/fxml/Settings.fxml"
        }
    }

    fun showView(viewName: String) {
        // Elvis operator `?:` — if rootLayout is null, throw
        val layout = rootLayout ?: run {
            logger.error("Root layout not initialized; cannot show view: {}", viewName)
            throw IllegalStateException("Root layout not initialized")
        }
        try {
            // `viewPaths[viewName]` returns null if absent (no getOrDefault needed)
            val fxmlPath = viewPaths[viewName]
                ?: throw IllegalArgumentException("View not found: $viewName")  // Elvis throw

            val view = viewCache.getOrPut(viewName) {
                // `getOrPut` — if key absent, compute and store; replaces computeIfAbsent
                try {
                    val loader = FXMLLoader(javaClass.getResource(fxmlPath)).apply {
                        setControllerFactory(springContext::getBean)
                    }
                    loader.load<Node>()
                } catch (e: IOException) {
                    logger.error("Failed to load view: {}", viewName, e)
                    throw RuntimeException("Failed to load view: $viewName", e)
                }
            }
            (layout as BorderPane).center = view
        } catch (e: Exception) {
            logger.error("Error showing view: {}", viewName, e)
            throw e
        }
    }

    fun setRootLayout(borderPane: BorderPane) {
        rootLayout = borderPane
    }

    fun setupStage(primaryStage: Stage) {
        try {
            logger.info("Loading FXML: /fxml/RootLayout.fxml")
            val loader = FXMLLoader(javaClass.getResource("/fxml/RootLayout.fxml")).apply {
                setControllerFactory(springContext::getBean)
            }
            rootLayout = loader.load()

            val controller = loader.getController<RootLayoutController>()
            controller.setSceneManager(this)
            showView("dashboard")

            primaryStage.scene = Scene(rootLayout)  // `primaryStage.scene = ...` replaces setScene()
            primaryStage.show()
        } catch (e: IOException) {
            logger.error("Failed to load root layout", e)
            throw RuntimeException("Could not load root layout", e)
        }
    }

    fun getRootLayout(): Parent? = rootLayout
}
