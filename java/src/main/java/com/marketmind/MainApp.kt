package com.marketmind

import com.marketmind.client.BackendClient
import com.marketmind.ml.InferenceJNI
import com.marketmind.services.DataFetchService
import com.marketmind.services.UserAuthService
import com.marketmind.utils.Config
import com.marketmind.utils.PythonRunner
import com.marketmind.utils.SceneManager
import io.grpc.netty.GrpcSslContexts
import javafx.application.Application
import javafx.application.Platform
import javafx.fxml.FXMLLoader
import javafx.scene.Node
import javafx.scene.Scene
import javafx.scene.control.Alert
import javafx.scene.layout.BorderPane
import javafx.stage.Stage
import org.slf4j.LoggerFactory
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.boot.SpringApplication
import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.context.ConfigurableApplicationContext
import java.io.File
import java.io.IOException
import java.io.InputStream
import java.util.Properties

/**
 * JAVA → KOTLIN changes:
 *
 * 1. `@SpringBootApplication` on a Kotlin class — works identically. `open` required because
 *    Spring Boot creates a CGLIB proxy subclass. Without the kotlin-spring plugin, you must
 *    add `open` manually to @SpringBootApplication classes and their @Bean methods.
 *
 * 2. Two constructors — Kotlin supports multiple constructors. The no-arg constructor
 *    (required by JavaFX) uses `constructor()` syntax with a body.
 *    The `@Autowired` constructor uses the primary constructor syntax.
 *    We use `secondary constructor` (the no-arg one) with `this(...)` delegation pattern.
 *    Here the no-arg constructor is kept as a secondary constructor since the primary
 *    has @Autowired parameters.
 *
 * 3. `companion object` — holds the static instance reference and logger.
 *    `@JvmStatic` on `getInstance()` and `main()` lets Java code call them as static methods.
 *    Without `@JvmStatic`, Java would call `MainApp.Companion.getInstance()`.
 *
 * 4. `setOnCloseRequest { event -> ... }` — lambda replaces anonymous EventHandler class.
 *
 * 5. Kotlin `Thread.setDefaultUncaughtExceptionHandler { thread, throwable -> ... }` —
 *    SAM conversion; lambda passed to a method expecting UncaughtExceptionHandler.
 *
 * 6. `is?.use { }` pattern for InputStream — Kotlin's `use` replaces try-with-resources.
 *    The `?.` handles the case where `getResourceAsStream` returns null.
 *
 * 7. `scene.stylesheets` — property access (was `scene.getStylesheets()`).
 *    `scene.stylesheets += url` replaces `scene.getStylesheets().add(url)`.
 */
@SpringBootApplication(scanBasePackages = ["com.marketmind", "com.marketmind.client"])
open class MainApp : Application {

    companion object {
        private val logger = LoggerFactory.getLogger(MainApp::class.java)
        private const val CONFIG_FILE = "/config/application.properties"
        private const val VIEWS_FILE = "/config/views.properties"

        @JvmStatic  // Makes this callable as MainApp.getInstance() from Java
        var instance: MainApp? = null
            private set

        @JvmStatic
        fun main(args: Array<String>) = launch(MainApp::class.java, *args)
    }

    private var springContext: ConfigurableApplicationContext? = null
    private lateinit var primaryStage: Stage
    private var rootLayout: BorderPane? = null
    private val viewCache = mutableMapOf<String, Node>()
    private val viewPaths = mutableMapOf<String, String>()

    // These are late-initialized because the no-arg constructor fetches them from Spring
    private lateinit var dataFetchService: DataFetchService
    private lateinit var userAuthService: UserAuthService
    private lateinit var backendClient: BackendClient
    private lateinit var inferenceJNI: InferenceJNI
    private lateinit var pythonRunner: PythonRunner
    private lateinit var config: Config
    private lateinit var sceneManager: SceneManager

    private var currentTheme = "light"

    // No-arg constructor required by JavaFX
    constructor() : super() {
        if (springContext == null) {
            springContext = SpringApplication.run(MainApp::class.java)
        }
        val ctx = springContext!!
        config          = ctx.getBean(Config::class.java)
        sceneManager    = ctx.getBean(SceneManager::class.java)
        dataFetchService = ctx.getBean(DataFetchService::class.java)
        userAuthService = ctx.getBean(UserAuthService::class.java)
        backendClient   = ctx.getBean(BackendClient::class.java)
        inferenceJNI    = ctx.getBean(InferenceJNI::class.java)
        pythonRunner    = ctx.getBean(PythonRunner::class.java)
    }

    // @Autowired constructor — Spring uses this when creating the bean
    @Autowired
    constructor(
        springContext: ConfigurableApplicationContext,
        dataFetchService: DataFetchService,
        userAuthService: UserAuthService,
        backendClient: BackendClient,
        inferenceJNI: InferenceJNI,
        pythonRunner: PythonRunner,
        config: Config,
        sceneManager: SceneManager
    ) : super() {
        this.springContext    = springContext
        this.dataFetchService = dataFetchService
        this.userAuthService  = userAuthService
        this.backendClient    = backendClient
        this.inferenceJNI     = inferenceJNI
        this.pythonRunner     = pythonRunner
        this.config           = config
        this.sceneManager     = sceneManager
    }

    override fun init() {
        // Spring context initialized via constructor
    }

    override fun start(stage: Stage) {
        instance = this
        Application.setUserAgentStylesheet(atlantafx.base.theme.PrimerLight().userAgentStylesheet)
        primaryStage = stage
        setupExceptionHandler()
        logger.info("Starting JavaFX application with Spring")
        sceneManager.setupStage(stage)
        stage.title = "MarketMind"  // Property assignment replaces setTitle()

        // Lambda replaces EventHandler anonymous class
        stage.setOnCloseRequest {
            try { stop() } catch (e: Exception) { logger.error("Error during shutdown", e) }
            Platform.exit()
        }
        stage.show()
    }

    private fun setupExceptionHandler() {
        // SAM conversion: lambda passed as UncaughtExceptionHandler
        Thread.setDefaultUncaughtExceptionHandler { _, throwable ->
            logger.error("Uncaught exception", throwable)
            Platform.runLater {
                Alert(Alert.AlertType.ERROR, "An unexpected error occurred.").showAndWait()
            }
        }
    }

    private fun loadConfig() {
        // `?.use { }` — safe call on nullable InputStream + auto-close
        javaClass.getResourceAsStream(CONFIG_FILE)?.use { stream ->
            val props = Properties()
            props.load(stream)
            config.setProperties(props)
        } ?: logger.error("Failed to load configuration")
    }

    private fun loadViewPaths() {
        try {
            javaClass.getResourceAsStream(VIEWS_FILE)?.use { stream ->
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

    private fun loadSslContext() = try {
        val certPath = config.getProperty("grpc.cert.path", null)
        val keyPath  = config.getProperty("grpc.key.path", null)
        GrpcSslContexts.forClient()
            .trustManager(File(certPath!!))
            .keyManager(File(certPath), File(keyPath!!))
            .build()
    } catch (e: Exception) {
        logger.error("Failed to load SSL context", e)
        null
    }

    private fun initRootLayout() {
        try {
            val loader = FXMLLoader(javaClass.getResource("/fxml/RootLayout.fxml"))
            rootLayout = loader.load()
            sceneManager.setRootLayout(rootLayout!!)
            val scene = Scene(rootLayout)
            val cssUrl = javaClass.getResource("/css/$currentTheme.css")?.toExternalForm()
            if (cssUrl != null) scene.stylesheets += cssUrl  // `+=` replaces .add()
            primaryStage.scene = scene
            primaryStage.title = "MarketMind"
        } catch (e: IOException) {
            logger.error("Failed to load root layout", e)
            showErrorAlert("Startup Error", "Could not start application.")
        }
    }

    fun showDashboard() = sceneManager.showView("dashboard")
    fun showLogin()     = sceneManager.showView("login")
    fun showSettings()  = sceneManager.showView("settings")

    private fun showErrorAlert(title: String, message: String) {
        Platform.runLater {
            Alert(Alert.AlertType.ERROR, message).also { it.title = title }.showAndWait()
        }
    }

    fun getVersion(): String = config.getProperty("app.version", "1.0.0") ?: "1.0.0"

    fun setTheme(theme: String) {
        currentTheme = theme
        val scene = primaryStage.scene
        scene.stylesheets.clear()
        scene.stylesheets += javaClass.getResource("/css/$theme.css")?.toExternalForm() ?: return
    }

    override fun stop() {
        logger.info("Shutting down MarketMind application")
        springContext?.close()   // `?.` safe call — does nothing if already null
        springContext = null
    }
}
