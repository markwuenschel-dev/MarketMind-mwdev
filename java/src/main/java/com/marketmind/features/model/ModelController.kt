package com.marketmind.features.model

import javafx.fxml.FXML
import javafx.scene.control.Label
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.stereotype.Component

/**
 * JAVA → KOTLIN: Minimal changes.
 * `@Autowired lateinit var viewModel` — field injection for the ViewModel.
 * `@FXML lateinit var modelLabel` — FXML-injected field.
 */
@Component
class ModelController {

    @FXML private lateinit var modelLabel: Label

    @Autowired
    private lateinit var viewModel: ModelViewModel

    @FXML
    fun initialize() {
        modelLabel.textProperty().bind(viewModel.modelDataProperty())
    }
}
