package com.marketmind.features.model

import javafx.beans.property.SimpleStringProperty
import javafx.beans.property.StringProperty
import org.springframework.stereotype.Component

/**
 * JAVA → KOTLIN: Minimal changes.
 * Property declared as `private val` (immutable reference, mutable contents).
 * Single-expression functions for the getter and update method.
 */
@Component
class ModelViewModel {
    private val modelData = SimpleStringProperty()

    fun modelDataProperty(): StringProperty = modelData
    fun updateModelData(data: String) = modelData.set(data)
}
