package com.marketmind.features.realtime

import javafx.beans.property.SimpleStringProperty
import javafx.beans.property.StringProperty
import org.springframework.stereotype.Component

@Component
class RealTimeViewModel {
    private val realTimeData = SimpleStringProperty()

    fun realTimeDataProperty(): StringProperty = realTimeData
    fun updateRealTimeData(data: String) = realTimeData.set(data)
}
