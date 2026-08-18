package com.marketmind.hashing.oi15;

import org.junit.jupiter.api.Test;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class Adr007ParityTest {
    @Test
    void replaysAllAdr007FixturesAcrossTheJavaLane() throws Exception {
        Adr007ParityRunner.Report report = Adr007ParityRunner.run(Path.of("tests", "golden", "adr007"));
        assertTrue(report.success());
        assertEquals(8, report.suiteCount());
    }
}
