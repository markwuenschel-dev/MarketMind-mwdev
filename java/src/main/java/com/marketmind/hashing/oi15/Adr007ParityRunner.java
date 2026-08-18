package com.marketmind.hashing.oi15;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;

public final class Adr007ParityRunner {
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final List<String> SUITES = List.of(
        "blake3", "hmac_sha256", "jcs_sha256", "minhash",
        "rabin", "simhash", "sip24", "xxh3"
    );

    public record CaseResult(
        String suite,
        String caseId,
        String expectedOutput,
        String actualOutput,
        boolean success
    ) {}

    public record Report(
        String language,
        String toolchain,
        int suiteCount,
        int caseCount,
        boolean success,
        List<CaseResult> cases
    ) {}

    private Adr007ParityRunner() {}

    public static void main(String[] args) {
        try {
            Path root = Path.of("tests", "golden", "adr007");
            Path reportPath = null;
            for (int i = 0; i < args.length; i++) {
                if ("--root".equals(args[i]) && i + 1 < args.length) {
                    root = Path.of(args[++i]);
                } else if ("--report".equals(args[i]) && i + 1 < args.length) {
                    reportPath = Path.of(args[++i]);
                } else {
                    throw new IllegalArgumentException("Unsupported argument: " + args[i]);
                }
            }
            Report report = run(root);
            if (reportPath != null) {
                writeReport(reportPath, report);
            }
            System.exit(report.success() ? 0 : 1);
        } catch (Exception exc) {
            System.err.println("adr007 parity runner failed: " + exc.getClass().getSimpleName() + ": " + exc.getMessage());
            System.exit(2);
        }
    }

    public static Report run(Path root) throws Exception {
        List<CaseResult> results = new ArrayList<>();
        for (String suite : SUITES) {
            Path suiteRoot = root.resolve(suite);
            JsonNode manifest = MAPPER.readTree(suiteRoot.resolve("manifest.json").toFile());
            for (JsonNode caseSpec : manifest.get("cases")) {
                String expected = expectedOutput(suite, caseSpec, suiteRoot);
                String actual = actualOutput(suite, caseSpec, suiteRoot);
                results.add(new CaseResult(
                    suite,
                    caseSpec.get("id").asText(),
                    expected,
                    actual,
                    expected.equals(actual)
                ));
            }
        }
        boolean success = results.stream().allMatch(CaseResult::success);
        return new Report(
            "java-21",
            "Java " + System.getProperty("java.version"),
            SUITES.size(),
            results.size(),
            success,
            results
        );
    }

    private static void writeReport(Path reportPath, Report report) throws IOException {
        Files.createDirectories(reportPath.getParent());
        ObjectNode root = MAPPER.createObjectNode();
        root.put("language", report.language());
        root.put("toolchain", report.toolchain());
        root.put("suite_count", report.suiteCount());
        root.put("case_count", report.caseCount());
        root.put("success", report.success());
        ArrayNode cases = root.putArray("cases");
        for (CaseResult result : report.cases()) {
            ObjectNode node = cases.addObject();
            node.put("suite", result.suite());
            node.put("case_id", result.caseId());
            node.put("expected_output", result.expectedOutput());
            node.put("actual_output", result.actualOutput());
            node.put("success", result.success());
        }
        MAPPER.writeValue(reportPath.toFile(), root);
    }

    private static String expectedOutput(String suite, JsonNode caseSpec, Path suiteRoot) throws IOException {
        if ("rabin".equals(suite)) {
            return caseSpec.get("expected_fingerprint_hex").asText();
        }
        if ("jcs_sha256".equals(suite)) {
            return MAPPER.readTree(suiteRoot.resolve(caseSpec.get("path").asText()).toFile()).get("expected_digest_hex").asText();
        }
        if ("minhash".equals(suite)) {
            return MAPPER.readTree(suiteRoot.resolve(caseSpec.get("path").asText()).toFile()).get("expected_sip_key_hex").asText();
        }
        if ("simhash".equals(suite)) {
            return MAPPER.readTree(suiteRoot.resolve(caseSpec.get("path").asText()).toFile()).get("expected_projection_seed_hex").asText();
        }
        return caseSpec.get("expected_digest_hex").asText();
    }

    private static String actualOutput(String suite, JsonNode caseSpec, Path suiteRoot) throws Exception {
        Path casePath = suiteRoot.resolve(caseSpec.get("path").asText());
        byte[] caseBytes = Files.readAllBytes(casePath);
        if ("blake3".equals(suite)) {
            return toHex(Blake2b.digest(caseBytes, 32, new byte[0], "mm-b3-fallback"));
        }
        if ("hmac_sha256".equals(suite)) {
            return toHex(hmacSha256(fromHex(caseSpec.get("master_seed_hex").asText()), caseBytes));
        }
        if ("jcs_sha256".equals(suite)) {
            JsonNode payload = MAPPER.readTree(casePath.toFile()).get("payload");
            return toHex(sha256(canonicalJson(payload).getBytes(StandardCharsets.UTF_8)));
        }
        if ("minhash".equals(suite)) {
            JsonNode payload = MAPPER.readTree(casePath.toFile());
            byte[] ctx = concat("mm/minhash/v1".getBytes(StandardCharsets.UTF_8), u32be(payload.get("hash_family_index").asInt()));
            byte[] digest = hmacSha256(fromHex(payload.get("master_seed_hex").asText()), ctx);
            byte[] first16 = new byte[16];
            System.arraycopy(digest, 0, first16, 0, 16);
            return toHex(first16);
        }
        if ("rabin".equals(suite)) {
            return rabinFingerprint(caseBytes, caseSpec.get("window_size").asInt());
        }
        if ("simhash".equals(suite)) {
            JsonNode payload = MAPPER.readTree(casePath.toFile());
            byte[] ctx = concat(
                "mm/simhash/v1".getBytes(StandardCharsets.UTF_8),
                u32be(payload.get("dim").asInt()),
                u32be(payload.get("bit_index").asInt())
            );
            return toHex(sha256(concat(fromHex(payload.get("master_seed_hex").asText()), ctx)));
        }
        if ("sip24".equals(suite)) {
            byte[] namespaceBytes = caseSpec.get("namespace").asText().getBytes(StandardCharsets.UTF_8);
            byte[] preimage = buildCompositePreimage("mm/sip/v1", namespaceBytes, caseBytes);
            return toHex(Blake2b.digest(preimage, 8, fromHex(caseSpec.get("key_hex").asText()), "mm/sip24"));
        }
        if ("xxh3".equals(suite)) {
            return toHex(Blake2b.digest(caseBytes, 16, new byte[0], "mm-xxh3-128"));
        }
        throw new IllegalArgumentException("Unsupported suite: " + suite);
    }

    private static String canonicalJson(JsonNode node) {
        if (node.isObject()) {
            List<String> fields = new ArrayList<>();
            Iterator<String> names = node.fieldNames();
            while (names.hasNext()) {
                fields.add(names.next());
            }
            Collections.sort(fields);
            StringBuilder out = new StringBuilder("{");
            boolean first = true;
            for (String field : fields) {
                if (!first) {
                    out.append(',');
                }
                first = false;
                out.append('"').append(escape(field)).append('"').append(':').append(canonicalJson(node.get(field)));
            }
            return out.append('}').toString();
        }
        if (node.isArray()) {
            StringBuilder out = new StringBuilder("[");
            for (int i = 0; i < node.size(); i++) {
                if (i > 0) {
                    out.append(',');
                }
                out.append(canonicalJson(node.get(i)));
            }
            return out.append(']').toString();
        }
        if (node.isTextual()) {
            return "\"" + escape(node.textValue()) + "\"";
        }
        if (node.isBoolean()) {
            return node.booleanValue() ? "true" : "false";
        }
        // ADR-007 fixtures currently constrain canonicalized numbers to integral values.
        if (node.isIntegralNumber()) {
            return node.numberValue().toString();
        }
        if (node.isNull()) {
            return "null";
        }
        throw new IllegalArgumentException("Unsupported JSON node: " + node);
    }

    private static String escape(String text) {
        StringBuilder out = new StringBuilder();
        for (char ch : text.toCharArray()) {
            switch (ch) {
                case '"' -> out.append("\\\"");
                case '\\' -> out.append("\\\\");
                case '\b' -> out.append("\\b");
                case '\f' -> out.append("\\f");
                case '\n' -> out.append("\\n");
                case '\r' -> out.append("\\r");
                case '\t' -> out.append("\\t");
                default -> out.append(ch);
            }
        }
        return out.toString();
    }

    private static byte[] sha256(byte[] data) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        return digest.digest(data);
    }

    private static byte[] hmacSha256(byte[] key, byte[] message) throws Exception {
        Mac mac = Mac.getInstance("HmacSHA256");
        mac.init(new SecretKeySpec(key, "HmacSHA256"));
        return mac.doFinal(message);
    }

    private static byte[] fromHex(String hex) {
        byte[] out = new byte[hex.length() / 2];
        for (int i = 0; i < hex.length(); i += 2) {
            out[i / 2] = (byte) Integer.parseInt(hex.substring(i, i + 2), 16);
        }
        return out;
    }

    private static String toHex(byte[] data) {
        StringBuilder out = new StringBuilder();
        for (byte b : data) {
            out.append(String.format("%02x", Byte.toUnsignedInt(b)));
        }
        return out.toString();
    }

    private static byte[] u32be(int value) {
        return new byte[] {
            (byte) ((value >>> 24) & 0xFF),
            (byte) ((value >>> 16) & 0xFF),
            (byte) ((value >>> 8) & 0xFF),
            (byte) (value & 0xFF),
        };
    }

    private static byte[] concat(byte[]... values) {
        int size = 0;
        for (byte[] value : values) {
            size += value.length;
        }
        byte[] out = new byte[size];
        int offset = 0;
        for (byte[] value : values) {
            System.arraycopy(value, 0, out, offset, value.length);
            offset += value.length;
        }
        return out;
    }

    private static byte[] buildCompositePreimage(String domain, byte[] first, byte[] second) {
        byte[] domainBytes = domain.getBytes(StandardCharsets.UTF_8);
        byte[] fieldCount = new byte[] {0, 0, 0, 0, 0, 0, 0, 2};
        return concat(domainBytes, fieldCount, u64be(first.length), first, u64be(second.length), second);
    }

    private static byte[] u64be(long value) {
        return new byte[] {
            (byte) ((value >>> 56) & 0xFF),
            (byte) ((value >>> 48) & 0xFF),
            (byte) ((value >>> 40) & 0xFF),
            (byte) ((value >>> 32) & 0xFF),
            (byte) ((value >>> 24) & 0xFF),
            (byte) ((value >>> 16) & 0xFF),
            (byte) ((value >>> 8) & 0xFF),
            (byte) (value & 0xFF),
        };
    }

    private static String rabinFingerprint(byte[] data, int windowSize) {
        final long poly = 0x8000000000000003L;
        final long mask = (1L << 63) - 1L;
        long[] reduceTable = new long[256];
        long[] popTable = new long[256];
        for (int byteValue = 0; byteValue < 256; byteValue++) {
            long entry = byteValue;
            for (int i = 0; i < 56; i++) {
                long carry = (entry >>> 62) & 1L;
                entry = ((entry << 1) & mask) ^ (carry == 1L ? (poly & mask) : 0L);
            }
            reduceTable[byteValue] = entry & mask;
        }
        int shiftCount = Math.max(windowSize * 8, 1);
        for (int byteValue = 0; byteValue < 256; byteValue++) {
            long entry = byteValue;
            for (int i = 0; i < shiftCount; i++) {
                long carry = (entry >>> 62) & 1L;
                entry = ((entry << 1) & mask) ^ (carry == 1L ? (poly & mask) : 0L);
            }
            popTable[byteValue] = entry & mask;
        }
        long state = 0L;
        byte[] ring = new byte[windowSize];
        int position = 0;
        for (byte value : data) {
            int outgoing = Byte.toUnsignedInt(ring[position]);
            ring[position] = value;
            position = (position + 1) % ring.length;
            int highByte = (int) ((state >>> 55) & 0xFF);
            state = (
                reduceTable[highByte]
                    ^ ((state << 8) & mask)
                    ^ Byte.toUnsignedLong(value)
                    ^ popTable[outgoing]
            ) & mask;
        }
        return String.format("%016x", state);
    }
}
