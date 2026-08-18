package com.marketmind.hashing.oi15;

import java.util.Arrays;

final class Blake2b {
    private static final long[] IV = {
        0x6a09e667f3bcc908L, 0xbb67ae8584caa73bL, 0x3c6ef372fe94f82bL, 0xa54ff53a5f1d36f1L,
        0x510e527fade682d1L, 0x9b05688c2b3e6c1fL, 0x1f83d9abfb41bd6bL, 0x5be0cd19137e2179L,
    };

    private static final byte[][] SIGMA = {
        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
        { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
        { 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
        { 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
        { 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
        { 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
        { 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
        { 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
        { 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
        { 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
        { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
    };

    private final long[] h = Arrays.copyOf(IV, IV.length);
    private final long[] t = new long[2];
    private final byte[] buffer = new byte[128];
    private int bufferLength = 0;
    private final int outLength;

    Blake2b(int outLength, byte[] key, String personalization) {
        if (outLength <= 0 || outLength > 64 || key.length > 64) {
            throw new IllegalArgumentException("Invalid BLAKE2b parameters.");
        }
        this.outLength = outLength;
        byte[] param = new byte[64];
        param[0] = (byte) outLength;
        param[1] = (byte) key.length;
        param[2] = 1;
        param[3] = 1;
        byte[] personalizationBytes = personalization.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        System.arraycopy(personalizationBytes, 0, param, 48, Math.min(personalizationBytes.length, 16));
        for (int i = 0; i < 8; i++) {
            h[i] ^= load64(param, i * 8);
        }
        if (key.length > 0) {
            byte[] block = new byte[128];
            System.arraycopy(key, 0, block, 0, key.length);
            update(block);
        }
    }

    void update(byte[] data) {
        int offset = 0;
        while (offset < data.length) {
            if (bufferLength == 128) {
                incrementCounter(128);
                compress(false);
                bufferLength = 0;
            }
            int take = Math.min(128 - bufferLength, data.length - offset);
            System.arraycopy(data, offset, buffer, bufferLength, take);
            bufferLength += take;
            offset += take;
        }
    }

    byte[] digest() {
        incrementCounter(bufferLength);
        while (bufferLength < 128) {
            buffer[bufferLength++] = 0;
        }
        compress(true);
        byte[] out = new byte[outLength];
        for (int i = 0; i < out.length; i++) {
            out[i] = (byte) ((h[i / 8] >>> (8 * (i % 8))) & 0xFFL);
        }
        return out;
    }

    private static long load64(byte[] src, int offset) {
        long value = 0L;
        for (int i = 0; i < 8; i++) {
            value |= (long) (src[offset + i] & 0xFF) << (8 * i);
        }
        return value;
    }

    private void incrementCounter(int amount) {
        t[0] += amount;
        if (Long.compareUnsigned(t[0], Integer.toUnsignedLong(amount)) < 0) {
            t[1] += 1;
        }
    }

    private static long rotr64(long x, int n) {
        return (x >>> n) | (x << (64 - n));
    }

    private void g(long[] v, long[] m, int round, int a, int b, int c, int d, int x, int y) {
        v[a] = v[a] + v[b] + m[SIGMA[round][x]];
        v[d] = rotr64(v[d] ^ v[a], 32);
        v[c] = v[c] + v[d];
        v[b] = rotr64(v[b] ^ v[c], 24);
        v[a] = v[a] + v[b] + m[SIGMA[round][y]];
        v[d] = rotr64(v[d] ^ v[a], 16);
        v[c] = v[c] + v[d];
        v[b] = rotr64(v[b] ^ v[c], 63);
    }

    private void compress(boolean lastBlock) {
        long[] m = new long[16];
        long[] v = new long[16];
        for (int i = 0; i < 16; i++) {
            m[i] = load64(buffer, i * 8);
        }
        System.arraycopy(h, 0, v, 0, 8);
        System.arraycopy(IV, 0, v, 8, 8);
        v[12] ^= t[0];
        v[13] ^= t[1];
        if (lastBlock) {
            v[14] = ~v[14];
        }

        for (int round = 0; round < 12; round++) {
            g(v, m, round, 0, 4, 8, 12, 0, 1);
            g(v, m, round, 1, 5, 9, 13, 2, 3);
            g(v, m, round, 2, 6, 10, 14, 4, 5);
            g(v, m, round, 3, 7, 11, 15, 6, 7);
            g(v, m, round, 0, 5, 10, 15, 8, 9);
            g(v, m, round, 1, 6, 11, 12, 10, 11);
            g(v, m, round, 2, 7, 8, 13, 12, 13);
            g(v, m, round, 3, 4, 9, 14, 14, 15);
        }
        for (int i = 0; i < 8; i++) {
            h[i] ^= v[i] ^ v[i + 8];
        }
    }

    static byte[] digest(byte[] data, int outLength, byte[] key, String personalization) {
        Blake2b blake2b = new Blake2b(outLength, key, personalization);
        blake2b.update(data);
        return blake2b.digest();
    }
}
