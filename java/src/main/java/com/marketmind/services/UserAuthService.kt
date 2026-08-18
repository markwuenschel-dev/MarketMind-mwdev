package com.marketmind.services

import org.springframework.stereotype.Service

/**
 * JAVA → KOTLIN changes:
 *
 * 1. `private var activeSessionToken: String? = null` — the `?` marks the type as nullable.
 *    This is the explicit Kotlin way of saying "this can be null". Java has no such distinction;
 *    all reference types are implicitly nullable in Java.
 *
 * 2. The function `isSessionValid()` — in Kotlin you could write this as a computed property:
 *    `val isSessionValid: Boolean get() = activeSessionToken != null`
 *    But a function is kept here to preserve the Java method call style (`service.isSessionValid()`).
 *
 * 3. No need for `null` check on String before `.equals()` — in Kotlin `"admin" == username`
 *    is null-safe (would be false if username were null, but it can't be null here since
 *    the parameter is `String`, not `String?`).
 */
@Service
class UserAuthService {

    // `String?` = nullable String (can hold null)
    private var activeSessionToken: String? = null

    fun authenticate(username: String, password: String): Boolean {
        val isAuthenticated = username == "admin" && password == "password"
        if (isAuthenticated) {
            activeSessionToken = "placeholder-session-token"
        }
        return isAuthenticated
    }

    // `!= null` check on nullable type — idiomatic Kotlin null check
    fun isSessionValid(): Boolean = activeSessionToken != null

    fun invalidateSession() {
        activeSessionToken = null
    }
}
