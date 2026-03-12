package org.spartan.api.exception;

/**
 * Exception thrown when a native Spartan engine operation fails.
 * <p>
 * This can occur during:
 * <ul>
 *   <li>Model registration/unregistration</li>
 *   <li>Tick operations</li>
 *   <li>Any FFM call that returns an error code</li>
 * </ul>
 */
public class SpartanNativeException extends RuntimeException {

    private final int errorCode;

    /**
     * Creates a new native exception with a message and error code.
     *
     * @param message   the error message
     * @param errorCode the native error code returned by C++
     */
    public SpartanNativeException(String message, int errorCode) {
        super(message + " (error code: " + errorCode + ")");
        this.errorCode = errorCode;
    }

    /**
     * Creates a new native exception with a message, cause, and error code.
     *
     * @param message   the error message
     * @param cause     the underlying cause
     * @param errorCode the native error code
     */
    public SpartanNativeException(String message, Throwable cause, int errorCode) {
        super(message + " (error code: " + errorCode + ")", cause);
        this.errorCode = errorCode;
    }

    /**
     * Returns the native error code from C++.
     *
     * @return the error code (-1 typically indicates failure)
     */
    public int getErrorCode() {
        return errorCode;
    }
}
