package org.spartan.api.exception;

/**
 * Exception thrown when model persistence operations fail.
 * <p>
 * This can occur during:
 * <ul>
 *   <li>Save operations (I/O failure, invalid agent ID)</li>
 *   <li>Load operations (file not found, CRC mismatch, format error)</li>
 * </ul>
 */
public class SpartanPersistenceException extends RuntimeException {

    /**
     * Creates a new persistence exception with a message.
     *
     * @param message the error message
     */
    public SpartanPersistenceException(String message) {
        super(message);
    }

    /**
     * Creates a new persistence exception with a message and cause.
     *
     * @param message the error message
     * @param cause   the underlying cause
     */
    public SpartanPersistenceException(String message, Throwable cause) {
        super(message, cause);
    }
}
