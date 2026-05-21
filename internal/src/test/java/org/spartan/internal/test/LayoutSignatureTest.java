package org.spartan.internal.test;

import org.junit.jupiter.api.Test;
import org.spartan.internal.bridge.SpartanNative;
import org.spartan.internal.engine.model.SpartanModelAllocator;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class LayoutSignatureTest {

    @Test
    public void nativeAndJvmLayoutSignaturesMatch() {
        // Force class initialization and retrieval of native signature
        int nativeSig = SpartanNative.spartanGetLayoutSignature();
        // SpartanNative static init should have already set the allocator, but ensure setter is called
        SpartanModelAllocator.setNativeLayoutSignature(nativeSig);

        int jvmSig = SpartanModelAllocator.getLayoutSignature();
        assertEquals(nativeSig, jvmSig, "Native layout signature and JVM-computed signature must match");
    }
}

