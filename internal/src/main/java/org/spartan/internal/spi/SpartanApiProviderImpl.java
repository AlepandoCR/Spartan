package org.spartan.internal.spi;

import org.spartan.api.SpartanApi;
import org.spartan.api.spi.SpartanApiProvider;
import org.spartan.api.spi.SpartanCoreRegistry;
import org.spartan.internal.facade.SpartanApiImpl;

public class SpartanApiProviderImpl implements SpartanApiProvider {

    public SpartanApiProviderImpl() {
        SpartanCoreRegistry.set(this);
    }

    @Override
    public SpartanApi createApi() {
        return new SpartanApiImpl();
    }
}
