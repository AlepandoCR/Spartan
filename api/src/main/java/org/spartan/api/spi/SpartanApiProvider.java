package org.spartan.api.spi;

import org.spartan.api.SpartanApi;

public interface SpartanApiProvider {
    SpartanApi createApi();
}
