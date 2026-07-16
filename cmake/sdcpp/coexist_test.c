// Phase 0.4 coexistence validation: load libstable-diffusion (SD.cpp) and
// libggml-base (llama.cpp's vendored ggml) in the same process and call a
// symbol from each, confirming no duplicate-symbol linker errors occur.
//
// SD.cpp vendors its own ggml copy. Without the hidden-visibility + version-
// script isolation enforced in cmake/sdcpp/CMakeLists.txt, its ggml_* symbols
// would collide with llama.cpp's at link time or dynamic-load time. This
// program links against both shared libraries and resolves one symbol from
// each, proving the symbol namespaces are disjoint.
//
// Build/run via the `sdcpp-coexist` CMake target under cmake/sdcpp.

#include <stdio.h>

#include "stable-diffusion.h"

// Declared here (not via ggml.h) so the test does not depend on the llama.cpp
// include tree. The symbol is exported by libggml-base.so.
#ifdef __cplusplus
extern "C" {
#endif
const char *ggml_type_name(int type);
#ifdef __cplusplus
}
#endif

int main(void) {
    // SD.cpp symbol: sd_get_system_info returns a pointer to a static
    // std::string (not heap-allocated); do not free it.
    const char *sd_info = sd_get_system_info();
    if (sd_info == NULL) {
        fprintf(stderr, "FAIL: sd_get_system_info returned NULL\n");
        return 1;
    }
    printf("SD.cpp: %s\n", sd_info);

    // llama.cpp ggml symbol: ggml_type_name returns a static string.
    // type 0 == GGML_TYPE_F32 in ggml; any valid type demonstrates the symbol
    // resolves from libggml-base without colliding with SD.cpp's hidden copy.
    const char *type_name = ggml_type_name(0);
    if (type_name == NULL) {
        fprintf(stderr, "FAIL: ggml_type_name returned NULL\n");
        return 1;
    }
    printf("llama.cpp ggml: ggml_type_name(0) = %s\n", type_name);

    printf("PASS: SD.cpp and llama.cpp ggml coexist without symbol conflicts\n");
    return 0;
}
