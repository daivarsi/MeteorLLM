cmake_minimum_required(VERSION 3.16)
set(CMAKE_TOOLCHAIN_FILE "$ENV{VITASDK}/share/vita.toolchain.cmake")
project(MeteorLLM CXX)
include("$ENV{VITASDK}/share/vita.cmake" REQUIRED)
set(CMAKE_CXX_STANDARD 20)
add_executable(meteor vita_main.cpp engine.cpp quant.cpp tokenizer.cpp)
target_compile_options(meteor PRIVATE -O3 -Wall -Wextra -ffp-contract=off -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard)
target_link_libraries(meteor vita2d SceDisplay_stub SceGxm_stub SceSysmem_stub SceCtrl_stub ScePgf_stub SceCommonDialog_stub SceIme_stub SceSysmodule_stub ScePower_stub SceIofilemgr_stub SceProcessmgr_stub SceAppMgr_stub m)
vita_create_self(eboot.bin meteor UNSAFE)
# Use explicit file dependencies as well as target ordering. The SDK VPK
# helper in this pinned SDK does not quote resource paths containing spaces.
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/param.sfo"
 COMMAND "${VITA_MKSFOEX}" -s APP_VER=00.02 -s TITLE_ID=METEOR001 MeteorLLM "${CMAKE_CURRENT_BINARY_DIR}/param.sfo"
 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/eboot.bin" VERBATIM)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/MeteorLLM.vpk"
 COMMAND "${VITA_PACK_VPK}" -a "${CMAKE_CURRENT_SOURCE_DIR}/assets/icon0.png=sce_sys/icon0.png" -s "${CMAKE_CURRENT_BINARY_DIR}/param.sfo" -b "${CMAKE_CURRENT_BINARY_DIR}/eboot.bin" "${CMAKE_CURRENT_BINARY_DIR}/MeteorLLM.vpk"
 DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/eboot.bin" "${CMAKE_CURRENT_BINARY_DIR}/param.sfo" "${CMAKE_CURRENT_SOURCE_DIR}/assets/icon0.png" VERBATIM)
add_custom_target(meteor-vpk ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/MeteorLLM.vpk")
add_dependencies(meteor-vpk eboot.bin)
