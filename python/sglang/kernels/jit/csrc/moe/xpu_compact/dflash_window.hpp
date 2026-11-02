// SPDX-License-Identifier: Apache-2.0
#pragma once

#ifndef DFLASH_NATIVE_WINDOW
#define DFLASH_NATIVE_WINDOW 4
#endif
#if DFLASH_NATIVE_WINDOW != 4 && DFLASH_NATIVE_WINDOW != 8
#error "DFLASH_NATIVE_WINDOW must be 4 or 8"
#endif

// Keep device kernel names distinct when both window libraries are loaded.
#if DFLASH_NATIVE_WINDOW == 8
#define native_seq08 native_seq08_m8
#define dflash_planner dflash_planner_m8
#define dflash_routes01 dflash_routes01_m8
#define dflash_metadata dflash_metadata_m8
#define dflash_head_rescore_impl dflash_head_rescore_impl_m8
#define dflash_target_head_impl dflash_target_head_impl_m8
#define dflash_head_conditional_impl dflash_head_conditional_impl_m8
#endif
