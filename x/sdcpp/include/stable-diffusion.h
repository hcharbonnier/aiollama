// stable-diffusion.h
//
// Vendored subset of the stable-diffusion.cpp public C API surface, used by
// the CGO bridge in x/sdcpp. This mirrors the declarations from
// https://github.com/leejet/stable-diffusion.cpp so the Go package can compile
// against the installed libstable-diffusion shared library without the full
// upstream source tree.
//
// The real header is fetched at CMake configure time; if present it takes
// precedence. This vendored copy provides the API contract the bridge targets.
#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
#if defined(SD_BUILD_SHARED_LIBS)
#if defined(sd_EXPORTS)
#define SD_API __declspec(dllexport)
#else
#define SD_API __declspec(dllimport)
#endif
#else
#define SD_API
#endif
#else
#if __GNUC__ >= 4
#define SD_API __attribute__((visibility("default")))
#else
#define SD_API
#endif
#endif

typedef struct sd_ctx_t sd_ctx_t;
typedef struct sd_image_t sd_image_t;

struct sd_image_t {
    int width;
    int height;
    int channel;
    uint8_t* data;
};

typedef struct sd_audio_t {
    uint32_t sample_rate;
    uint32_t sample_count;
    uint16_t channel_count;
    uint8_t* data;
} sd_audio_t;

typedef enum {
    SD_TYPE_F32  = 0,
    SD_TYPE_F16  = 1,
    SD_TYPE_Q4_0 = 2,
    SD_TYPE_Q4_1 = 3,
    SD_TYPE_Q5_0 = 6,
    SD_TYPE_Q5_1 = 7,
    SD_TYPE_Q8_0 = 8,
    SD_TYPE_Q8_1 = 9,
    SD_TYPE_Q2_K = 10,
    SD_TYPE_Q3_K = 11,
    SD_TYPE_Q4_K = 12,
    SD_TYPE_Q5_K = 13,
    SD_TYPE_Q6_K = 14,
    SD_TYPE_Q8_K = 15,
    SD_TYPE_IQ2_XXS = 16,
    SD_TYPE_IQ2_XS  = 17,
    SD_TYPE_IQ3_XXS = 18,
    SD_TYPE_IQ1_S   = 19,
    SD_TYPE_IQ4_NL  = 20,
    SD_TYPE_IQ3_S   = 21,
    SD_TYPE_IQ2_S   = 22,
    SD_TYPE_IQ4_XS  = 23,
    SD_TYPE_I8      = 24,
    SD_TYPE_I16     = 25,
    SD_TYPE_I32     = 26,
    SD_TYPE_I64     = 27,
    SD_TYPE_F64     = 28,
    SD_TYPE_BF16    = 29,
    SD_TYPE_COUNT,
} sd_type_t;

typedef enum {
    SD_SAMPLE_EULER_A = 0,
    SD_SAMPLE_EULER   = 1,
    SD_SAMPLE_HEUN    = 2,
    SD_SAMPLE_DPM2    = 3,
    SD_SAMPLE_DPMPP2M = 4,
    SD_SAMPLE_DPMPP2M_A = 5,
    SD_SAMPLE_LCM     = 6,
    SD_SAMPLE_DPMPP_SDE = 7,
    SD_SAMPLE_DPM_FAST = 8,
    SD_SAMPLE_DPM_ADAPTIVE = 9,
    SD_SAMPLE_TCD = 10,
    SD_SAMPLE_EULER_A_TRAILING = 11,
    SD_SAMPLE_EULER_TRAILING = 12,
} sd_sample_method_t;

typedef enum {
    SD_SCHEDULER_DEFAULT = 0,
    SD_SCHEDULER_DISCRETE = 1,
    SD_SCHEDULER_KARRAS = 2,
    SD_SCHEDULER_EXPONENTIAL = 3,
    SD_SCHEDULER_AYS = 4,
    SD_SCHEDULER_GITS = 5,
    SD_SCHEDULER_BETA = 6,
    SD_SCHEDULER_SIMPLE = 7,
    SD_SCHEDULER_UNIFORM = 8,
} sd_schedule_t;

typedef enum {
    SD_PREVIEW_NATIVE = 0,
    SD_PREVIEW_ACCURATE = 1,
} preview_t;

typedef enum {
    SD_CANCEL_NONE = 0,
    SD_CANCEL_IMAGE = 1,
    SD_CANCEL_ALL = 2,
} sd_cancel_mode_t;

typedef struct {
    int sample_steps;
    int thread_count;
    float cfg_scale;
    float guidance;
    float clip_skip;
    sd_sample_method_t sample_method;
    sd_schedule_t schedule;
    float flow_shift;
    float old_rate_coef;
    float omega_at_x0;
    float omega_at_xt;
    float omega_at_vt;
    int smpt_at;
    int smpt_at_dynamic_thresholding_max;
    float vary_at_x0;
    int vary_at_xt;
    float x0_weight;
    int xt_weight;
    float eta;
    int discrete_flow_shift;
    int neg_tau_at_x0;
    float neg_tau_at_xt;
    int use_karras;
    int use_beta_dy_shift;
    float beta_dy_shift_strength;
} sd_sample_params_t;

typedef struct {
    int enable;
    int upscale_factor;
    float strength;
    float denoise;
    float scale_emphasis;
} sd_tiling_params_t;

typedef struct {
    bool no_unload;
    bool model_offload;
    float moe_boundary;
    float vace_strength;
    bool use_cache;
    bool keep_model_loaded;
    bool neg_embd_mask;
    bool use_guidance;
    float guidance_scale;
    float seed_frame_idx;
} sd_cache_params_t;

typedef struct {
    const char* model_path;
    const char* clip_l_path;
    const char* clip_g_path;
    const char* clip_vision_path;
    const char* t5xxl_path;
    const char* vae_path;
    const char* taesd_path;
    const char* control_net_path;
    const char* backend;
    const char* params_backend;
    bool flash_attn;
    bool diffusion_flash_attn;
    bool vae_conv_direct;
    bool diffusion_conv_direct;
    sd_type_t wtype;
    int n_threads;
    bool enable_mmap;
    const char* max_vram;
    bool stream_layers;
    const char* lora_model_dir;
    size_t* lora_models;
    float* lora_multipliers;
    int n_lora_models;
    const char* embedding_path;
    const char* id_embeddings_path;
    bool rfb;
    const char* photo_maker_path;
    const char* ipadapter_path;
    const char* pulid_path;
    int pulid_strength;
    float ipadapter_strength;
    const char* boogu_edit_path;
    float boogu_edit_strength;
    int upscale_repeats;
    float style_ratio;
    bool normalization;
    bool clip_apply_weighted_sum;
    const char* koala_ii_path;
} sd_ctx_params_t;

typedef struct {
    const char* prompt;
    const char* negative_prompt;
    int width;
    int height;
    sd_sample_params_t sample_params;
    int64_t seed;
    int batch_count;
    sd_image_t init_image;
    sd_image_t mask_image;
    sd_image_t control_image;
    float control_strength;
    sd_tiling_params_t vae_tiling_params;
    const char* lora_models;
    const float* lora_multipliers;
    int n_lora_models;
    const char* id_embeddings_path;
    float style_ratio;
    bool normalization;
    bool clip_apply_weighted_sum;
    sd_image_t* pm_params;
    int n_pm_params;
    bool use_karras;
    sd_image_t* pulid_images;
    int n_pulid_images;
    bool is_kontext;
    sd_image_t* clip_vision_h;
    int clip_vision_h_count;
    sd_image_t* image2image_init;
    bool keep_model_loaded;
    bool use_cache;
    const char* upscale_path;
    int upscale_repeats;
    int upscale_factor;
    float upscale_strength;
    float upscale_denoise;
    bool video_generation;
    float image2image_strength;
    float image2image_steps;
} sd_img_gen_params_t;

typedef struct {
    const char* prompt;
    const char* negative_prompt;
    sd_image_t init_image;
    sd_image_t end_image;
    sd_image_t* control_frames;
    int width;
    int height;
    sd_sample_params_t sample_params;
    sd_sample_params_t high_noise_sample_params;
    float moe_boundary;
    float vace_strength;
    int64_t seed;
    int video_frames;
    int fps;
    sd_tiling_params_t vae_tiling_params;
    sd_cache_params_t cache;
} sd_vid_gen_params_t;

typedef void (*sd_progress_cb_t)(int step, int steps, float time, void* data);
typedef void (*sd_preview_cb_t)(int step, int frame_count, sd_image_t* frames, bool is_noisy, void* data);
typedef void (*sd_log_cb_t)(int level, char* text, void* data);

SD_API sd_ctx_t* new_sd_ctx(const sd_ctx_params_t* params);
SD_API void free_sd_ctx(sd_ctx_t* ctx);
SD_API bool sd_ctx_supports_image_generation(const sd_ctx_t* ctx);
SD_API bool sd_ctx_supports_video_generation(const sd_ctx_t* ctx);

SD_API bool generate_image(sd_ctx_t* ctx, const sd_img_gen_params_t* params,
                           sd_image_t** images_out, int* num_images_out);

SD_API bool generate_video(sd_ctx_t* ctx, const sd_vid_gen_params_t* params,
                           sd_image_t** frames_out, int* num_frames_out,
                           sd_audio_t** audio_out);

SD_API void free_sd_images(sd_image_t* images);
SD_API void free_sd_audio(sd_audio_t* audio);

SD_API void sd_set_progress_callback(sd_progress_cb_t cb, void* data);
SD_API void sd_set_preview_callback(sd_preview_cb_t cb, preview_t mode,
                                    int interval, bool denoised, bool noisy, void* data);
SD_API void sd_set_log_callback(sd_log_cb_t cb, void* data);
SD_API void sd_cancel_generation(sd_ctx_t* ctx, sd_cancel_mode_t mode);

SD_API const char* sd_get_system_info();
SD_API int sd_list_devices(int* count);

SD_API bool convert(const char* input_path, const char* output_path,
                    sd_type_t output_type, int unet_variant);
SD_API bool convert_with_components(const char* input_path, const char* output_dir,
                                    sd_type_t output_type);

SD_API const char* sd_type_name(sd_type_t type);
SD_API const char* sd_sample_method_name(sd_sample_method_t method);
SD_API const char* sd_schedule_name(sd_schedule_t schedule);

#ifdef __cplusplus
}
#endif
