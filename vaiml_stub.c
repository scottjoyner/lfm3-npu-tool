/*
 * Stub libvaiml.so - dummy implementations of all VAIML functions
 * Returns capability=1 for vaiml_get_capability_v5 to indicate NPU support
 */

#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef void* vaiml_context_t;
typedef void* vaiml_string_list_t;
typedef void* vaiml_string_map_t;
typedef void* vaiml_json_result_t;
typedef void* mladf_binary_map_t;

vaiml_context_t vaiml_context_create(void) { return (vaiml_context_t)0x1; }
void vaiml_context_destroy(vaiml_context_t ctx) { (void)ctx; }
void vaiml_context_set_input_path(vaiml_context_t ctx, const char* path) { (void)ctx; (void)path; }

vaiml_string_list_t vaiml_string_list_create(void) { return (vaiml_string_list_t)0; }
void vaiml_string_list_destroy(vaiml_string_list_t list) { (void)list; }
int vaiml_string_list_size(vaiml_string_list_t list) { (void)list; return 0; }
const char* vaiml_string_list_get(vaiml_string_list_t list, int idx) { (void)list; (void)idx; return NULL; }

vaiml_string_map_t vaiml_string_map_create(void) { return (vaiml_string_map_t)0; }
void vaiml_string_map_destroy(vaiml_string_map_t map) { (void)map; }
int vaiml_string_map_size(vaiml_string_map_t map) { (void)map; return 0; }
const char* vaiml_string_map_key(vaiml_string_map_t map, int idx) { (void)map; (void)idx; return NULL; }
const char* vaiml_string_map_values(vaiml_string_map_t map, const char* key) { (void)map; (void)key; return NULL; }

/* Return 1 = capable, to hint VitisAI EP that NPU can handle ops */
int vaiml_get_capability_v3(void) { return 1; }
int vaiml_get_capability_v4(void) { return 1; }
int vaiml_get_capability_v5(void) { return 1; }
const char* vaiml_get_version(void) { return "stub"; }
const char* vaiml_get_flexml_version(void) { return "stub"; }
vaiml_string_list_t vaiml_get_supported_opsets(void) { return NULL; }

int vaiml_compile_v3(void* ctx, const char* model_path, const char* options, vaiml_json_result_t* result) {
    (void)ctx; (void)model_path; (void)options; (void)result; return -1;
}
int vaiml_compile_v4(void* ctx, const char* model_path, const char* options, vaiml_json_result_t* result) {
    (void)ctx; (void)model_path; (void)options; (void)result; return -1;
}
int vaiml_compile_whitebox(void* ctx, const char* model_path, const char* options, vaiml_json_result_t* result) {
    (void)ctx; (void)model_path; (void)options; (void)result; return -1;
}
int vaiml_compile_x2(void* ctx, const char* model_path, const char* options, vaiml_json_result_t* result) {
    (void)ctx; (void)model_path; (void)options; (void)result; return -1;
}

vaiml_json_result_t vaiml_json_result_create(void) { return (vaiml_json_result_t)0; }
void vaiml_json_result_destroy(vaiml_json_result_t result) { (void)result; }
const char* vaiml_json_result_get_data(vaiml_json_result_t result) { (void)result; return NULL; }

mladf_binary_map_t mladf_binary_map_create(void) { return (mladf_binary_map_t)0; }
void mladf_binary_map_destroy(mladf_binary_map_t map) { (void)map; }
void mladf_binary_map_insert(mladf_binary_map_t map, const char* key, const char* value) { (void)map; (void)key; (void)value; }
const char* mladf_binary_map_get(mladf_binary_map_t map, const char* key) { (void)map; (void)key; return NULL; }
vaiml_string_list_t mladf_binary_map_list(mladf_binary_map_t map) { (void)map; return NULL; }
void mladf_string_list_destroy(vaiml_string_list_t list) { (void)list; }

int vaiml_dmac_compile(void* ctx, const char* input, const char* output, void* options) {
    (void)ctx; (void)input; (void)output; (void)options; return -1;
}
void* vaiml_dmac_compile_result_create(void) { return NULL; }
void vaiml_dmac_compile_result_destroy(void* result) { (void)result; }
void* vaiml_dmac_compile_result_from_file(const char* path) { (void)path; return NULL; }
int vaiml_dmac_get_return_code(void* result) { (void)result; return -1; }
vaiml_string_list_t vaiml_dmac_get_unique_nodes(void* result) { (void)result; return NULL; }
vaiml_string_list_t vaiml_dmac_unique_nodes_from_mlir_file(const char* path) { (void)path; return NULL; }

int vaiml_generate_xclbin(void* ctx, const char* input, const char* output, void* options) {
    (void)ctx; (void)input; (void)output; (void)options; return -1;
}

int dmac_partition_graph(void* ctx, const char* model, void* options) {
    (void)ctx; (void)model; (void)options; return -1;
}

void populateFileSystem(void) {}
