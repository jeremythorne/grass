#include "mymath.h"

#define SOKOL_IMPL
#define SOKOL_GLES3
#include "sokol_gfx.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <stdio.h>
#include <string.h>
/* a uniform block with a model-view-projection matrix */
typedef struct {
    mat4s mvp;
} params_t;

typedef struct {
    mat4s mv;
    mat4s mvp;
} grass_params_t;


typedef struct {
    vec3s position;
    vec3s normal;
    vec3s colour;
    vec2s texcoord;
    float density;
} vertex_t;

#define POD
#define NOT_INTEGRAL
#define T vertex_t
#include <ctl/vector.h>

#define POD
#define T uint8_t
#include <ctl/vector.h>

typedef struct {
    vec2s offset;
    float scale;
} atlas_t;

typedef enum object_type_e {
    GROUND,
    GRASS,
    GRASS_V_SLICES,
    GRASS_H_SLICE,
    MAX_OBJECT_TYPE
} object_type_e;

typedef struct {
    long frame;
    vec_vertex_t vertices[MAX_OBJECT_TYPE];
    vec_uint8_t pixels[MAX_OBJECT_TYPE];
    sg_image img[MAX_OBJECT_TYPE];
    sg_bindings bind[MAX_OBJECT_TYPE];
    sg_pipeline pip[MAX_OBJECT_TYPE];
    sg_pipeline grass_density_pip;
    sg_pass_action pass_action;
    sg_pass offscreen_h_slice_pass;
    sg_pass offscreen_v_slice_pass;
    sg_pass_action offscreen_pass_action;
    mat4s view;
    mat4s view_proj;
    float rx;
    float ry;
} renderer_t;

// recursively subdivide the image, assumes the source is power of two square
sg_image_desc mip_chain(size_t dim_w, size_t dim_h, uint8_t *data, vec_uint8_t *output) {
    size_t width = dim_w;
    size_t height = dim_h;
    // a mip map chain always fits in twice the original size
    vec_uint8_t_resize(output, dim_w * dim_h * 4 * 2, 0);
    uint8_t *p = vec_uint8_t_data(output);
    memcpy(p, data, dim_w * dim_h * 4);
    int num_mipmaps = 0;
    sg_image_data img_data;
    do {
        size_t size = dim_w * dim_h * 4;
        img_data.subimage[0][num_mipmaps].ptr = p;
        img_data.subimage[0][num_mipmaps].size = size;

        num_mipmaps += 1;
        uint8_t * src = p;
        p += size;
        dim_w = dim_w >> 1;
        dim_h = dim_h >> 1;
        for (size_t y = 0; y < dim_h; y++) {
            for (size_t x = 0; x < dim_w; x++) {
                for (size_t k = 0; k < 4; k++) {
                    size_t s00 = k + (x * 2 + y * 4 * dim_w) * 4; 
                    p[k + (x + y * dim_w) * 4] =
                        (src[s00] +
                        src[s00 + 4] +
                        src[s00 + 2 * dim_w * 4] +
                        src[s00 + 2 * dim_w * 4 + 4]) / 4;
                }
            }
        }
    } while(dim_w > 0 && dim_h > 0);

    printf("%d x %d X %d\n", width, height, num_mipmaps);
    sg_image_desc img_desc = {
        .width = width,
        .height = height,
        .num_mipmaps = num_mipmaps,
        .pixel_format = SG_PIXELFORMAT_RGBA8,
        .mag_filter = SG_FILTER_LINEAR,
        .min_filter = SG_FILTER_LINEAR_MIPMAP_LINEAR,
        .data = img_data
    };
    return img_desc;
}

void renderer_free(renderer_t ** renderer) {
    if (*renderer) {
        for (int i = 0; i < MAX_OBJECT_TYPE; i++) {
            vec_vertex_t_free(&(*renderer)->vertices[i]);
            vec_uint8_t_free(&(*renderer)->pixels[i]);
        }
        sg_shutdown();
        free(*renderer);
        *renderer = NULL;
    }
}

static sg_pipeline create_grass_density_pipeline(size_t vertex_size) {
    // only outputs grass density attribute

    /* create shader */
    sg_shader shd = sg_make_shader(&(sg_shader_desc) {
        .vs.uniform_blocks[0] = {
            .size = sizeof(params_t),
            .uniforms = {
                [0] = { .name="mvp", .type=SG_UNIFORMTYPE_MAT4 }
            }
        },
        .vs.source =
            "#version 310 es\n"
            "uniform mat4 mv;\n"
            "uniform mat4 mvp;\n"
            "layout(location=0) in vec4 position;\n"
            "layout(location=1) in vec3 normal;\n"
            "layout(location=2) in vec4 colour;\n"
            "layout(location=3) in vec2 texcoord;\n"
            "layout(location=4) in float density;\n"
            "out float vdensity;\n" 
            "void main() {\n"
            "  vdensity = density;\n"
            "  gl_Position = mvp * position;\n"
            "}\n",
        .fs = {
            .images[0] = { .name="tex", .image_type = SG_IMAGETYPE_2D },
            .source =
                "#version 310 es\n"
                "precision mediump float;\n"
                "uniform sampler2D tex;"
                "in float vdensity;\n"
                "out vec4 frag_color;\n"
                "void main() {\n"
                "  frag_color = vec4(vec3(vdensity), 1.0);\n"
                "}\n"
        }
    });

    /* create pipeline object */
    sg_pipeline pip = sg_make_pipeline(&(sg_pipeline_desc){
        .layout = {
            /* test to provide buffer stride, but no attr offsets */
            .buffers[0].stride = vertex_size,
            .attrs = {
                [0].format=SG_VERTEXFORMAT_FLOAT3,
                [1].format=SG_VERTEXFORMAT_FLOAT3,
                [2].format=SG_VERTEXFORMAT_FLOAT3,
                [3].format=SG_VERTEXFORMAT_FLOAT2,
                [4].format=SG_VERTEXFORMAT_FLOAT,
            }
        },
        .shader = shd,
        .index_type = SG_INDEXTYPE_NONE,
        .depth = {
            .compare = SG_COMPAREFUNC_LESS_EQUAL,
            .write_enabled = true,
        },
        .colors[0] = {
            .blend = {
                .src_factor_rgb =  SG_BLENDFACTOR_SRC_ALPHA,
                .dst_factor_rgb = SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
                .enabled = true
            }
        },
        .face_winding = SG_FACEWINDING_CCW,
        //.cull_mode = SG_CULLMODE_BACK,
    });
    return pip;
}

static sg_pipeline create_grass_pipeline(size_t vertex_size) {
    /* create shader */
    sg_shader shd = sg_make_shader(&(sg_shader_desc) {
        .vs.uniform_blocks[0] = {
            .size = sizeof(grass_params_t),
            .uniforms = {
                [0] = { .name="mv", .type=SG_UNIFORMTYPE_MAT4 },
                [1] = { .name="mvp", .type=SG_UNIFORMTYPE_MAT4 }
            }
        },
        /* NOTE: since the shader defines explicit attribute locations,
           we don't need to provide an attribute name lookup table in the shader
        */
        .vs.source =
            "#version 310 es\n"
            "uniform mat4 mv;\n"
            "uniform mat4 mvp;\n"
            "layout(location=0) in vec4 position;\n"
            "layout(location=1) in vec3 normal;\n"
            "layout(location=2) in vec4 colour;\n"
            "layout(location=3) in vec2 texcoord;\n"
            "layout(location=4) in float density;\n"
            "out vec3 vnormal;\n" 
            "out vec4 vcolour;\n" 
            "out vec2 uv;\n" 
            "out float vdensity;\n" 
            "out vec4 view_pos;\n" 
            "void main() {\n"
            "  vnormal = normal;\n"
            "  vcolour = colour;\n"
            "  uv = texcoord;\n"
            "  vdensity = density;\n"
            "  view_pos = mv * position;\n"
            "  gl_Position = mvp * position;\n"
            "}\n",
        .fs = {
            .images[0] = { .name="tex", .image_type = SG_IMAGETYPE_2D },
            .source =
                "#version 310 es\n"
                "precision mediump float;\n"
                "uniform sampler2D tex;"
                "in vec3 vnormal;\n"
                "in vec4 vcolour;\n"
                "in vec2 uv;\n"
                "in float vdensity;\n"
                "in vec4 view_pos;\n"
                "out vec4 frag_color;\n"
                "void main() {\n"
                "  vec3 pos = view_pos.xyz/view_pos.w;\n"
                "  float distance = length(pos);\n"
                "  float density_map = 1.0 - smoothstep(0.5, 1.0, distance);\n"
                "  if (density_map < vdensity) { discard; };\n"
                "  vec3 light_dir = vec3(0.5, -0.5, 0.0);\n"
                "  vec3 light_colour = vec3(1.9, 1.9, 1.7);\n"
                "  vec3 ambient_colour = vec3(1.9, 1.9, 1.9);\n"
                "  float lambert = dot(light_dir, vnormal);\n"
                "  vec4 colour = vcolour * texture(tex, uv);\n"
                "  if (colour.a < 0.5) { discard; };\n"
                "  frag_color = colour * vec4(lambert * light_colour + ambient_colour, 1.0);\n"
                "}\n"
        }
    });

    /* create pipeline object */
    sg_pipeline pip = sg_make_pipeline(&(sg_pipeline_desc){
        .layout = {
            /* test to provide buffer stride, but no attr offsets */
            .buffers[0].stride = vertex_size,
            .attrs = {
                [0].format=SG_VERTEXFORMAT_FLOAT3,
                [1].format=SG_VERTEXFORMAT_FLOAT3,
                [2].format=SG_VERTEXFORMAT_FLOAT3,
                [3].format=SG_VERTEXFORMAT_FLOAT2,
                [4].format=SG_VERTEXFORMAT_FLOAT,
            }
        },
        .shader = shd,
        .index_type = SG_INDEXTYPE_NONE,
        .depth = {
            .compare = SG_COMPAREFUNC_LESS_EQUAL,
            .write_enabled = true,
        },
        .colors[0] = {
            .blend = {
                .src_factor_rgb =  SG_BLENDFACTOR_SRC_ALPHA,
                .dst_factor_rgb = SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
                .enabled = true
            }
        },
        .face_winding = SG_FACEWINDING_CCW,
        //.cull_mode = SG_CULLMODE_BACK,
    });
    return pip;
}

static sg_pipeline create_v_slice_pipeline(size_t vertex_size) {
    // unlit, discarding on alpha and distance

    /* create shader */
    sg_shader shd = sg_make_shader(&(sg_shader_desc) {
        .vs.uniform_blocks[0] = {
            .size = sizeof(grass_params_t),
            .uniforms = {
                [0] = { .name="mv", .type=SG_UNIFORMTYPE_MAT4 },
                [1] = { .name="mvp", .type=SG_UNIFORMTYPE_MAT4 }
            }
        },
        /* NOTE: since the shader defines explicit attribute locations,
           we don't need to provide an attribute name lookup table in the shader
        */
        .vs.source =
            "#version 310 es\n"
            "uniform mat4 mv;\n"
            "uniform mat4 mvp;\n"
            "layout(location=0) in vec4 position;\n"
            "layout(location=1) in vec3 normal;\n"
            "layout(location=2) in vec4 colour;\n"
            "layout(location=3) in vec2 texcoord;\n"
            "layout(location=4) in float density;\n"
            "out vec4 vcolour;\n" 
            "out vec2 uv;\n" 
            "out vec4 view_pos;\n" 
            "void main() {\n"
            "  vcolour = colour;\n"
            "  uv = texcoord;\n"
            "  view_pos = mv * position;\n"
            "  gl_Position = mvp * position;\n"
            "}\n",
        .fs = {
            .images[0] = { .name="tex", .image_type = SG_IMAGETYPE_2D },
            .source =
                "#version 310 es\n"
                "precision mediump float;\n"
                "uniform sampler2D tex;"
                "in vec4 vcolour;\n"
                "in vec2 uv;\n"
                "in vec4 view_pos;\n"
                "out vec4 frag_color;\n"
                "void main() {\n"
                "  vec3 pos = view_pos.xyz/view_pos.w;\n"
                "  float distance = length(pos);\n"
                "  float density_map = 1.0 - smoothstep(4.0, 5.0, distance);\n"
                "  density_map = density_map * smoothstep(0.2, 0.7, distance);\n"
                "  float density = texture(tex, uv + vec2(0.5, 0.0)).r;\n"
                "  if (density_map < density) { discard; };\n"
                "  vec4 colour = vcolour * texture(tex, uv);\n"
                "  if (colour.a < 0.5) { discard; };\n"
                "  frag_color = colour;\n"
                "}\n"
        }
    });

    /* create pipeline object */
    sg_pipeline pip = sg_make_pipeline(&(sg_pipeline_desc){
        .layout = {
            /* test to provide buffer stride, but no attr offsets */
            .buffers[0].stride = vertex_size,
            .attrs = {
                [0].format=SG_VERTEXFORMAT_FLOAT3,
                [1].format=SG_VERTEXFORMAT_FLOAT3,
                [2].format=SG_VERTEXFORMAT_FLOAT3,
                [3].format=SG_VERTEXFORMAT_FLOAT2,
                [4].format=SG_VERTEXFORMAT_FLOAT,
            }
        },
        .shader = shd,
        .index_type = SG_INDEXTYPE_NONE,
        .depth = {
            .compare = SG_COMPAREFUNC_LESS_EQUAL,
            .write_enabled = true,
        },
        .colors[0] = {
            .blend = {
                .src_factor_rgb =  SG_BLENDFACTOR_SRC_ALPHA,
                .dst_factor_rgb = SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
                .enabled = true
            }
        },
        .face_winding = SG_FACEWINDING_CCW,
        //.cull_mode = SG_CULLMODE_BACK,
    });
    return pip;
}

sg_pipeline create_basic_pipeline(size_t vertex_size) {
    // lambertian lighting

    /* create shader */
    sg_shader shd = sg_make_shader(&(sg_shader_desc) {
        .vs.uniform_blocks[0] = {
            .size = sizeof(params_t),
            .uniforms = {
                [0] = { .name="mvp", .type=SG_UNIFORMTYPE_MAT4 }
            }
        },
        /* NOTE: since the shader defines explicit attribute locations,
           we don't need to provide an attribute name lookup table in the shader
        */
        .vs.source =
            "#version 310 es\n"
            "uniform mat4 mvp;\n"
            "layout(location=0) in vec4 position;\n"
            "layout(location=1) in vec3 normal;\n"
            "layout(location=2) in vec4 colour;\n"
            "layout(location=3) in vec2 texcoord;\n"
            "out vec3 vnormal;\n" 
            "out vec4 vcolour;\n" 
            "out vec2 uv;\n" 
            "void main() {\n"
            "  vnormal = normal;\n"
            "  vcolour = colour;\n"
            "  uv = texcoord;\n"
            "  gl_Position = mvp * position;\n"
            "}\n",
        .fs = {
            .images[0] = { .name="tex", .image_type = SG_IMAGETYPE_2D },
            .source =
                "#version 310 es\n"
                "precision mediump float;\n"
                "uniform sampler2D tex;"
                "in vec3 vnormal;\n"
                "in vec4 vcolour;\n"
                "in vec2 uv;\n"
                "out vec4 frag_color;\n"
                "void main() {\n"
                "  vec3 light_dir = vec3(0.5, -0.5, 0.0);\n"
                "  vec3 light_colour = vec3(1.9, 1.9, 1.7);\n"
                "  vec3 ambient_colour = vec3(1.9, 1.9, 1.9);\n"
                "  float lambert = dot(light_dir, vnormal);\n"
                "  vec4 colour = vcolour * texture(tex, uv);\n"
                "  frag_color = colour * vec4(lambert * light_colour + ambient_colour, 1.0);\n"
                "}\n"
        }
    });

    /* create pipeline object */
    sg_pipeline pip = sg_make_pipeline(&(sg_pipeline_desc){
        .layout = {
            /* test to provide buffer stride, but no attr offsets */
            .buffers[0].stride = vertex_size,
            .attrs = {
                [0].format=SG_VERTEXFORMAT_FLOAT3,
                [1].format=SG_VERTEXFORMAT_FLOAT3,
                [2].format=SG_VERTEXFORMAT_FLOAT3,
                [3].format=SG_VERTEXFORMAT_FLOAT2,
            }
        },
        .shader = shd,
        .index_type = SG_INDEXTYPE_NONE,
        .depth = {
            .compare = SG_COMPAREFUNC_LESS_EQUAL,
            .write_enabled = true,
        },
        .colors[0] = {
            .blend = {
                .src_factor_rgb =  SG_BLENDFACTOR_SRC_ALPHA,
                .dst_factor_rgb = SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
                .enabled = true
            }
        },
        .face_winding = SG_FACEWINDING_CCW,
        //.cull_mode = SG_CULLMODE_BACK,
    });
    return pip;
}

sg_image load_image(const char *filename, vec_uint8_t *pixels) {
    sg_image img;
    if (strlen(filename)) {
        int x,y,n;
        uint8_t *data = stbi_load(filename, &x, &y, &n, 4);
        sg_image_desc img_desc = mip_chain(x, y, data, pixels);
        img = sg_make_image(&img_desc);
        stbi_image_free(data);
    }
    return img;
}

renderer_t * renderer_init(int width, int height) {
    /* setup sokol_gfx */
    sg_desc desc = {0};
    sg_setup(&desc);
    assert(sg_isvalid());

    /* view-projection matrix */
    mat4s proj = glms_perspective(glm_rad(60.0f), (float)width/(float)height, 0.1f, 20.0f);
    mat4s view = glms_lookat((vec3s){0.0f, 0.25f, 0.5f}, (vec3s){0.0f, 0.25f, 0.0f}, (vec3s){0.0f, 1.0f, 0.0f});
 
    size_t vertex_size = sizeof(vertex_t);

    renderer_t * renderer = malloc(sizeof(renderer_t));

    *renderer = (renderer_t){
        .frame = 0,
        .pip = {
            create_basic_pipeline(vertex_size),
            create_grass_pipeline(vertex_size),
            create_v_slice_pipeline(vertex_size),
            create_basic_pipeline(vertex_size),
        },
        .grass_density_pip = create_grass_density_pipeline(vertex_size),
        .pass_action = (sg_pass_action){ 0 },
        .view = view,
        .view_proj = glms_mat4_mul(proj, view),
        .offscreen_pass_action = (sg_pass_action){
            .colors[0] = { .action = SG_ACTION_CLEAR, .value = { 0.0f, 0.0f, 0.0f, 0.0f } }
        }
    };

    char texfile[MAX_OBJECT_TYPE][32] = {
        "mud.png",
        "leaf.png",
        "",
        ""
    };

    for (int i = 0; i < MAX_OBJECT_TYPE; i++) {
        renderer->vertices[i] = vec_vertex_t_init();
        renderer->pixels[i] = vec_uint8_t_init();
        renderer->img[i] = load_image(texfile[i], &renderer->pixels[i]);
    }

    /* offscreen pass */
    sg_image_desc img_desc = {
        .render_target = true,
        .width = 64,
        .height = 64,
        .min_filter = SG_FILTER_LINEAR,
        .mag_filter = SG_FILTER_LINEAR,
    };
    renderer->img[GRASS_H_SLICE] = sg_make_image(&img_desc);
    img_desc.pixel_format = SG_PIXELFORMAT_DEPTH_STENCIL;
    sg_image depth_h_slice_img = sg_make_image(&img_desc);

    /* an offscreen render pass into those images */
    renderer->offscreen_h_slice_pass = sg_make_pass(&(sg_pass_desc){
        .color_attachments[0].image = renderer->img[GRASS_H_SLICE],
        .depth_stencil_attachment.image = depth_h_slice_img
    });

    img_desc.pixel_format = SG_PIXELFORMAT_RGBA8;
    img_desc.width = img_desc.height = 512;
    renderer->img[GRASS_V_SLICES] = sg_make_image(&img_desc);
    img_desc.pixel_format = SG_PIXELFORMAT_DEPTH_STENCIL;
    sg_image depth_v_slice_img = sg_make_image(&img_desc);

    renderer->offscreen_v_slice_pass = sg_make_pass(&(sg_pass_desc){
        .color_attachments[0].image = renderer->img[GRASS_V_SLICES],
        .depth_stencil_attachment.image = depth_v_slice_img
    });

    return renderer;
}

static void add_quad(vec_vertex_t * vertices, mat4s m0, vec2s scale, vec3s colour,
        vec2s tex_off, vec2s tex_scale, float density) {
    // quad facing z axis
    vec3s scale3 = (vec3s) {scale.x, scale.y, 1.0f};
    vec3s va = glms_vec3_mul(scale3, (vec3s){-0.5f, 0.0f, 0.0f});
    vec3s vb = glms_vec3_mul(scale3, (vec3s){ 0.5f, 0.0f, 0.0f});
    vec3s vc = glms_vec3_mul(scale3, (vec3s){-0.5f, 1.0f, 0.0f});
    vec3s vd = glms_vec3_mul(scale3, (vec3s){ 0.5f, 1.0f, 0.0f});
    vec3s normal = transform_normal(m0, (vec3s){ 0.0f, 0.0f, 1.0f});

    vec2s ta = glms_vec2_add(tex_off, glms_vec2_mul((vec2s){0.0f, 0.0f}, tex_scale));
    vec2s tb = glms_vec2_add(tex_off, glms_vec2_mul((vec2s){1.0f, 0.0f}, tex_scale));
    vec2s tc = glms_vec2_add(tex_off, glms_vec2_mul((vec2s){0.0f, 1.0f}, tex_scale));
    vec2s td = glms_vec2_add(tex_off, glms_vec2_mul((vec2s){1.0f, 1.0f}, tex_scale));
 
    vertex_t v0 =  (vertex_t) {transform(m0, va), normal, colour, ta, density};
    vertex_t v1 =  (vertex_t) {transform(m0, vb), normal, colour, tb, density};
    vertex_t v2 =  (vertex_t) {transform(m0, vc), normal, colour, tc, density};
    vertex_t v3 =  (vertex_t) {transform(m0, vd), normal, colour, td, density};

    vertex_t triangles[] = {
        v0, v1, v2,
        v2, v1, v3,
    };

    for (int i = 0; i < 6; i++) {
        vec_vertex_t_push_back(vertices, triangles[i]);
    }
}

void renderer_add_blade_quad(renderer_t * renderer, mat4s m0, vec2s scale,
        vec3s colour, vec2s tex_off, vec2s tex_scale, float density) {
    add_quad(&renderer->vertices[GRASS], m0, scale, colour, tex_off, tex_scale, density);
}

static void add_h_quad(vec_vertex_t * vertices) {
    mat4s m0 = glms_mat4_mul(glms_quat_mat4(glms_quatv(-GLM_PI_2, (vec3s){1.0f, 0.0f, 0.0f})),
        glms_translate(glms_mat4_identity(), (vec3s){0.0f, -0.5f, 0.0f}));
    vec2s scale = glms_vec2_one();
    vec3s colour = glms_vec3_one();
    vec2s tex_off = glms_vec2_zero();
    vec2s tex_scale = glms_vec2_one(); 
    add_quad(vertices, m0, scale, colour, tex_off, tex_scale, 1.0f);
}

static void add_z_slice(vec_vertex_t * vertices, int i, float off) {
    mat4s m0 = glms_translate(glms_mat4_identity(), (vec3s){0.0f, 0.0f, off});
    vec2s scale = (vec2s){1.0f, 0.1f};
    vec3s colour = glms_vec3_one();
    vec2s tex_off = (vec2s){0.0f, off + 0.5f};
    vec2s tex_scale = (vec2s){0.5f, 1.0f / 10}; 
    add_quad(vertices, m0, scale, colour, tex_off, tex_scale, 1.0f);
}

static void add_x_slice(vec_vertex_t * vertices, int i, float off) {
    mat4s m0 = glms_mat4_mul(
        glms_translate(glms_mat4_identity(), (vec3s){off, 0.0f, 0.0f}),
        glms_quat_mat4(glms_quatv(-GLM_PI_2, (vec3s){0.0f, 1.0f, 0.0f})));
    vec2s scale = (vec2s){1.0f, 0.1f};
    vec3s colour = glms_vec3_one();
    vec2s tex_off = (vec2s){0.0f, off + 0.5f};
    vec2s tex_scale = (vec2s){0.5f, 1.0f / 10}; 
    add_quad(vertices, m0, scale, colour, tex_off, tex_scale, 1.0f);
}

static void add_v_slices(vec_vertex_t * vertices) {
    for (int i = 0; i < 10; i++) {
        float off = -0.5f + i / 10.0f;
        add_z_slice(vertices, i, off);
        add_x_slice(vertices, i, off);
    }
}

void renderer_generate_grass_lod(renderer_t * renderer) {
    add_h_quad(&renderer->vertices[GRASS_H_SLICE]);
    add_v_slices(&renderer->vertices[GRASS_V_SLICES]);
}

static void add_horiz_triangle(vec_vertex_t * vertices, vec3s origin, float radius,
        atlas_t tex) {
    // a 2D triangle
    const float pi = 3.1416f;
    float a = cos(pi / 3.0f);
    float b = sin(pi / 3.0f);
    vec3s c[3] = {
        (vec3s){0.0f, 0.0f, -1.0f},
        (vec3s){  -b, 0.0f,     a},
        (vec3s){   b, 0.0f,     a}
    };
    vec3s normal = (vec3s){0.0f, 1.0f, 0.0f};
    vec3s colour = (vec3s){1.0f, 1.0f, 1.0f};
    for(int i = 0; i < 3; i++) { 
        vec3s t = glms_vec3_add(origin, glms_vec3_scale(c[i], radius));
        vec2s uv = glms_vec2_add(tex.offset,
                    glms_vec2_add((vec2s){0.5f, 0.5f}, 
                        glms_vec2_scale((vec2s){c[i].x, c[i].z}, tex.scale * 0.5f)));
        vertex_t v = (vertex_t) {t, normal, colour, uv, 1.0f};
        vec_vertex_t_push_back(vertices, v);
    }
}

void renderer_clear_vertices(renderer_t * renderer) {
    for (int i = 0; i < MAX_OBJECT_TYPE; i++) {
        vec_vertex_t_clear(&renderer->vertices[i]);
    }
}

void renderer_add_ground_plane(renderer_t * renderer, float radius) {
    add_horiz_triangle(&renderer->vertices[GROUND], (vec3s){0.0f, -0.001f, 0.0f},
        radius, (atlas_t){.scale = radius * 0.25f });
}

static void upload_vertices(size_t vertex_size, int frame, vec_vertex_t *vertices, 
        sg_bindings *bind, sg_image *image) {
    size_t size = 
        vec_vertex_t_size(vertices) * vertex_size;

    if (size == 0) {
        return;
    }

    if (frame > 0) {
        sg_destroy_buffer(bind->vertex_buffers[0]);
    }
    sg_buffer vbuf = sg_make_buffer(&(sg_buffer_desc){
        .data = (sg_range){vec_vertex_t_data(vertices), size}
    });
    *bind = (sg_bindings) {
        .vertex_buffers[0] = vbuf,
        .fs_images[0] = *image
    };
}

void renderer_upload_vertices(renderer_t * renderer) {
    for (int i = 0; i < MAX_OBJECT_TYPE; i++) {
        upload_vertices(sizeof(vertex_t), 
            renderer->frame, &renderer->vertices[i], 
            &renderer->bind[i], &renderer->img[i]);
    }
}

void renderer_update(renderer_t * renderer) {
    /* rotated model matrix */
    // app->rx += 0.1f; 
    renderer->ry += 0.2f;
}

static void render_mesh(renderer_t * renderer, mat4s world, object_type_e obj_type) {
    params_t vs_params;
    vs_params.mvp = glms_mat4_mul(renderer->view_proj, world);
    sg_apply_uniforms(SG_SHADERSTAGE_VS, 0, &SG_RANGE(vs_params));
    size_t size = vec_vertex_t_size(&renderer->vertices[obj_type]);
    sg_apply_bindings(&renderer->bind[obj_type]);
    sg_draw(0, size, 1);
} 

static void render_grass_mesh(renderer_t * renderer, mat4s world, object_type_e obj_type) {
    grass_params_t vs_params;
    vs_params.mv = glms_mat4_mul(renderer->view, world);
    vs_params.mvp = glms_mat4_mul(renderer->view_proj, world);
    sg_apply_uniforms(SG_SHADERSTAGE_VS, 0, &SG_RANGE(vs_params));
    size_t size = vec_vertex_t_size(&renderer->vertices[obj_type]);
    sg_apply_bindings(&renderer->bind[obj_type]);
    sg_draw(0, size, 1);
} 

static vec3s project(mat4s mvp, vec3s pos) {
    vec4s projected = glms_mat4_mulv(mvp, glms_vec4(pos, 1.0f));
    return (vec3s) {
        projected.x / projected.w,
        projected.y / projected.w,
        projected.z / projected.w,
    };
}

static bool is_clipped_point(vec3s projected) {
        return projected.z < 0.0f ||
        projected.z > 1.0f ||
        projected.x < -1.0f ||
        projected.x > 1.0f ||
        projected.y < -1.0f ||
        projected.y > 1.0f;
}

static bool is_clipped_horiz_rect(mat4s mvp, vec2s min, vec2s max) {
    vec3s p[] = {
        (vec3s){min.x, 0.0f, min.y},
        (vec3s){min.x, 0.0f, max.y},
        (vec3s){max.x, 0.0f, min.y},
        (vec3s){max.x, 0.0f, max.y},
    };

    for(int i = 0; i < 4; i++) {
        if (!is_clipped_point(project(mvp, p[i]))) {
            return false;
        }
    }

    return true;
}

static void render_h_slice_offscreen(renderer_t * renderer) {
    sg_begin_pass(renderer->offscreen_h_slice_pass, &renderer->offscreen_pass_action);
    sg_apply_pipeline(renderer->pip[GROUND]);

    params_t vs_params;
    // orthographic projection looking down
    vs_params.mvp = glms_mat4_mul(glms_ortho_default_s(1.0f, 0.5f), 
        glms_lookat((vec3s){0.0f, 1.0f, 0.0f}, glms_vec3_zero(), (vec3s){0.0f, 0.0f, 1.0f}));
    sg_apply_uniforms(SG_SHADERSTAGE_VS, 0, &SG_RANGE(vs_params));

    // draw one grass patch
    size_t size = vec_vertex_t_size(&renderer->vertices[GRASS]);
    sg_apply_bindings(&renderer->bind[GRASS]);
    sg_draw(0, size, 1);
   
    sg_end_pass();
}

static void render_v_slice_offscreen(renderer_t * renderer) {
    sg_begin_pass(renderer->offscreen_v_slice_pass, &renderer->offscreen_pass_action);

    // render twice
    // * left half of texture - lit, full colour
    // * right half of texture - grass density
    for (int j = 0; j < 2; j++) {
        if (j == 0) {
            sg_apply_pipeline(renderer->pip[GROUND]);
        } else {
            sg_apply_pipeline(renderer->grass_density_pip);
        }
        for(int i = 0; i < 10; i++) {
            sg_apply_viewport(j * 256, i * 512 / 10, 256, 512/10, true);

            params_t vs_params;
            float far = -0.5f + i / 10.0f;
            float near = -0.5f + (i + 1) / 10.0f;
            // orthographic projection looking horizontally
            vs_params.mvp = glms_ortho(-0.5f,   0.5f,
                                       0.0f, 0.1f,
                                       near, far);
            sg_apply_uniforms(SG_SHADERSTAGE_VS, 0, &SG_RANGE(vs_params));

            // draw one grass patch
            size_t size = vec_vertex_t_size(&renderer->vertices[GRASS]);
            sg_apply_bindings(&renderer->bind[GRASS]);
            sg_draw(0, size, 1);
        }
    }
    sg_end_pass();
}

static void render_offscreen(renderer_t * renderer) {
    render_h_slice_offscreen(renderer);
    render_v_slice_offscreen(renderer);
}

void renderer_render(renderer_t * renderer, int cur_width, int cur_height) {

    if (renderer->frame == 0) {
        render_offscreen(renderer);
    }

    mat4s rxm = glms_quat_mat4(glms_quatv(glm_rad(renderer->rx), (vec3s){1.0f, 0.0f, 0.0f}));
    mat4s rym = glms_quat_mat4(glms_quatv(glm_rad(renderer->ry), (vec3s){0.0f, 1.0f, 0.0f}));
    mat4s world = glms_mat4_mul(rxm, rym);

    sg_begin_default_pass(&renderer->pass_action, cur_width, cur_height);

    object_type_e grass_types[] = {GRASS, GRASS_V_SLICES, GRASS_H_SLICE};

    for (int i = 0; i < 3; i++) {
        object_type_e grass_type = grass_types[i];
        sg_apply_pipeline(renderer->pip[grass_type]);
        // grass X x Z instances
        for (int x = -10; x < 10; x++) {
            for (int z = -10; z < 10; z++) {
                mat4s patch = glms_translate(glms_mat4_identity(),
                    (vec3s){x, 0.0f, z});
                mat4s model = glms_mat4_mul(world, patch);
                mat4s mv = glms_mat4_mul(renderer->view, model);
                mat4s mvp = glms_mat4_mul(renderer->view_proj, model);
                if (!is_clipped_horiz_rect(mvp, (vec2s){-0.5f, -0.5f}, (vec2s){0.5f, 0.5f})) {
                    float distance = glms_vec3_norm((transform(mv, glms_vec3_zero())));
                    float min_distance = distance - 0.5f;
                    float max_distance = distance + 0.5f;

                    if (min_distance < 1.0f && grass_type == GRASS) {
                        render_grass_mesh(renderer, model, GRASS);
                    }
                    if ((max_distance < 0.5f || min_distance < 5.0f) && grass_type == GRASS_V_SLICES) {
                        render_grass_mesh(renderer, model, GRASS_V_SLICES);
                    }
                    if (grass_type == GRASS_H_SLICE) {
                        render_mesh(renderer, model, GRASS_H_SLICE);
                    }
                }
            }
        }
    }
    // ground plane
    render_mesh(renderer, world, GROUND);
    
    sg_end_pass();
    sg_commit();
    renderer->frame++;
}


