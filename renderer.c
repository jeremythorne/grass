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
    GRASS_H_SLICE,
    MAX_OBJECT_TYPE
} object_type_e;

typedef struct {
    long frame;
    vec_vertex_t vertices[MAX_OBJECT_TYPE];
    int floats_per_vertex;
    vec_uint8_t pixels[MAX_OBJECT_TYPE];
    sg_image img[MAX_OBJECT_TYPE];
    sg_bindings bind[MAX_OBJECT_TYPE];
    sg_pipeline pip;
    sg_pipeline grass_pip;
    sg_pass_action pass_action;
    sg_pass offscreen_pass;
    sg_pass_action offscreen_pass_action;
    mat4s view;
    mat4s view_proj;
    float rx;
    float ry;
} renderer_t;

static void add_h_quad(renderer_t * renderer);

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

static sg_pipeline create_grass_pipeline(int floats_per_vertex) {
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
            .buffers[0].stride = floats_per_vertex * 4,
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

renderer_t * renderer_init(int width, int height) {
    /* setup sokol_gfx */
    sg_desc desc = {0};
    sg_setup(&desc);
    assert(sg_isvalid());

    renderer_t * renderer = malloc(sizeof(renderer_t));

    *renderer = (renderer_t){
        .frame = 0,
        .floats_per_vertex = sizeof(vertex_t) / sizeof(float),
    };

    renderer->grass_pip = create_grass_pipeline(renderer->floats_per_vertex);

    char texture_file[MAX_OBJECT_TYPE][32] = {
        "mud.png",
        "leaf.png",
        "",
    };

    for (int i = 0; i < MAX_OBJECT_TYPE; i++) {
        renderer->vertices[i] = vec_vertex_t_init();
        renderer->pixels[i] = vec_uint8_t_init();

        if (strlen(texture_file[i])) {
            int x,y,n;
            uint8_t *data = stbi_load(texture_file[i], &x, &y, &n, 4);
            sg_image_desc img_desc = mip_chain(x, y, data, &renderer->pixels[i]);
            renderer->img[i] = sg_make_image(&img_desc);
            stbi_image_free(data);
        }
    } 

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
    renderer->pip = sg_make_pipeline(&(sg_pipeline_desc){
        .layout = {
            /* test to provide buffer stride, but no attr offsets */
            .buffers[0].stride = renderer->floats_per_vertex * 4,
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

    /* default pass action */
    renderer->pass_action = (sg_pass_action){ 0 };

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
    sg_image depth_img = sg_make_image(&img_desc);

    /* an offscreen render pass into those images */
    renderer->offscreen_pass = sg_make_pass(&(sg_pass_desc){
        .color_attachments[0].image = renderer->img[GRASS_H_SLICE],
        .depth_stencil_attachment.image = depth_img
    });

    /* pass action for offscreen pass, clearing to black */
    renderer->offscreen_pass_action = (sg_pass_action){
        .colors[0] = { .action = SG_ACTION_CLEAR, .value = { 0.0f, 0.0f, 0.0f, 0.0f } }
    };

    /* view-projection matrix */
    mat4s proj = glms_perspective(glm_rad(60.0f), (float)width/(float)height, 0.1f, 20.0f);
    mat4s view = glms_lookat((vec3s){0.0f, 0.25f, 0.5f}, (vec3s){0.0f, 0.0f, 0.0f}, (vec3s){0.0f, 1.0f, 0.0f});
    renderer->view = view;
    renderer->view_proj = glms_mat4_mul(proj, view);

    add_h_quad(renderer);

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

static void add_h_quad(renderer_t * renderer) {
    mat4s m0 = glms_mat4_mul(glms_quat_mat4(glms_quatv(-GLM_PI_2, (vec3s){1.0f, 0.0f, 0.0f})),
        glms_translate(glms_mat4_identity(), (vec3s){0.0f, -0.5f, 0.0f}));
    vec2s scale = glms_vec2_one();
    vec3s colour = glms_vec3_one();
    vec2s tex_off = glms_vec2_zero();
    vec2s tex_scale = glms_vec2_one(); 
    add_quad(&renderer->vertices[GRASS_H_SLICE], m0, scale, colour, tex_off, tex_scale, 1.0f);
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

static void upload_vertices(int floats_per_vertex, int frame, vec_vertex_t *vertices, 
        sg_bindings *bind, sg_image *image) {
    size_t size = 
        vec_vertex_t_size(vertices) * floats_per_vertex * sizeof(float);

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
        upload_vertices(renderer->floats_per_vertex, 
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

float closest_corner_horiz_rect(mat4s mv, vec2s min, vec2s max) {
    vec3s p[] = {
        (vec3s){min.x, 0.0f, min.y},
        (vec3s){min.x, 0.0f, max.y},
        (vec3s){max.x, 0.0f, min.y},
        (vec3s){max.x, 0.0f, max.y},
    };
    
    float min_distance = INFINITY;
    for(int i = 0; i < 4; i++) {
        float distance = glms_vec3_norm((transform(mv, p[i])));
        min_distance = distance < min_distance ? distance : min_distance;
    }
    return min_distance;
}

static void render_offscreen(renderer_t * renderer) {
    sg_begin_pass(renderer->offscreen_pass, &renderer->offscreen_pass_action);
    sg_apply_pipeline(renderer->pip);

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

void renderer_render(renderer_t * renderer, int cur_width, int cur_height) {

    if (renderer->frame == 0) {
        render_offscreen(renderer);
    }

    mat4s rxm = glms_quat_mat4(glms_quatv(glm_rad(renderer->rx), (vec3s){1.0f, 0.0f, 0.0f}));
    mat4s rym = glms_quat_mat4(glms_quatv(glm_rad(renderer->ry), (vec3s){0.0f, 1.0f, 0.0f}));
    mat4s world = glms_mat4_mul(rxm, rym);

    sg_begin_default_pass(&renderer->pass_action, cur_width, cur_height);

    object_type_e grass_types[] = {GRASS, GRASS_H_SLICE};

    for (int i = 0; i < 2; i++) {
        object_type_e grass_type = grass_types[i];
        if (grass_type == GRASS) {
            sg_apply_pipeline(renderer->grass_pip);
        } else {
            sg_apply_pipeline(renderer->pip);
        }
        // grass X x Z instances
        for (int x = -10; x < 10; x++) {
            for (int z = -10; z < 10; z++) {
                mat4s patch = glms_translate(glms_mat4_identity(),
                    (vec3s){x, 0.0f, z});
                mat4s model = glms_mat4_mul(world, patch);
                mat4s mv = glms_mat4_mul(renderer->view, model);
                mat4s mvp = glms_mat4_mul(renderer->view_proj, model);
                if (!is_clipped_horiz_rect(mvp, (vec2s){-0.5f, -0.5f}, (vec2s){0.5f, 0.5f})) {
                    float distance = closest_corner_horiz_rect(mv, 
                        (vec2s){-0.5f, -0.5f}, (vec2s){0.5f, 0.5f});
                    if (distance < 1.0f && grass_type == GRASS) {
                        render_grass_mesh(renderer, model, GRASS);
                    } else if (grass_type == GRASS_H_SLICE) {
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


