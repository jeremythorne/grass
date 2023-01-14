#ifndef RENDERER_H
#define RENDERER_H

typedef struct renderer_s renderer_t;

void renderer_free(renderer_t ** renderer); 

renderer_t * renderer_init(int width, int height); 

void renderer_clear_vertices(renderer_t * renderer); 

void renderer_add_blade_quad(renderer_t * renderer, mat4s m0, vec2s scale,
            vec3s colour, vec2s tex_off, vec2s tex_scale, float density); 

void renderer_generate_grass_lod(renderer_t * renderer);

void renderer_add_ground_plane(renderer_t * renderer, float radius); 

void renderer_upload_vertices(renderer_t * renderer); 

void renderer_update(renderer_t * renderer);

void renderer_render(renderer_t * renderer, int cur_width, int cur_height);

#endif
