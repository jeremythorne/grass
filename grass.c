#include "mymath.h"

#include "renderer.h"

#define GLFW_INCLUDE_NONE
#include "GLFW/glfw3.h"

#include <stdio.h>
#include <string.h>
#include <time.h>

typedef struct timespec timespec_t;

timespec_t now() {
    timespec_t now;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &now);
    return now;
}

long elapsed_ns(timespec_t start) {
    timespec_t end = now();
    return (long)(end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
}

typedef struct blade_s {
    vec3s origin;
    vec3s colour;
    float orientation;
    vec3s initial_velocity;
    float density;
} blade_t;

#define POD
#define NOT_INTEGRAL
#define T blade_t
#include <ctl/vector.h>

typedef struct {
    GLFWwindow * window;
    long frame;
    renderer_t *renderer;
    vec_blade_t blades;
} app_t;

void add_quads(renderer_t * renderer, const blade_t * blade) {
    vec3s x, y, z;
    vec3s velocity = blade->initial_velocity;
    vec3s origin = blade->origin;
    int N = 4;
    float Nf = (float)N;
    for(int i = 0; i < N; i++) {
        vec3s direction = glms_quat_rotatev(
            glms_quatv(blade->orientation, (vec3s){0.0f, 1.0f, 0.0f}),
            (vec3s){0.0f, 0.0f, 1.0f}); 
        axes_from_dir_up(velocity, direction, &x, &y, &z);
        mat4s m0 = mat_from_axes(x, y, z, origin);
        float length = glms_vec3_norm(velocity);
        vec2s tex_off = (vec2s){0.0f, i/Nf};
        vec2s tex_scale = (vec2s){1.0f, 1.0f/Nf};
        renderer_add_blade_quad(renderer, m0, (vec2s){0.005f, length}, blade->colour,
            tex_off, tex_scale, blade->density);
        origin = glms_vec3_add(origin, velocity);
        velocity = glms_vec3_add(velocity, (vec3s){0.0f, -0.008f, 0.0f});
    }
}

void create_geometry(app_t * app) {
    foreach(vec_blade_t, &app->blades, it) {
        blade_t *blade = it.ref;
        add_quads(app->renderer, blade);
    }
    renderer_add_ground_plane(app->renderer, 30.0f);
}

blade_t create_blade(vec3s origin) {
    return (blade_t){
        .origin =  origin,
        .colour = (vec3s){0.2f + rand_float(0.2f), 1.0f, 1.0f}, 
        .orientation = rand_float(GLM_PI),
        .initial_velocity = glms_vec3_add((vec3s){0.0f, 0.02f, 0.0f}, rand_vec(0.02f)),
        .density = 0.5f + rand_float(1.0f)
    };
}

void create_blades(app_t *app) {
    int A = 100;
    float horiz_scale = 1.0f / A;
    float off = 0.5f;

    for(int x = 0; x < A; x++) {
        for(int z = 0; z < A; z++) {
            vec3s root_pos =
                    glms_vec3_add(
                        glms_vec3_scale(
                            glms_vec3_add(
                                rand_vec(1.0f),
                                (vec3s){x, 0.0f, z}),
                            horiz_scale),
                        (vec3s){-off, 0.0f, -off});
            root_pos.y = 0.0f;
            blade_t blade = create_blade(root_pos);
            vec_blade_t_push_back(&app->blades, blade);
        }
    }
    create_geometry(app);
    renderer_upload_vertices(app->renderer);
}

bool should_quit(app_t * app) {
    return glfwWindowShouldClose(app->window);
}

void init(app_t * app) {
    const int WIDTH = 800;
    const int HEIGHT = 600;

    srand(time(0));

    /* create GLFW window and initialize GL */
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    GLFWwindow* w = glfwCreateWindow(WIDTH, HEIGHT, "Sokol Cube GLFW", 0, 0);
    glfwMakeContextCurrent(w);
    glfwSwapInterval(1);

    renderer_t * renderer = renderer_init(WIDTH, HEIGHT);

    *app = (app_t){
        .frame = 0,
        .window = w,
        .renderer = renderer,
        .blades = vec_blade_t_init(),
    };

    create_blades(app);
}

void update(app_t * app) {
    renderer_update(app->renderer);
}

void render(app_t * app) {
    int cur_width, cur_height;
    glfwGetFramebufferSize(app->window, &cur_width, &cur_height);
    renderer_render(app->renderer, cur_width, cur_height);
    glfwSwapBuffers(app->window);
    glfwPollEvents();
    app->frame++;
}

void terminate(app_t *app) {
    renderer_free(&app->renderer);
    vec_blade_t_free(&app->blades);
    glfwTerminate();
}

int main() {
    app_t app;
    init(&app);
    while(!should_quit(&app)) {
        update(&app);
        render(&app);
    }
    terminate(&app);
    return 0;
}
