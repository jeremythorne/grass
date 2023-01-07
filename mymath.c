#include "mymath.h"
#include <stdlib.h>

float rand_float(float s) {
    return (rand() * s) / RAND_MAX;
}

bool rand_prob(float p) {
    return rand_float(1.0f) < p;
}

vec3s rand_vec(float s) {
    float h = s / 2.0f;
    return (vec3s) {
        .x = rand_float(s) - h,
        .y = rand_float(s) - h,
        .z = rand_float(s) - h,
    };
}

mat4s mat_from_axes(
    vec3s x,
    vec3s y,
    vec3s z,
    vec3s t
    ) {
    mat4s result = {0};

    result.col[0] = glms_vec4(x, 0.0f);
    result.col[1] = glms_vec4(y, 0.0f);
    result.col[2] = glms_vec4(z, 0.0f);
    result.col[3] = glms_vec4(t, 1.0f);

    return result;
}

vec3s transform(mat4s m, vec3s v) {
    return glms_mat4_mulv3(m, v, 1.0f);
}

vec3s transform_normal(mat4s m, vec3s v) {
    return glms_mat4_mulv3(m, v, 0.0f);
}

void axes_from_dir_up(vec3s dir, vec3s up,
                vec3s *x, vec3s *y, vec3s *z) {
    *y = glms_vec3_normalize(dir);
    *x = glms_vec3_cross(*y, glms_vec3_normalize(up));
    *z = glms_vec3_cross(*x, *y);
}

