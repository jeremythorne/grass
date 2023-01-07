#ifndef MATH_H
#define MATH_H

#include <cglm/struct.h>
#include <stdbool.h>

float rand_float(float s);

bool rand_prob(float p);

vec3s rand_vec(float s);

mat4s mat_from_axes(
    vec3s x,
    vec3s y,
    vec3s z,
    vec3s t
    );

vec3s transform(mat4s m, vec3s v);

vec3s transform_normal(mat4s m, vec3s v);

void axes_from_dir_up(vec3s dir, vec3s up,
                vec3s *x, vec3s *y, vec3s *z);

#endif
