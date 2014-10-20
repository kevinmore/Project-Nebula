#pragma once
#include <Qt/qvector.h>
#include <Qt/qvector2d.h>
#include <Qt/qvector3d.h>
#include <Qt/qvector4d.h>
#include <Qt/qmatrix.h>
#include <Qt/qmatrix4x4.h>
#include <Qt/qstring.h>

typedef QVector2D vec2;
typedef QVector3D vec3;
typedef QVector4D vec4;
typedef QMatrix3x3 mat3;
typedef QMatrix4x4 mat4;

#define ZERO_MEM(a) memset(a, 0, sizeof(a))
#define ARRAY_SIZE_IN_ELEMENTS(a) (sizeof(a)/sizeof(*a))