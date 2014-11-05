#pragma once
#include <QtCore/QVector>
#include <QtCore/QMap>
#include <QtGui/QVector2D>
#include <QtGui/QVector3D>
#include <QtGui/QVector4D>
#include <QtGui/QMatrix3x3>
#include <QtGui/QMatrix4x4>
#include <QtCore/QString>
#include <QtGui/QQuaternion>
#include <QtCore/QtMath>

typedef QVector2D vec2;
typedef QVector3D vec3;
typedef QVector4D vec4;
typedef QMatrix3x3 mat3;
typedef QMatrix4x4 mat4;

#define ZERO_MEM(a) memset(a, 0, sizeof(a))
#define ARRAY_SIZE_IN_ELEMENTS(a) (sizeof(a)/sizeof(*a))