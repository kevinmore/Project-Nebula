#pragma once
#include <assert.h>
#include <glm/glm.hpp>
#include <QMap>
#include <QLinkedList>
#include <QVector>
#include <QVector2D>
#include <QVector3D>
#include <QVector4D>
#include <QMatrix3x3>
#include <QMatrix4x4>
#include <QQuaternion>
#include <QString>
#include <QtMath>
#include <QSharedPointer>
#include <QColor>

typedef QVector2D vec2;
typedef QVector3D vec3;
typedef QVector4D vec4;
typedef QMatrix3x3 mat3;
typedef QMatrix4x4 mat4;
typedef QQuaternion quat;

#define ZERO_MEM(a) memset(a, 0, sizeof(a))
#define ARRAY_SIZE_IN_ELEMENTS(a) (sizeof(a)/sizeof(*a))
#define SAFE_DELETE(p) if (p) { delete p; p = NULL; }

#define FORCE_INLINE __forceinline