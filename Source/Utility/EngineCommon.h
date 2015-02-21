#pragma once
#include <assert.h>

//////////////////////////////////////////////////////////////////////////
#include <glm/glm.hpp>
#include <glm/gtc/bitfield.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/epsilon.hpp>
#include <glm/gtc/integer.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/matrix_integer.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/noise.hpp>
#include <glm/gtc/packing.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/random.hpp>
#include <glm/gtc/reciprocal.hpp>
#include <glm/gtc/round.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/ulp.hpp>
#include <glm/gtc/vec1.hpp>
#include <glm/gtx/string_cast.hpp>

//////////////////////////////////////////////////////////////////////////
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