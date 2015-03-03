#pragma once
#include <assert.h>
#include <iostream>

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

//////////////////////////////////////////////////////////////////////////
// Math and base include
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/System/Util/hkMemoryInitUtil.h>
#include <Common/Base/System/Error/hkDefaultError.h>
#include <Common/Base/Monitor/hkMonitorStream.h>
#include <Common/Base/Config/hkConfigVersion.h>
#include <Common/Base/Memory/System/hkMemorySystem.h>
#include <Common/Base/Memory/Allocator/Malloc/hkMallocAllocator.h>
#include <Common/Base/Container/String/hkStringBuf.h>

// Dynamics includes
#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Agent/ConvexAgent/SphereBox/hkpSphereBoxAgent.h>
#include <Physics2012/Collide/Shape/Convex/Box/hkpBoxShape.h>
#include <Physics2012/Collide/Shape/Convex/Sphere/hkpSphereShape.h>
#include <Physics2012/Collide/Dispatch/hkpAgentRegisterUtil.h>


#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Utilities/Dynamics/Inertia/hkpInertiaTensorComputer.h>

// Visual Debugger includes
#include <Common/Visualize/hkVisualDebugger.h>
#include <Physics2012/Utilities/VisualDebugger/hkpPhysicsContext.h>


#include <stdio.h>
#include <stdlib.h>

#if defined( HK_PLATFORM_TIZEN )
#include <osp/FBase.h>
#endif

#ifdef HK_PLATFORM_CTR
#define PRINTF nndbgDetailPrintf
#elif defined(HK_PLATFORM_ANDROID)
#include <android/log.h>
#define PRINTF(...) __android_log_print(ANDROID_LOG_INFO, "Havok", __VA_ARGS__)
#elif defined(HK_PLATFORM_TIZEN)
#define PRINTF(...) AppLogTag("Havok", __VA_ARGS__)
#else
#define PRINTF printf
#endif

#ifdef HK_ANDROID_PLATFROM_8
#    include <jni.h>
#endif

static void HK_CALL errorReport(const char* msg, void*)
{
	PRINTF("%s", msg);
#ifdef HK_PLATFORM_WIN32
#ifndef HK_PLATFORM_WINRT
	OutputDebugStringA(msg);
#else
	// Unicode only 
	int sLen = hkString::strLen(msg) + 1;
	wchar_t* wideStr = hkAllocateStack<wchar_t>( sLen );
	mbstowcs_s(HK_NULL, wideStr, sLen, msg, sLen - 1); 
	OutputDebugString(wideStr);
	hkDeallocateStack<wchar_t>( wideStr, sLen);
#endif
#endif
}
