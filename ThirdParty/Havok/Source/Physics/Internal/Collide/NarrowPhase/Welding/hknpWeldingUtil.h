/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#ifndef HKNP_WELDING_UTIL_H
#define HKNP_WELDING_UTIL_H

#include <Geometry/Internal/Types/hkcdManifold4.h>
#include <Geometry/Internal/Types/hkcdVertex.h>
#include <Geometry/Internal/Algorithms/Welding/hkcdWeldingUtil.h>

namespace hkcdGsk { struct Cache; }
struct hknpShapeCollector;
struct hknpCdBodyBase;


/// Utility to help with welding
struct hknpWeldingUtil: public hkcdWeldingUtil
{
	struct ManifoldData
	{
		HK_FORCE_INLINE bool operator<( const ManifoldData& b ) const { return m_distance.m_asInt< b.m_distance.m_asInt; }

		hkVector4  m_triangleNormal;	// signed triangle normal
		hkVector4  m_normal;			// Set to gskNormal before calling neighborWeld
		hkVector4  m_gskPosition;

		union
		{
			hkUint32  m_asInt;
			hkFloat32 m_asFloat;		// the minimum distance of all points
		} m_distance;

		hkUint8 m_dimBInverse;			// Set to 3 minus the the dimension of the gskCache for object B
		hkBool  m_isTriangleOrQuad;		// Set if the object is a triangle or a quad

		hknpManifold*  m_manifold;
		hkUint16	   m_weldedBy;		// index of the Manifold data which welded this point (top bit is set if not welded)
	};

	static void HK_CALL neighborWeldConvex(
		const Config& config, const hknpCdBodyBase& cdBodyB,
		hknpManifold* manifolds, ManifoldData* extraDatas, int manifoldStriding, int numManifolds,
		hknpShapeCollector* leafShapeCollector );

	static void HK_CALL neighborWeld(
		hknpManifold* manifolds, ManifoldData* extraDatas, int manifoldStriding, int numManifolds,
		const hknpCdBody& cdBodyA, const hknpCdBodyBase& cdBodyB, hknpShapeCollector* shapeCollector,
		const Config& config, MotionWeldConfig* motionConfig );

	static hkResult HK_CALL setupMotionWeldConfigBodyB(
		const hknpCdBody& cdBodyA, const hknpCdBodyBase& cdBodyB, const hknpManifold& manifold, hkSimdRealParameter accuracy,
		MotionWeldConfig* config, hkcdVertex* vertexBufferB, hknpShapeCollector* leafShapeCollector );
};

#include <Physics/Internal/Collide/NarrowPhase/Welding/hknpWeldingUtil.inl>

#endif // HKNP_WELDING_UTIL_H

/*
 * Havok SDK - Product file, BUILD(#20130912)
 * 
 * Confidential Information of Havok.  (C) Copyright 1999-2013
 * Telekinesys Research Limited t/a Havok. All Rights Reserved. The Havok
 * Logo, and the Havok buzzsaw logo are trademarks of Havok.  Title, ownership
 * rights, and intellectual property rights in the Havok software remain in
 * Havok and/or its suppliers.
 * 
 * Use of this software for evaluation purposes is subject to and indicates
 * acceptance of the End User licence Agreement for this product. A copy of
 * the license is included with this software and is also available from salesteam@havok.com.
 * 
 */
