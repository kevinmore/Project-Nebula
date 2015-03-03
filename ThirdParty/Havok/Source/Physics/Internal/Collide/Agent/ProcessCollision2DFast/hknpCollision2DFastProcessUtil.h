/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#ifndef HKNP_COLLISION_2D_FAST_PROCESS_UTIL_H
#define HKNP_COLLISION_2D_FAST_PROCESS_UTIL_H

#include <Physics/Physics/Collide/Shape/Convex/hknpConvexShape.h>

class hknpConvexShape;
struct hknpCdBody;


/// Class which provides all the functions necessary to run convex convex collision detection.
class hknpCollision2DFastProcessUtil
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpCollision2DFastProcessUtil );

		//
		static int HK_CALL collide(
			const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData, const hkTransform& aTb, hkBool32 delayProcessCallback,
			const hknpCdBody& cdBodyA, const hknpCdBody& cdBodyB,
			struct hknpConvexConvexCollisionCache* HK_RESTRICT cache, struct hknpConvexConvexCollisionCache* HK_RESTRICT cacheOnPpu,
			hknpManifold* HK_RESTRICT manifoldsOut );

		//
		static int HK_CALL collideFlat(
			const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData, const hkTransform& aTb, hkBool32 delayProcessCallback,
			const hknpCdBody& cdBodyA, const hknpCdBody& cdBodyB,
			struct hknpConvexConvexCollisionCache* HK_RESTRICT cache, struct hknpConvexConvexCollisionCache* HK_RESTRICT cacheOnPpu,
			hknpManifold* HK_RESTRICT manifoldsOut );

		//
		static HK_FORCE_INLINE void HK_CALL getIndexedVertices(
			const hknpConvexShape* HK_RESTRICT shape, int faceIx,
			int& numVerticesOut, const hkVector4* HK_RESTRICT &verticesOut,
			const hknpConvexShape::VertexIndex* HK_RESTRICT &indicesOut );
};


#endif	// HKNP_COLLISION_2D_FAST_PROCESS_UTIL_H

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
