/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#ifndef HKNP_CVX_COLLISION_PROCESS_HELPER_H
#define HKNP_CVX_COLLISION_PROCESS_HELPER_H


#if defined(HK_DEBUG)
#	include <Common/Visualize/hkDebugDisplay.h>
#endif
#include <Physics/Physics/Collide/Shape/Convex/hknpConvexShape.h>

class hknpConvexShape;
struct hknpCdBody;

#if 0
#	define HKGSK_TIMER_BEGIN_LIST2( stream, ITEM, ITEM2 )			HK_TIMER_BEGIN_LIST2( stream, ITEM, ITEM2 )
#	define HKGSK_TIMER_SPLIT_LIST2( stream, ITEM )					HK_TIMER_SPLIT_LIST2( stream, ITEM )
#	define HKGSK_TIMER_END_LIST2( stream )							HK_TIMER_END_LIST2( stream );
#else
#	define HKGSK_TIMER_BEGIN_LIST2( stream, ITEM, ITEM2 )
#	define HKGSK_TIMER_SPLIT_LIST2( stream, ITEM )
#	define HKGSK_TIMER_END_LIST2( stream )
#endif

/// Class with some internal helper methods for cvx collision detection
class hknpCvxCollisionProcessHelper
{
	public:
		static HK_FORCE_INLINE int calcAngTim( const int minAngleA, const int minAngleB, hkVector4Parameter planeAinA, hkVector4Parameter planeBinA );

		static HK_FORCE_INLINE int calcAngTim_2(
		const hknpConvexPolytopeShape* HK_RESTRICT shapeA, const hknpConvexPolytopeShape* HK_RESTRICT shapeB,
		int faceIndexA, int faceIndexB,
		hkVector4Parameter planeAinA, hkVector4Parameter planeBinA,
		hkVector4Parameter normalInA, const hkRotation& aTbRotation);
};

/// Gsk interface for hknpConvexShape - hknpConvexShape.
struct hknpConvexConvexShapeInterface : public hkcdGskBase::ShapeInterface
{
	HK_FORCE_INLINE virtual void getSupportingVertices(
		const void* HK_RESTRICT shapeA, hkVector4Parameter directionA,
		const void* HK_RESTRICT shapeB, const hkTransform& aTb,
		hkcdVertex* HK_RESTRICT vertexAinAOut, hkcdVertex* HK_RESTRICT vertexBinBOut, hkVector4* vertexBinAout
		) const
	{
		const hknpConvexShape* HK_RESTRICT csA = static_cast<const hknpConvexShape*>(shapeA);
		const hknpConvexShape* HK_RESTRICT csB = static_cast<const hknpConvexShape*>(shapeB);
		hknpConvexShapeUtil::getSupportingVertices( *csA, directionA, *csB, aTb, vertexAinAOut, vertexBinBOut, vertexBinAout );
	}

	HK_FORCE_INLINE void _unpackCache(
		const hkcdGsk::Cache* cache,
		const void* shapeA, const void* shapeB,
		hkPadSpu<int>& dimA, hkcdVertex* verticesOfAinA,
		hkPadSpu<int>& dimB, hkcdVertex* verticesOfBinB
		) const
	{
		const hknpConvexShape* HK_RESTRICT csA = static_cast<const hknpConvexShape*>(shapeA);
		const hknpConvexShape* HK_RESTRICT csB = static_cast<const hknpConvexShape*>(shapeB);
		dimA = cache->getDimA();
		dimB = cache->getDimB();
		csA->hknpConvexShape::convertVertexIdsToVertices( cache->getVertexIdsA(), dimA, verticesOfAinA );
		csB->hknpConvexShape::convertVertexIdsToVertices( cache->getVertexIdsB(), dimB, verticesOfBinB );
	}
};

/// GSK interface for hknpConvexBaseShape - hknpConvexBaseShape.
struct hknpConvexConvexShapeBaseInterface : public hkcdGskBase::ShapeInterface
{
	HK_FORCE_INLINE virtual void getSupportingVertices(
		const void* HK_RESTRICT shapeA, hkVector4Parameter directionA,
		const void* HK_RESTRICT shapeB, const hkTransform& aTb,
		hkcdVertex* HK_RESTRICT vertexAinAOut, hkcdVertex* HK_RESTRICT vertexBinBOut, hkVector4* vertexBinAout
		) const
	{
		const hknpShape* HK_RESTRICT csA = static_cast<const hknpShape*>(shapeA);
		const hknpShape* HK_RESTRICT csB = static_cast<const hknpShape*>(shapeB);
		//hknpConvexShapeUtil::getSupportingVertices( *csA, directionA, *csB, aTb, vertexAinAOut, vertexBinBOut, vertexBinAout );

		csA->getSupportingVertex(directionA, vertexAinAOut);
		hkVector4 negDir;	 negDir.setNeg<4>(directionA);
		// directionBMem is in memory because its being passed to the virtual function
		hkVector4 directionBMem; directionBMem._setRotatedInverseDir( aTb.getRotation(), negDir);
		csB->getSupportingVertex(directionBMem, vertexBinBOut);
		hkVector4 vertexB; vertexB._setTransformedPos(aTb, *vertexBinBOut );
		vertexBinAout->setXYZ_W( vertexB, *vertexBinBOut);
	}

	// Inline implementation
	HK_FORCE_INLINE void _unpackCacheInlinedImpl(
		const hkcdGsk::Cache* cache,
		const void* shapeA, const void* shapeB,
		hkPadSpu<int>& dimA, hkcdVertex* verticesOfAinA,
		hkPadSpu<int>& dimB, hkcdVertex* verticesOfBinB) const
	{
		const hknpShape* HK_RESTRICT csA = static_cast<const hknpShape*>(shapeA);
		const hknpShape* HK_RESTRICT csB = static_cast<const hknpShape*>(shapeB);
		dimA = cache->getDimA();
		dimB = cache->getDimB();
		csA->convertVertexIdsToVertices( cache->getVertexIdsA(), dimA, verticesOfAinA );
		csB->convertVertexIdsToVertices( cache->getVertexIdsB(), dimB, verticesOfBinB );
	}

	// Out of line implementation, for Ps3 GCC only
	void _unpackCacheImpl(
		const hkcdGsk::Cache* cache,
		const void* shapeA, const void* shapeB,
		hkPadSpu<int>& dimA, hkcdVertex* verticesOfAinA,
		hkPadSpu<int>& dimB, hkcdVertex* verticesOfBinB) const;

	HK_FORCE_INLINE void _unpackCache(const hkcdGsk::Cache* cache, const void* shapeA, const void* shapeB, hkPadSpu<int>& dimA, hkcdVertex* verticesOfAinA, hkPadSpu<int>& dimB, hkcdVertex* verticesOfBinB) const
	{
		// The Ps3 GCC compiler generates bad code when the code for this method is optimized
#if defined(HK_PLATFORM_PS3_PPU) && defined(HK_COMPILER_GCC)
		_unpackCacheImpl(cache, shapeA, shapeB, dimA, verticesOfAinA, dimB, verticesOfBinB);
#else
		_unpackCacheInlinedImpl(cache, shapeA, shapeB, dimA, verticesOfAinA, dimB, verticesOfBinB);
#endif
	}
};

// Trampoline function to allow us to make the actual GSK code out of line on SPU for PHYSICS only.
hkcdGsk::GetClosestPointStatus hknpConvexConvexShapeBaseInterfaceGetClosestPoint(
	const void* shapeA, const void* shapeB, const hkcdGsk::GetClosestPointInput& input, hkcdGsk::Cache* HK_RESTRICT cache,
	hkcdGsk::GetClosestPointOutput& output );

#include <Physics/Internal/Collide/Process/CvxHelper/hknpCvxCollisionProcessHelper.inl>



#endif	// HKNP_CVX_COLLISION_PROCESS_HELPER_H

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
