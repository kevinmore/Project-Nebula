/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Shape/Convex/hkpConvexShape.h>
#include <Physics2012/Collide/Shape/Query/hkpRayHitCollector.h>

#if defined(HK_PLATFORM_SPU)
#	include <Common/Base/Spu/Dma/Manager/hkSpuDmaManager.h>
#endif

#if (HK_POINTER_SIZE==4) && (HK_NATIVE_ALIGNMENT==16) && !defined(HK_REAL_IS_DOUBLE) && !defined(HK_COMPILER_HAS_INTRINSICS_NEON)
HK_COMPILE_TIME_ASSERT( sizeof(hkpConvexShape) == 20 );
#endif

hkReal hkConvexShapeDefaultRadius = 0.05f;

#if !defined(HK_PLATFORM_SPU)

//
//	Serialization constructor

hkpConvexShape::hkpConvexShape( hkFinishLoadedObjectFlag flag )
:	hkpSphereRepShape(flag)
{
	setType(HKCD_SHAPE_TYPE_FROM_CLASS(hkpConvexShape));
}

hkReal hkpConvexShape::getMaximumProjection(const hkVector4& direction) const
{
	hkcdVertex supportingVertex;
	getSupportingVertex(direction, supportingVertex);
	const hkSimdReal projection = supportingVertex.dot<3>(direction);
	const hkSimdReal radiusProjection = hkSimdReal::fromFloat(m_radius) * direction.length<3>();
	return (projection + radiusProjection).getReal();
}

#endif

void hkpConvexShape::castRayWithCollector(const hkpShapeRayCastInput& inputLocal, const hkpCdBody& cdBody, hkpRayHitCollector& collector) const
{
	
	//HK_ASSERT2(0x7f1735a0,  cdBody.getShape() == thisShape, "inconsistent cdBody, shapePointer is wrong" );

	hkpShapeRayCastOutput results;
	results.m_hitFraction = collector.m_earlyOutHitFraction;

	if ( castRay( inputLocal, results ) )
	{
		HK_ASSERT2(0x6ad83e81, results.m_shapeKeys[0] == HK_INVALID_SHAPE_KEY, "Non leaf convex shape needs to override castRayWithCollector");
		results.m_normal._setRotatedDir( cdBody.getTransform().getRotation(), results.m_normal );
		collector.addRayHit( cdBody, results );
	}
}

int hkpConvexShape::weldContactPoint(hkpVertexId* featurePoints, hkUint8& numFeaturePoints, hkVector4& contactPointWs, const hkTransform* thisObjTransform, const hkpConvexShape* collidingShape, const hkTransform* collidingTransform, hkVector4& separatingNormalInOut) const
{
	return WELD_RESULT_ACCEPT_CONTACT_POINT_UNMODIFIED;
}

void hkpConvexShape::getCentre(hkVector4& centreOut) const
{
	hkAabb aabb;
	getAabb( hkTransform::getIdentity(), 0, aabb );
	centreOut.setAdd(aabb.m_max, aabb.m_min);
	centreOut.mul(hkSimdReal_Inv2);
}

#if ! defined (HK_PLATFORM_SPU)

hkpConvexTransformShapeBase::hkpConvexTransformShapeBase( ShapeType type, hkReal radius, const hkpConvexShape* childShape, hkpShapeContainer::ReferencePolicy ref ) 
:	hkpConvexShape(type, radius)
,	m_childShape(childShape, ref) 
{}

#endif

#if defined(HK_PLATFORM_SPU)
void hkpConvexTransformShapeBase::getChildShapeFromPpu(int thisShapeSize) const
{
	// pointer to the memory right after this shape
	hkpShape* dstInhkpShapeBuffer = const_cast<hkpShape*>( hkAddByteOffsetConst(static_cast<const hkpShape*>(this), thisShapeSize) );

	// get child shape from main memory; put it right after this shape
	hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion(dstInhkpShapeBuffer, m_childShape.getChild(), m_childShapeSizeForSpu, hkSpuDmaManager::READ_COPY);
	HK_SPU_DMA_PERFORM_FINAL_CHECKS(m_childShape.getChild(), dstInhkpShapeBuffer, m_childShapeSizeForSpu);

	// flag this shape as locally available
	m_childShapeSizeForSpu = 0;
}
#endif

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
