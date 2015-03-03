/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Shape/Misc/Transform/hkpTransformShape.h>
#include <Common/Base/Types/Geometry/Aabb/hkAabbUtil.h>
#include <Physics2012/Collide/Shape/Query/hkpShapeRayCastInput.h>
#if defined(HK_PLATFORM_SPU)
#include <Common/Base/Spu/Dma/Manager/hkSpuDmaManager.h>
#endif

#if ! defined (HK_PLATFORM_SPU)

//
//	Serialization constructor

hkpTransformShape::hkpTransformShape( hkFinishLoadedObjectFlag flag )
:	hkpShape(flag)
,	m_childShape(flag)
{
	setType(HKCD_SHAPE_TYPE_FROM_CLASS(hkpTransformShape));
}

void hkpTransformShape::setTransform(const hkTransform& transform)
{
	m_transform = transform;
	m_rotation.set( m_transform.getRotation() );
}

hkpTransformShape::hkpTransformShape(const hkpShape* childShape, const hkTransform& transform)
:	hkpShape(HKCD_SHAPE_TYPE_FROM_CLASS(hkpTransformShape))
,	m_childShape( childShape )
{
	HK_ASSERT2(0x6acf0520, childShape != HK_NULL, "Child shape cannot be NULL");
	setTransform(transform);
}

#endif

void hkpTransformShape::getAabb(const hkTransform& localToWorld, hkReal tolerance, hkAabb& out) const
{
	hkTransform worldTshape; worldTshape.setMul ( localToWorld, m_transform );
	getChildShape()->getAabb( worldTshape, tolerance, out );
}

#if !defined (HK_PLATFORM_SPU)

hkReal hkpTransformShape::getMaximumProjection(const hkVector4& direction) const
{		
	// Transform the projection direction to the child shape local space and obtain the maximum projection
	hkVector4 localDir; localDir._setRotatedInverseDir(m_transform.getRotation(), direction);
	hkReal localProjection = getChildShape()->getMaximumProjection(localDir);

	// Compute the translation in the projection direction
	const hkReal offset = direction.dot<3>(m_transform.getTranslation()).getReal();

	return localProjection + offset;
}

#endif

hkBool hkpTransformShape::castRay(const hkpShapeRayCastInput& input, hkpShapeRayCastOutput& results) const
{
	HK_TIMER_BEGIN("rcTransform", HK_NULL);
	hkpShapeRayCastInput subInput = input;

	subInput.m_from._setTransformedInversePos( m_transform, input.m_from );
	subInput.m_to._setTransformedInversePos( m_transform, input.m_to );

	results.changeLevel(1);
	const hkBool hit = getChildShape()->castRay( subInput, results );
	results.changeLevel(-1);
	if (hit)
	{
		//transform hitnormal from 'childshape' into 'transformshapes' space
		const hkVector4 oldnormal = results.m_normal;
		results.m_normal._setRotatedDir( m_transform.getRotation(), oldnormal );
		results.setKey(0);
	}
	HK_TIMER_END();
	return hit;
}


void hkpTransformShape::castRayWithCollector(const hkpShapeRayCastInput& input, const hkpCdBody& cdBody, hkpRayHitCollector& collector ) const
{
	HK_TIMER_BEGIN("rcTransform", HK_NULL);
	hkpShapeRayCastInput subInput = input;

	subInput.m_from._setTransformedInversePos( m_transform, input.m_from );
	subInput.m_to._setTransformedInversePos( m_transform, input.m_to );

	hkTransform t; t.setMul( cdBody.getTransform(), m_transform);

	hkpCdBody body( &cdBody, &t);
	const hkpShape* childShape = getChildShape();
	body.setShape ( childShape, 0 );

	childShape->castRayWithCollector( subInput, body, collector );
	HK_TIMER_END();
}

#if !defined (HK_PLATFORM_SPU)

const hkpShapeContainer* hkpTransformShape::getContainer() const
{
	return &m_childShape;
}

int hkpTransformShape::calcSizeForSpu(const CalcSizeForSpuInput& input, int spuBufferSizeLeft) const
{
	// convex translate code is no good

	// only cascades that will fit in total into one of the spu's shape buffers are allowed to be uploaded onto spu.

	int maxAvailableBufferSizeForChild = spuBufferSizeLeft - sizeof(*this);

	int childSize = m_childShape.getChild()->calcSizeForSpu(input, maxAvailableBufferSizeForChild);
	if ( childSize < 0 )
	{
		// Child shape will print a more detailed error message (with a reason).
		HK_WARN(0xad23432a, "hkpTransformShape child (" << hkGetShapeTypeName(getChildShape()->getType()) << ") cannot be processed on SPU.");
		return -1;
	}

	if ( childSize > maxAvailableBufferSizeForChild )
	{
		// early out if cascade will not fit into spu's shape buffer
		HK_WARN(0xad23432a, "hkpTransformShape child (" << hkGetShapeTypeName(getChildShape()->getType()) << ") will not fit on SPU.");
		return -1;
	}

	// if child is consecutive in memory, set flag and return total size
	if ( hkUlong(m_childShape.getChild()) == hkUlong((this+1)) )
	{
		m_childShapeSize = 0;
		return sizeof(*this) + childSize;
	}

	// the spu will need this value to properly dma the child shape in one go
	m_childShapeSize = childSize;

	// if child is not consecutive in memory, restart size calculation with just us
	return sizeof(*this);
}

#else

void hkpTransformShape::getChildShapeFromPpu() const
{
	// pointer to the memory right after this hkConvexTranslate
	hkpShape* dstInhkpShapeBuffer = const_cast<hkpShape*>( reinterpret_cast<const hkpShape*>( this+1 ) );

	// get child shape from main memory; put it right after this hkConvexTranslate
	hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion(dstInhkpShapeBuffer, m_childShape.getChild(), m_childShapeSize, hkSpuDmaManager::READ_COPY);
	HK_SPU_DMA_PERFORM_FINAL_CHECKS(m_childShape.getChild(), dstInhkpShapeBuffer, m_childShapeSize);

	// flag this shape as locally available
	m_childShapeSize = 0;
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
