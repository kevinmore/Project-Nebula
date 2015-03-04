/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Shape/Convex/ConvexTransform/hkpConvexTransformShape.h>
#include <Common/Base/Types/Geometry/Aabb/hkAabbUtil.h>
#include <Physics2012/Collide/Shape/Query/hkpShapeRayCastInput.h>
#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesShape.h>
#include <Physics2012/Collide/Shape/Query/hkpRayHitCollector.h>

#if !defined(HK_PLATFORM_SPU)

hkpConvexTransformShape::hkpConvexTransformShape( const hkpConvexShape* childShape, const hkTransform& transform, hkpShapeContainer::ReferencePolicy ref )
:	hkpConvexTransformShapeBase(HKCD_SHAPE_TYPE_FROM_CLASS(hkpConvexTransformShape), childShape->getRadius(), childShape, ref )
{
	HK_ASSERT2( 0x6acf0520, childShape != HK_NULL, "Child shape cannot be NULL" );
	setTransform( transform );
	m_userData = childShape->getUserData();
}

hkpConvexTransformShape::hkpConvexTransformShape( const hkpConvexShape* childShape, const hkQsTransform& transform, hkpShapeContainer::ReferencePolicy ref )
:	hkpConvexTransformShapeBase(HKCD_SHAPE_TYPE_FROM_CLASS(hkpConvexTransformShape), childShape->getRadius(), childShape, ref )
{
	HK_ASSERT2( 0x6acf0520, childShape != HK_NULL, "Child shape cannot be NULL" );
	setTransform( transform );
	m_userData = childShape->getUserData();
}

//
//	Serialization constructor

hkpConvexTransformShape::hkpConvexTransformShape( class hkFinishLoadedObjectFlag flag )
:	hkpConvexTransformShapeBase(flag)
{
	setType(HKCD_SHAPE_TYPE_FROM_CLASS(hkpConvexTransformShape));
}

#endif

void hkpConvexTransformShape::getAabb(const hkTransform& localToWorld, hkReal tolerance, hkAabb& out) const
{
	

	// compute the scaled child shape AABB
	hkVector4 scaledAabbCenter;
	hkVector4 scaledAabbHalfExtents;
	{
		hkAabb childAabb;
		getChildShape()->getAabb( hkTransform::getIdentity(), 0.0f, childAabb );

		childAabb.getCenter( scaledAabbCenter );
		scaledAabbCenter.setMul( scaledAabbCenter, m_transform.getScale() );

		childAabb.getHalfExtents( scaledAabbHalfExtents ); 
		scaledAabbHalfExtents.setMul( scaledAabbHalfExtents, m_transform.getScale() );
	}

	// compute the non-scaled world space transform
	hkTransform worldTshape;
	{
		m_transform.copyToTransformNoScale( worldTshape );
		worldTshape.setMul( localToWorld, worldTshape );
	}

	// calculate the AABB
	hkAabbUtil::calcAabb( worldTshape, scaledAabbHalfExtents, scaledAabbCenter, hkSimdReal::fromFloat(tolerance), out );
}


hkBool hkpConvexTransformShape::castRay(const hkpShapeRayCastInput& input, hkpShapeRayCastOutput& results) const
{
	HK_TIMER_BEGIN("rcCxTransform", HK_NULL);

	hkVector4 invScale;
	invScale.setReciprocal( m_transform.getScale() );

	// copy and transform input data
	hkpShapeRayCastInput transformedInput = input;
	{
		// apply inverse transform, (RS)^-1 = S^-1 R^-1,
		// subtract translation, inverse rotation, then reciprocal scaling.
		

		transformedInput.m_from.sub( m_transform.getTranslation() );
		transformedInput.m_from._setRotatedInverseDir( m_transform.getRotation(), transformedInput.m_from );
		transformedInput.m_from.mul( invScale );

		transformedInput.m_to.sub( m_transform.getTranslation() );
		transformedInput.m_to._setRotatedInverseDir( m_transform.getRotation(), transformedInput.m_to );
		transformedInput.m_to.mul( invScale );
	}

	// perform ray cast on wrapped shape with the transformed ray
	results.changeLevel(1);
	const hkBool hit = getChildShape()->castRay( transformedInput, results );
	results.changeLevel(-1);

	if( hit )
	{
		// transform and normalize the normal
		// when transforming normals we use the inverse transpose of the transformation
		hkVector4 tmp; tmp.setMul( results.m_normal, invScale );
		tmp._setRotatedDir( m_transform.getRotation(), tmp );
		tmp.normalize<3>();
		results.m_normal = tmp;
		results.setKey(0);
	}

	HK_TIMER_END();
	return hit;
}

void hkpConvexTransformShape::castRayWithCollector(const hkpShapeRayCastInput& input, const hkpCdBody& cdBody, hkpRayHitCollector& collector) const
{
	HK_TIMER_BEGIN("rcCxTransform", HK_NULL);

	hkpShapeRayCastOutput rayOut;
	if ( castRay( input, rayOut ) )
	{
		hkpCdBody childBody( &cdBody );
		childBody.setShape( getChildShape(), 0 );
		rayOut.m_normal._setRotatedDir( cdBody.getTransform().getRotation(), rayOut.m_normal );
		collector.addRayHit( childBody, rayOut );
	}

	HK_TIMER_END();
}

void hkpConvexTransformShape::getSupportingVertex(hkVector4Parameter direction, hkcdVertex& supportingVertexOut) const
{
	// apply transposed transform, (RS)^T = S^T R^T,
	// inverse rotation, then scaling (unchanged by transpose).
	// note that translation must not be applied when transforming support directions
	hkVector4 localDir; localDir._setRotatedInverseDir(m_transform.getRotation(), direction);
	localDir.mul(m_transform.getScale());
	hkcdVertex localVertex; getChildShape()->getSupportingVertex(localDir, localVertex);
	transformVertex(localVertex, &supportingVertexOut);
	supportingVertexOut.setW(localVertex);
}

void hkpConvexTransformShape::convertVertexIdsToVertices(const hkpVertexId* ids, int numIds, hkcdVertex* verticesOut) const
{
	// Get the vertices from the child/wrapped shape	
	getChildShape()->convertVertexIdsToVertices(ids, numIds, verticesOut);		

	// Transform them, preserving the W component (contains the vertex ids)
	for (int i = 0; i < numIds; ++i)
	{				
		const hkVector4& localVertex = verticesOut[i];
		hkVector4 transformedVertex; transformVertex(localVertex, &transformedVertex);
		verticesOut[i].setXYZ(transformedVertex);
	}
}

void hkpConvexTransformShape::getCentre(hkVector4& centreOut) const
{
	hkVector4 centre; getChildShape()->getCentre(centre);
	transformVertex(centre, &centreOut);
}

void hkpConvexTransformShape::setTransform( const hkQsTransform& transform )
{	
	m_transform = transform;

	// Initializing w's to zero. (used to store aabb center position)
	m_transform.m_translation.zeroComponent<3>();
	m_transform.m_scale.zeroComponent<3>();

	// If there is no scale we can skip the extra scale computation	
	const hkVector4& scale = transform.getScale();
	if (scale.allEqual<3>(hkVector4::getConstant<HK_QUADREAL_1>(), hkSimdReal_Eps))
	{
		m_extraScale.setZero();
		return;
	}
	HK_ON_DEBUG( hkVector4 scaleX; scaleX.setAll(scale.getComponent<0>()); hkcdShape::ShapeType childType = getChildShape()->getType(); );
	HK_ASSERT2(0x7faacf3e, scale.allExactlyEqual<3>(scaleX) || childType == hkcdShapeType::BOX || childType == hkcdShapeType::CONVEX_VERTICES, "Non-uniform scale is not supported for this child shape type");
	
	// Scale the convex radius directly in shapes that are defined by it (spheres and capsules)
	const hkpConvexShape* childShape = getChildShape();
	hkVector4 scaleAbs; scaleAbs.setAbs(scale);
	if ( childShape->getNumCollisionSpheres() < 3 )
	{	
		m_radius = childShape->getRadius() * scaleAbs(0);
		m_extraScale.setZero();
		return;
	}

	// Obtain half extents and center of the child's aabb without the convex radius
	hkVector4 halfExtents;
	hkVector4 center;
	{		
		hkAabb aabb; childShape->getAabb(hkTransform::getIdentity(), 0, aabb);
		aabb.expandBy(hkSimdReal::fromFloat(-childShape->getRadius()));
		aabb.getHalfExtents(halfExtents);
		aabb.getCenter(center);				
	}	

	// Calculate the maximum radius the scale allows for each component: scaleAbs * (halfExtent + childRadius)
	const hkVector4Comparison isScaleNegative = scale.signBitSet();
	hkVector4 childRadius; childRadius.setAll(m_radius);	
	hkVector4 maxRadius; maxRadius.setAdd(halfExtents, childRadius);
	maxRadius.mul(scaleAbs);

	// If the child radius is over the maximum for any component, use the maximum allowed for all components
	if (!childRadius.allLess<3>(maxRadius))
	{
		maxRadius.setHorizontalMin<3>(maxRadius);
		m_radius = maxRadius(0);
		HK_WARN_ONCE(0x1592e03a, "The convex radius has been reduced to fit the scaled shape");
	}
	else
	{
		maxRadius = childRadius;
	}

	// Compute the extra scale
	scaleAbs.mul(childRadius);
	scaleAbs.sub(maxRadius);
	m_extraScale.setDiv<HK_ACC_23_BIT, HK_DIV_SET_ZERO>(scaleAbs, halfExtents);
	m_extraScale.setFlipSign(m_extraScale, isScaleNegative);

	// Store aabb center in w components	
	m_transform.m_translation.setW(center.getComponent<0>());
	m_transform.m_scale.setW(center.getComponent<1>());
	m_extraScale.setW(center.getComponent<2>());				
}


#ifndef HK_PLATFORM_SPU

void hkpConvexTransformShape::getFirstVertex( hkVector4& v ) const
{
	hkVector4 localVertex;
	getChildShape()->getFirstVertex( localVertex );
	transformVertex(localVertex, &v);	
}

#endif //HK_PLATFORM_SPU

const hkSphere* hkpConvexTransformShape::getCollisionSpheres(hkSphere* sphereBuffer) const
{
	// get the spheres from the child/wrapped shape
	const hkSphere* spheres = getChildShape()->getCollisionSpheres( sphereBuffer );

	// transform them
	int numSpheres = getChildShape()->getNumCollisionSpheres();
	const hkVector4 *in = &spheres->getPositionAndRadius();
	hkVector4 *out = &sphereBuffer->getPositionAndRadius();
	const hkSimdReal radius = hkSimdReal::fromFloat(m_radius);
	for (int i = numSpheres - 1; i >= 0; --i)
	{
		hkVector4 center;
		transformVertex(in[i], &center);
		out[i].setXYZ_W(center, radius);
	}

	return sphereBuffer;
}

#if !defined(HK_PLATFORM_SPU)

const hkpShapeContainer* hkpConvexTransformShape::getContainer() const
{
	return &m_childShape;
}

int hkpConvexTransformShape::calcSizeForSpu( const CalcSizeForSpuInput& input, int spuBufferSizeLeft ) const
{
	// only cascades that will fit in total into one of the SPU's shape buffers are allowed to be uploaded onto SPU.

	int maxAvailableBufferSizeForChild = spuBufferSizeLeft - sizeof(*this);

	int childSize = m_childShape.getChild()->calcSizeForSpu( input, maxAvailableBufferSizeForChild );

	if ( childSize < 0 )
	{
		// Child shape will print a more detailed error message (with a reason).
		HK_WARN( 0xdbc05911, "hkpConvexTransformShape child (" << hkGetShapeTypeName(getChildShape()->getType()) << ") cannot be processed on SPU." );
		return -1;
	}

	if ( childSize > maxAvailableBufferSizeForChild )
	{
		// early out if cascade will not fit into SPU's shape buffer
		HK_WARN( 0xdbc05911, "hkpConvexTransformShape child (" << hkGetShapeTypeName(getChildShape()->getType()) << ") will not fit on SPU." );
		return -1;
	}

	// if child is consecutive in memory, set flag and return total size
	if ( hkUlong(m_childShape.getChild()) == hkUlong((this+1)) )
	{
		m_childShapeSizeForSpu = 0;
		return sizeof(*this) + childSize;
	}

	// the SPU will need this value to properly DMA the child shape in one go
	m_childShapeSizeForSpu = childSize;

	// if child is not consecutive in memory, restart size calculation with just us
	return sizeof(*this);
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
