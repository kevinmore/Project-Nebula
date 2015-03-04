/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Shape/Convex/ConvexTranslate/hkpConvexTranslateShape.h>
#include <Common/Base/Types/Geometry/Aabb/hkAabbUtil.h>
#include <Physics2012/Collide/Shape/Query/hkpShapeRayCastInput.h>

void hkpConvexTranslateShape::getAabb(const hkTransform& localToWorld, hkReal tolerance, hkAabb& out) const
{
	getChildShape()->getAabb( localToWorld, tolerance, out );

	hkVector4 trans; trans._setRotatedDir( localToWorld.getRotation(), m_translation );
	out.m_min.add( trans );
	out.m_max.add( trans );
}

#if !defined(HK_PLATFORM_SPU)

//
//	Serialization constructor

hkpConvexTranslateShape::hkpConvexTranslateShape( class hkFinishLoadedObjectFlag flag )
:	hkpConvexTransformShapeBase(flag)
{
	setType(HKCD_SHAPE_TYPE_FROM_CLASS(hkpConvexTranslateShape));
}

//
//	support for MOPP

hkReal hkpConvexTranslateShape::getMaximumProjection( const hkVector4& direction ) const
{
	hkReal localProjection = m_childShape->getMaximumProjection( direction );
	hkReal offset = direction.dot<3>( m_translation ).getReal();
	return localProjection + offset;
}


int hkpConvexTranslateShape::calcSizeForSpu(const CalcSizeForSpuInput& input, int spuBufferSizeLeft) const
{
	// only cascades that will fit in total into one of the spu's shape buffers are allowed to be uploaded onto spu.

	int maxAvailableBufferSizeForChild = spuBufferSizeLeft - sizeof(*this);

	int childSize = m_childShape.getChild()->calcSizeForSpu(input, maxAvailableBufferSizeForChild);
	if ( childSize < 0 )
	{
		// Child shape will print a more detailed error message (with a reason).
		HK_WARN(0xdbc05911, "hkpConvexTranslateShape child (" << hkGetShapeTypeName(getChildShape()->getType()) << ") cannot be processed on SPU.");
		return -1;
	}

	if ( childSize > maxAvailableBufferSizeForChild )
	{
		// early out if cascade will not fit into spu's shape buffer
		HK_WARN(0xdbc05911, "hkpConvexTranslateShape child (" << hkGetShapeTypeName(getChildShape()->getType()) << ") will not fit on SPU.");
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

hkBool hkpConvexTranslateShape::castRay(const hkpShapeRayCastInput& input, hkpShapeRayCastOutput& results) const
{
	HK_TIMER_BEGIN("rcConvTransl", HK_NULL);
	hkpShapeRayCastInput subInput = input;

	subInput.m_from.setSub( input.m_from, m_translation );
	subInput.m_to.setSub( input.m_to, m_translation );

	results.changeLevel(1);
	const hkBool hit = getChildShape()->castRay( subInput, results );
	results.changeLevel(-1);
	if( hit )
	{
		results.setKey(0);
	}
	HK_TIMER_END();
	return hit;
}

void hkpConvexTranslateShape::castRayWithCollector(const hkpShapeRayCastInput& input, const hkpCdBody& cdBody, hkpRayHitCollector& collector) const
{
	HK_ASSERT2(0x7f1735a0, cdBody.getShape() == this, "inconsistent cdBody, shapePointer is wrong" );

	hkpShapeRayCastInput subInput = input;
	subInput.m_from.setSub( input.m_from, m_translation );
	subInput.m_to.setSub( input.m_to, m_translation );

	hkTransform thisTransform; thisTransform.setIdentity(); thisTransform.setTranslation(m_translation);
	hkTransform t; t.setMul( cdBody.getTransform(), thisTransform);
	hkpCdBody childBody( &cdBody, &t);

	const hkpShape* childShape = getChildShape();
	childBody.setShape(childShape, 0);
	childShape->castRayWithCollector(subInput, childBody, collector );
}

void hkpConvexTranslateShape::getSupportingVertex(hkVector4Parameter direction, hkcdVertex& supportingVertexOut) const
{
	HK_ASSERT2( 0x4835a45e, m_translation(3) == 0.0f, "The w component of hkpConvexTranslateShape::m_translation must be zero." );
	getChildShape()->getSupportingVertex( direction, supportingVertexOut );

	// Get the .w component and restore it, as on PlayStation(R)3 adding 0 can change the value of .w
	const hkSimdReal wComp = supportingVertexOut.getComponent<3>();
	HK_ON_DEBUG( int id = supportingVertexOut.getId() );
	supportingVertexOut.add( m_translation );
	supportingVertexOut.setComponent<3>(wComp);
	HK_ASSERT2( 0xf019fe43, id == supportingVertexOut.getId(), "The supporting vertex ID was changed while applying the translation in hkpConvexTranslateShape::getSupportingVertex()." );
}

void hkpConvexTranslateShape::convertVertexIdsToVertices(const hkpVertexId* ids, int numIds, hkcdVertex* verticesOut) const
{
	getChildShape()->convertVertexIdsToVertices( ids, numIds, verticesOut );
	{
		for (int i = 0; i < numIds; i++)
		{
			hkVector4 v; v.setAdd(verticesOut[i], m_translation);
			verticesOut[i].setXYZ_W(v, verticesOut[i]);
		}
	}
}

void hkpConvexTranslateShape::getCentre(hkVector4& centreOut) const
{
	getChildShape()->getCentre( centreOut );
	centreOut.add(m_translation );
}

#ifndef HK_PLATFORM_SPU

void hkpConvexTranslateShape::getFirstVertex(hkVector4& v) const
{
	getChildShape()->getFirstVertex( v );
	v.add( m_translation );
}

#endif //HK_PLATFORM_SPU

const hkSphere* hkpConvexTranslateShape::getCollisionSpheres(hkSphere* sphereBuffer) const
{
	const hkSphere* spheres = getChildShape()->getCollisionSpheres( sphereBuffer );
	hkSphere* spheresOut = sphereBuffer;

	int numSpheres = getChildShape()->getNumCollisionSpheres( );
	{
		for (int i = 0; i < numSpheres; i++)
		{
			spheresOut->getPositionAndRadius().setAdd( spheres->getPositionAndRadius(), m_translation );
			spheres++;
			spheresOut++;
		}
	}
	return sphereBuffer;
}

#if !defined(HK_PLATFORM_SPU)

const hkpShapeContainer* hkpConvexTranslateShape::getContainer() const
{
	return &m_childShape;
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
