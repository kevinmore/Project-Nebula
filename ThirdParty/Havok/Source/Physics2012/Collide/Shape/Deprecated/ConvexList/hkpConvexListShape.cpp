/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Shape/Deprecated/ConvexList/hkpConvexListShape.h>
#include <Physics2012/Collide/Shape/Query/hkpShapeRayCastInput.h>
#include <Physics2012/Collide/Shape/Query/hkpRayShapeCollectionFilter.h>
#include <Common/Base/Types/Geometry/Aabb/hkAabbUtil.h>

hkpConvexListShape::hkpConvexListShape(const hkpConvexShape*const* shapeArray, int numShapes)
:	hkpConvexShape( HKCD_SHAPE_TYPE_FROM_CLASS(hkpConvexListShape), 0 ) // We set radius from the first shape of the shape Array - see below
// [HVK-2338] const ptr to const ptrs, so that I/F match. Should not need the reinterpret_cast but VC doesn't like it otherwise.
{
	m_minDistanceToUseConvexHullForGetClosestPoints = 1.0f;

	#ifdef HK_DEBUG
		// This might be triggered when children shapes have been shrunk as the shrinking process may set a convex radius smaller than the requested one.
		// Not shrinking children shape should fix it.
		const hkReal radius = shapeArray[0]->getRadius();
		for (int i =1; i < numShapes; i++ )
		{
			HK_ASSERT2( 0xf032da3a, hkMath::equal( radius, shapeArray[i]->getRadius(), .1f ), "All child shapes of a hkpConvexListShape must have identical radius" );
			if ( !hkMath::equal( radius, shapeArray[i]->getRadius(), .01f ) )
			{
				HK_WARN( 0xf032da3a, "All child shapes of a hkpConvexListShape must have identical radius" );
			}
		}
	#endif

	HK_WARN( 0x17969f14, "Use of hkpConvexListShape is deprecated. Please use an alternate shape." );

	setShapesAndRadius( shapeArray, numShapes );
	setUseCachedAabb( true );
}

#if !defined(HK_PLATFORM_SPU)

hkpConvexListShape::hkpConvexListShape( class hkFinishLoadedObjectFlag flag )
:	hkpConvexShape(flag)
,	m_childShapes(flag)
{	
	setType(HKCD_SHAPE_TYPE_FROM_CLASS(hkpConvexListShape));
}

#endif

hkpConvexListShape::~hkpConvexListShape()
{
	for (int i = 0; i < m_childShapes.getSize(); i++)
	{
		m_childShapes[i]->removeReference();
	}
}




void hkpConvexListShape::getSupportingVertex( hkVector4Parameter dir, hkcdVertex& supportingVertexOut ) const
{
	hkSimdReal maxDot = hkSimdReal_MinusMax;
	int subShape = 0;
#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
	hkIntVector counter; counter.setZero();
	const hkIntVector one = hkIntVector::getConstant<HK_QUADINT_1>();
	hkIntVector subIdx; subIdx.setZero();
#endif

	for ( int i = 0; i < m_childShapes.getSize(); i++)
	{
		hkcdVertex support;
		const hkpConvexShape* shape = static_cast<const hkpConvexShape*>( m_childShapes[i] );
		shape->getSupportingVertex( dir, support );
		const hkSimdReal dot = support.dot<3>( dir );
#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
		hkVector4Comparison gt = dot.greater(maxDot);
		maxDot.setSelect(gt, dot, maxDot);
		supportingVertexOut.setSelect(gt, support, supportingVertexOut);
		subIdx.setSelect(gt, counter, subIdx);
		counter.setAddS32(counter, one);
#else
		if ( dot > maxDot )
		{
			maxDot = dot;
			supportingVertexOut = support;
			subShape = i;
		}
#endif
	}
	int id = supportingVertexOut.getId();
	HK_ASSERT2( 0xf0ad12f4, id < 256, "The convex list shape can only use child shapes with vertex ids < 256" );
#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
	subIdx.setShiftLeft32<8>(subIdx);
	subIdx.store<1, HK_IO_NATIVE_ALIGNED>((hkUint32*)&subShape);
#else
	subShape = subShape<<8;
#endif
	id += subShape;
	supportingVertexOut.setInt24W( id );
}

void hkpConvexListShape::convertVertexIdsToVertices( const hkpVertexId* ids, int numIds, hkcdVertex* verticesOut) const
{
	for (int i = 0; i < numIds; i++)
	{
		hkpVertexId id = ids[0];
		int subShape = id>>8;
		id &= 0xff;
		const hkpConvexShape* shape = static_cast<const hkpConvexShape*>( m_childShapes[subShape] );

		shape->convertVertexIdsToVertices( &id, 1, verticesOut );

		// patch the id
		{
			int vid = verticesOut->getId();
			vid +=  (subShape<<8);
			verticesOut->setInt24W( vid );
		}

		ids++;
		verticesOut++;
	}
}

void hkpConvexListShape::getFirstVertex(hkVector4& v) const
{
	const hkpConvexShape* shape = static_cast<const hkpConvexShape*>( m_childShapes[0] );
	shape->getFirstVertex( v );
}


int hkpConvexListShape::getNumCollisionSpheres(  ) const
{
	int numSpheres = 0;
	for (int i = 0; i < m_childShapes.getSize(); i++)
	{
		const hkpConvexShape* shape = static_cast<const hkpConvexShape*>( m_childShapes[i] );
		numSpheres += shape->getNumCollisionSpheres();
	}
	return numSpheres;
}


const hkSphere* hkpConvexListShape::getCollisionSpheres(hkSphere* sphereBuffer) const
{
	hkSphere* p = sphereBuffer;
	for (int i = 0; i < m_childShapes.getSize(); i++)
	{
		const hkpConvexShape* shape = static_cast<const hkpConvexShape*>( m_childShapes[i] );

		shape->getCollisionSpheres( p );

		int numSpheres = shape->getNumCollisionSpheres();
		p += numSpheres;
	}
	return sphereBuffer;
}


void hkpConvexListShape::setShapesAndRadius( const hkpConvexShape*const* shapeArray, int numShapes )
{
	HK_ASSERT2(0x282822c7,  m_childShapes.getSize()==0, "You can only call setShapes once during construction.");
	HK_ASSERT2(0x221e5b17,  numShapes, "You cannot create a hkpConvexListShape with no child shapes" );

	m_childShapes.setSize(numShapes);
	m_radius = shapeArray[0]->getRadius();

	for (int i = 0; i < numShapes; i++)
	{
		HK_ASSERT2( 0xfeaf9625, shapeArray[i] != HK_NULL, "You cannot create a hkpConvexListShape with a shapeArray containing NULL pointers" );
		HK_ASSERT2( 0xa456bdbd, hkMath::equal(m_radius, shapeArray[i]->getRadius()), "You cannot create a hkpConvexListShape with child shapes of different radii");

		m_childShapes[i] = shapeArray[i];
		shapeArray[i]->addReference();
	}
}

void hkpConvexListShape::setUseCachedAabb( bool useCachedAabb )
{
	m_useCachedAabb = useCachedAabb;
	if (useCachedAabb)
	{
		hkAabb aabb;
		aabb.m_min = hkVector4::getConstant<HK_QUADREAL_MAX>();
		aabb.m_max = hkVector4::getConstant<HK_QUADREAL_MINUS_MAX>();

		for (int i = 0; i < m_childShapes.getSize(); i++)
		{
			hkAabb localAabb;
			m_childShapes[i]->getAabb( hkTransform::getIdentity(),0, localAabb );

			aabb.m_min.setMin( aabb.m_min, localAabb.m_min );
			aabb.m_max.setMax( aabb.m_max, localAabb.m_max );
		}

		aabb.getCenter( m_aabbCenter );
		aabb.getHalfExtents( m_aabbHalfExtents );
	}
}

bool hkpConvexListShape::getUseCachedAabb()
{
	return m_useCachedAabb;
}

void hkpConvexListShape::getAabb(const hkTransform& localToWorld, hkReal tolerance, hkAabb& out) const
{
	if (m_useCachedAabb)
	{
		hkAabbUtil::calcAabb( localToWorld, m_aabbHalfExtents, m_aabbCenter, hkSimdReal::fromFloat(tolerance + m_radius),  out );
	}
	else
	{
		m_childShapes[0]->getAabb( localToWorld, tolerance, out );

		hkAabb t;
		for (int i = 1; i < m_childShapes.getSize(); i++)
		{
			m_childShapes[i]->getAabb( localToWorld, tolerance, t );
			out.m_min.setMin( out.m_min, t.m_min );
			out.m_max.setMax( out.m_max, t.m_max );
		}
	}
}


int hkpConvexListShape::getNumChildShapes() const
{
	return m_childShapes.getSize();
}


hkpShapeKey hkpConvexListShape::getFirstKey() const
{
	return 0;
}

hkpShapeKey hkpConvexListShape::getNextKey( hkpShapeKey oldKey ) const
{
	if ( static_cast<int>(oldKey + 1) < m_childShapes.getSize() )
	{
		return oldKey + 1;
	}
	else
	{
		return HK_INVALID_SHAPE_KEY;
	}
}


const hkpShape* hkpConvexListShape::getChildShape( HKP_SHAPE_VIRTUAL_THIS hkpShapeKey key, hkpShapeBuffer& buffer ) HKP_SHAPE_VIRTUAL_CONST
{
	const hkpConvexListShape* thisObj = static_cast<const hkpConvexListShape*>(HK_GET_THIS_PTR);
	return thisObj->m_childShapes[ key ];
}

hkUint32 hkpConvexListShape::getCollisionFilterInfo( hkpShapeKey key ) const
{
	HK_WARN_ONCE(0xeaab8764, "Collision filtering does not work for hkConvexListShapes" );
	return 0;
}

//
// TODO - implement these efficiently, using MOPP
//

hkBool hkpConvexListShape::castRay(const hkpShapeRayCastInput& input, hkpShapeRayCastOutput& results) const
{
	// Note there is no collision filtering with convex list shapes

	HK_TIME_CODE_BLOCK("rcCxList",HK_NULL);

	hkpShapeBuffer shapeBuffer;
	results.changeLevel(1);
	hkpShapeKey bestKey = HK_INVALID_SHAPE_KEY;

	for (hkpShapeKey key = getFirstKey(); key != HK_INVALID_SHAPE_KEY; key = getNextKey( key ) )
	{
		const hkpShape* childShape = getChildShape( key, shapeBuffer );
		if ( childShape->castRay( input, results ) )
		{
			bestKey = key;
		}
	}

	results.changeLevel(-1);
	if( bestKey != HK_INVALID_SHAPE_KEY )
	{
		results.setKey(bestKey);
	}

	return bestKey != HK_INVALID_SHAPE_KEY;
}


void hkpConvexListShape::castRayWithCollector( const hkpShapeRayCastInput& input, const hkpCdBody& cdBody, hkpRayHitCollector& collector ) const
{
	HK_TIME_CODE_BLOCK("rcShpCollect",HK_NULL);
	HK_ASSERT2(0x5c50f827,  cdBody.getShape() == this, "inconsistent cdBody, shapePointer is wrong" );

	hkpShapeBuffer shapeBuffer;

	for (hkpShapeKey key = getFirstKey(); key != HK_INVALID_SHAPE_KEY; key = getNextKey( key ) )
	{
		const hkpShape* childShape = getChildShape( key, shapeBuffer );
		hkpCdBody childBody( &cdBody );
		childBody.setShape( childShape, key );
		childShape->castRayWithCollector( input, childBody, collector );
	}
}

#if !defined(HK_PLATFORM_SPU)

const hkpShapeContainer* hkpConvexListShape::getContainer() const
{
	return this;
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
