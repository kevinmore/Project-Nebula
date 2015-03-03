/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Physics2012/Collide/Shape/Deprecated/ConvexPieceMesh/hkpConvexPieceShape.h>
#include <Physics2012/Collide/Shape/Query/hkpShapeRayCastInput.h>
#include <Physics2012/Collide/Shape/hkpShapeType.h>

#include <Physics2012/Collide/Shape/Query/hkpRayShapeCollectionFilter.h>

hkpConvexPieceShape::hkpConvexPieceShape( hkReal radius )
: hkpConvexShape(HKCD_SHAPE_TYPE_FROM_CLASS(hkpConvexPieceShape), radius )
{
	HK_WARN(0x4d5b8621,"Use of hkpConvexPieceShape is deprecated. Please use an alternate shape.");
}

void hkpConvexPieceShape::getFirstVertex(hkVector4& v) const
{
	HK_ASSERT(0x2a5e3059, m_numVertices > 0 );
	v = m_vertices[0];
}

void hkpConvexPieceShape::getAabb(const hkTransform& localToWorld, hkReal tolerance, hkAabb& out) const
{
	out.m_min.setXYZ(hkSimdReal_Max);
	out.m_max.setXYZ(hkSimdReal_MinusMax);

	for ( int i = 0 ; i < m_numVertices ; i++ )
	{
		hkVector4 vWorld; vWorld._setTransformedPos( localToWorld, m_vertices[ i ] );
		out.m_min.setMin( out.m_min, vWorld );
		out.m_max.setMax( out.m_max, vWorld );
	}

	hkVector4 tol4; tol4.setZero(); tol4.setXYZ( tolerance + m_radius);
	out.m_min.sub( tol4 );
	out.m_max.add( tol4 );
}


int hkpConvexPieceShape::getNumCollisionSpheres(  ) const
{
	return m_numVertices;
}

const hkSphere* hkpConvexPieceShape::getCollisionSpheres(hkSphere* sphereBuffer) const
{
	hkSphere* s = sphereBuffer;

	for ( int i = 0 ; i < m_numVertices ; i++ )
	{
		s->setPositionAndRadius(m_vertices[ i ], hkSimdReal::fromFloat(m_radius));
		s++;
	}

	return sphereBuffer;
}


void hkpConvexPieceShape::getSupportingVertex( hkVector4Parameter dir, hkcdVertex& supportingVertex ) const
{
	hkSimdReal maxDot = hkSimdReal_MinusMax;
	int vertexId = 0;

	hkpShapeBuffer triangleBuffer;
	for ( int i = 0; i < m_numDisplayShapeKeys; i++)
	{
		hkcdVertex support;

		const hkpTriangleShape* triangle = static_cast< const hkpTriangleShape*>( m_displayMesh->getChildShape(m_displayShapeKeys[i], triangleBuffer) );
		triangle->getSupportingVertex( dir, support );
		const hkSimdReal dot = support.dot<3>( dir );
		if ( dot > maxDot )
		{
			maxDot = dot;
			supportingVertex = support;
			vertexId = i*3 + (support.getId() / hkSizeOf(hkVector4));
		}
	}
	supportingVertex.setInt24W( vertexId );
}

void hkpConvexPieceShape::convertVertexIdsToVertices( const hkpVertexId* ids, int numIds, hkcdVertex* verticesOut) const
{
	hkpShapeBuffer triangleBuffer;
	for (int i = numIds-1; i>=0; i--)
	{
		int vertexId = ids[0];
		hkVector4& v = verticesOut[0];

		HK_ASSERT( 0xf03df445, vertexId < m_numDisplayShapeKeys*3 );

		const hkpTriangleShape* triangle = static_cast< const hkpTriangleShape*>( m_displayMesh->getChildShape(m_displayShapeKeys[vertexId/3], triangleBuffer) );
		v = triangle->getVertex( vertexId%3 );
		v.setInt24W( vertexId );
		verticesOut++;
		ids++;
	}
}

hkBool hkpConvexPieceShape::castRay(const hkpShapeRayCastInput& input,hkpShapeRayCastOutput& results) const
{
	HK_TIMER_BEGIN("rcConvxPiece", HK_NULL);
	if( m_numDisplayShapeKeys == 0 )
	{
		HK_WARN(0x530ccd4f, "You are trying to raycast against a triangulated convex shape with no plane equations. Raycasting will always return no hit in this case.");
	}

	hkpShapeBuffer buffer;
	int closestKey = HK_INVALID_SHAPE_KEY;
	results.changeLevel(1);

	for ( int i = 0 ; i < m_numDisplayShapeKeys ;i++ )
	{
		// only raycast against child Shapes that have collisions enabled.

		if ( input.m_rayShapeCollectionFilter )
		{
			if ( !input.m_rayShapeCollectionFilter->isCollisionEnabled( input, *m_displayMesh, m_displayShapeKeys[i] ) )
			{
				// ignore this childShape
				continue;
			}
		}
		
		const hkpShape* childShape = m_displayMesh->getChildShape( m_displayShapeKeys[i], buffer );

		// Return the closest hit
		if ( childShape->castRay( input, results ))
		{
			closestKey = i;
		}
	}
	results.changeLevel(-1);
	if( hkpShapeKey(closestKey) != HK_INVALID_SHAPE_KEY )
	{
		results.setKey(closestKey);
	}
	HK_TIMER_END();
	return ( closestKey != -1 );
}

void hkpConvexPieceShape::castRayWithCollector( const hkpShapeRayCastInput& input, const hkpCdBody& cdBody, hkpRayHitCollector& collector ) const
{
	HK_TIMER_BEGIN("rcConvxPiece", HK_NULL);
	if( m_numDisplayShapeKeys == 0 )
	{
		HK_WARN(0x530ccd4f, "You are trying to raycast against a triangulated convex shape with no plane equations. Raycasting will always return no hit in this case.");
	}

	hkpShapeBuffer buffer;

	for ( int i = 0 ; i < m_numDisplayShapeKeys ;i++ )
	{
		// only raycast against child Shapes that have collisions enabled.

		if ( input.m_rayShapeCollectionFilter )
		{
			if ( !input.m_rayShapeCollectionFilter->isCollisionEnabled( input, *m_displayMesh, m_displayShapeKeys[i] ) )
			{
				// ignore this childShape
				continue;
			}
		}
		
		const hkpShape* childShape = m_displayMesh->getChildShape( m_displayShapeKeys[i], buffer );
		hkpCdBody body(&cdBody);
		body.setShape(childShape, i);
		childShape->castRayWithCollector( input, body, collector );
	}
	HK_TIMER_END();
}

#if !defined(HK_PLATFORM_SPU)

const hkpShapeContainer* hkpConvexPieceShape::getContainer() const
{
	return this;
}

#endif

// Don't return our shape keys, but an index into our shape keys.

hkpShapeKey hkpConvexPieceShape::getFirstKey() const
{
	return 0;
}

hkpShapeKey hkpConvexPieceShape::getNextKey( hkpShapeKey oldKey ) const
{
	int newKey = oldKey + 1;
	return (newKey < m_numDisplayShapeKeys)
		? newKey
		: HK_INVALID_SHAPE_KEY;
}

const hkpShape* hkpConvexPieceShape::getChildShape( hkpShapeKey key, hkpShapeBuffer& buffer ) const
{
	HK_ASSERT(0x27c60c49, hkUlong(key) < hkUlong(m_numDisplayShapeKeys) );
	return m_displayMesh->getChildShape( m_displayShapeKeys[key], buffer );
}

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
