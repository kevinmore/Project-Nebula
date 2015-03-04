/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Common/Base/hkBase.h>
#include <Common/Base/Types/Geometry/hkGeometry.h>
//#include <hkmath/basetypes/hkMotionState.h>
//#include <hkmath/basetypes/hkStridedVertices.h>



#include <Physics2012/Collide/Shape/Deprecated/ConvexPieceMesh/hkpConvexPieceMeshShape.h>

#include <Physics2012/Collide/Shape/Convex/Triangle/hkpTriangleShape.h>
#include <Physics2012/Collide/Shape/Deprecated/ConvexPieceMesh/hkpConvexPieceShape.h>
#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/hkpMoppBvTreeShape.h>
#include <Physics2012/Collide/Shape/Convex/Sphere/hkpSphereShape.h>
#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/hkpMoppUtility.h>
#include <Physics2012/Collide/Shape/Deprecated/Mesh/hkpMeshShape.h>

#include <Physics2012/Collide/Shape/Query/hkpShapeRayCastInput.h>
#include <Physics2012/Collide/Shape/Query/hkpShapeRayCastOutput.h>

#include <Physics2012/Collide/Query/Collector/PointCollector/hkpClosestCdPointCollector.h>
#include <Physics2012/Collide/Agent/Util/LinearCast/hkpIterativeLinearCastAgent.h>
#include <Physics2012/Collide/Agent/ConvexAgent/Gjk/hkpGskConvexConvexAgent.h>

#include <Physics2012/Collide/Agent/CompoundAgent/BvTree/hkpMoppAgent.h>

#include <Physics2012/Collide/Util/hkpTriangleUtil.h>

//#define HK_DEBUG_CONVEXPIECE


hkpConvexPieceMeshShape::hkpConvexPieceMeshShape( const hkpShapeCollection* inputMesh, const hkpConvexPieceStreamData* convexPieceStream, hkReal radius )
:	hkpShapeCollection(HKCD_SHAPE_TYPE_FROM_CLASS(hkpConvexPieceMeshShape), COLLECTION_USER), m_convexPieceStream( convexPieceStream ), m_displayMesh( inputMesh ), m_radius( radius )
{
	HK_WARN(0x4d5b8621,"Use of hkpConvexPieceMeshShape is deprecated. Please use an alternate shape.");

	HK_ASSERT(0x4d5b8622, m_displayMesh && m_convexPieceStream);
	
	m_displayMesh->addReference();
	m_convexPieceStream->addReference();
}

#if !defined(HK_PLATFORM_SPU)

hkpConvexPieceMeshShape::hkpConvexPieceMeshShape( hkFinishLoadedObjectFlag flag )
:	hkpShapeCollection( flag )
{
	if( flag.m_finishing )
	{
		setType(HKCD_SHAPE_TYPE_FROM_CLASS(hkpConvexPieceMeshShape));
	}
}

#endif

hkpConvexPieceMeshShape::~hkpConvexPieceMeshShape()
{
	m_displayMesh->removeReference();
	m_convexPieceStream->removeReference();
}


hkpShapeKey hkpConvexPieceMeshShape::getFirstKey() const
{
	return 0;
}


//
//	These functions can be used when the stream data is being used instead
//	of the arrays to represent the convex pieces.
//


hkpShapeKey hkpConvexPieceMeshShape::getNextKey( hkpShapeKey oldKey ) const
{
	if ( static_cast<int>(oldKey + 1) < ( m_convexPieceStream->m_convexPieceOffsets.getSize() + m_convexPieceStream->m_convexPieceSingleTriangles.getSize() ) )
	{
		return oldKey + 1;
	}
	else
	{
		return HK_INVALID_SHAPE_KEY;
	}
}

const hkpShape* hkpConvexPieceMeshShape::getChildShape( hkpShapeKey key, hkpShapeBuffer& buffer ) const
{
	HK_WARN(0x4d5b8621,"Use of hkpConvexPieceMeshShape is deprecated. Please use an alternate shape.");


	// We are storing the children in two arrays - one for which the convex piece is a single triangle, and
	// one for convex pieces that have multiple triangles.

	HK_ASSERT( 0x10a3351f, (hkReal)key < ( m_convexPieceStream->m_convexPieceOffsets.getSize() + m_convexPieceStream->m_convexPieceSingleTriangles.getSize() ));

	// We create a hkpConvexPieceShape from the convexPieces array and return that.
	// Check that we have enough space for a triangulatedConvexShape ( and all it's vertices! ).

	HK_ASSERT(0x73f97fa7,  sizeof( hkpConvexPieceShape ) <= HK_SHAPE_BUFFER_SIZE );

	hkpConvexPieceShape *triangulatedConvexShape = new( buffer ) hkpConvexPieceShape( m_radius );
	triangulatedConvexShape->m_displayMesh = m_displayMesh;

	if ( key < (unsigned)m_convexPieceStream->m_convexPieceOffsets.getSize() )
	{
		// The offsets are set for every element except the first.
		const hkUint32* childShapeStream = &m_convexPieceStream->m_convexPieceStream.begin()[0];

		childShapeStream += m_convexPieceStream->m_convexPieceOffsets[ key ];

		triangulatedConvexShape->m_numDisplayShapeKeys = *childShapeStream;
		triangulatedConvexShape->m_displayShapeKeys = childShapeStream + 1;

		// create the array of vertices from the convex piece and verts used info.
		{
			char* afterConvexPiece = reinterpret_cast< char* >(buffer + sizeof( hkpConvexPieceShape ));
			hkVector4* newVertexBase = reinterpret_cast<hkVector4*>( HK_NEXT_MULTIPLE_OF( HK_REAL_ALIGNMENT, reinterpret_cast<hkUlong>(afterConvexPiece)) );

			triangulatedConvexShape->m_vertices = newVertexBase;
			triangulatedConvexShape->m_numVertices = 0;

			hkpShapeBuffer buffer2;

			for ( int i = 0 ; i < triangulatedConvexShape->m_numDisplayShapeKeys ; i++ )
			{
				hkpShapeKey triangleKey = triangulatedConvexShape->m_displayShapeKeys[ i ];
				const hkpShape* shape = m_displayMesh->getChildShape( triangleKey, buffer2 );

				HK_ASSERT( 0x3789877b, shape->getType() == hkcdShapeType::TRIANGLE );

				const hkpTriangleShape& triangleShape = *( static_cast< const hkpTriangleShape* >( shape ));

				HK_ASSERT(0x75d34f0f, hkpTriangleUtil::isDegenerate( triangleShape.getVertex<0>(), triangleShape.getVertex<1>(), triangleShape.getVertex<2>(), 1e-7f ) == false );

				for ( int j = 0 ; j < 3 ; j++ )
				{
					if ( vertexIsSet(childShapeStream, (i*3+j)) )
					{
						(*newVertexBase) = triangleShape.getVertex(j);
						++newVertexBase;
						triangulatedConvexShape->m_numVertices++;
					}
				}
			}
		}
	}
	else
	{
		// There is only one triangle in this childShape.

		triangulatedConvexShape->m_numDisplayShapeKeys = 1;
		triangulatedConvexShape->m_displayShapeKeys = &(m_convexPieceStream->m_convexPieceSingleTriangles[ key - m_convexPieceStream->m_convexPieceOffsets.getSize() ]);

		char* afterConvexPiece = reinterpret_cast< char* >(buffer + sizeof( hkpConvexPieceShape ));
		hkVector4* newVertexBase = reinterpret_cast<hkVector4*>( HK_NEXT_MULTIPLE_OF( HK_REAL_ALIGNMENT, reinterpret_cast<hkUlong>(afterConvexPiece)) );

		triangulatedConvexShape->m_vertices = newVertexBase;
		triangulatedConvexShape->m_numVertices = 3;

		hkpShapeKey triangleKey = triangulatedConvexShape->m_displayShapeKeys[ 0 ];

		hkpShapeBuffer buffer2;
		const hkpTriangleShape& triangleShape = *( static_cast< const hkpTriangleShape* >( m_displayMesh->getChildShape( triangleKey, buffer2 ) ));

		for ( int j = 0 ; j < 3 ; j++ )
		{
			*newVertexBase++ = triangleShape.getVertex(j);
		}
	}

	// final check to ensure that we haven't overrun the buffer!
	int totalSizeUsed = sizeof( hkpConvexPieceShape ) + ( sizeof( hkVector4 ) * triangulatedConvexShape->m_numVertices );
	if ( totalSizeUsed > HK_SHAPE_BUFFER_SIZE )
	{
		#ifdef HK_DEBUG_SIMULATIONMESH
		printf("\n\n\nWARNING!!! The total size used (%d) is greater than HK_GET_SHAPE_BUFFER_SIZE (%f)", totalSizeUsed, HK_GET_SHAPE_BUFFER_SIZE );
		#endif
	}
	HK_ASSERT( 0x5592f44d, totalSizeUsed <= HK_SHAPE_BUFFER_SIZE );

	return triangulatedConvexShape;
}


#define HK_MAX_BITS		(sizeof(hkUint32)*8-1)	// 31
const hkBool hkpConvexPieceMeshShape::vertexIsSet( const hkUint32* stream, int key ) const
{
	// Find the bitstream - it's at the end of the list of vertices.
	const hkUint32* bitStream = stream + *stream + 1;

	int bitValue = *(bitStream + ( key/HK_MAX_BITS )) & (1 << ( key % HK_MAX_BITS ));
	return ( bitValue != 0 );
}


hkUint32 hkpConvexPieceMeshShape::getCollisionFilterInfo( hkpShapeKey key ) const
{
	hkpShapeKey firstTriangleKey;

	HK_ASSERT( 0x3ea728a6, (hkReal)key < ( m_convexPieceStream->m_convexPieceOffsets.getSize() + m_convexPieceStream->m_convexPieceSingleTriangles.getSize() ));

	if ( key < (unsigned)m_convexPieceStream->m_convexPieceOffsets.getSize() )
	{
		// The offsets are set for every element except the first.
		const hkUint32* childShapeStream = &m_convexPieceStream->m_convexPieceStream.begin()[0];
		childShapeStream += m_convexPieceStream->m_convexPieceOffsets[ key ];

		firstTriangleKey = childShapeStream[1];
	}
	else
	{
		firstTriangleKey = m_convexPieceStream->m_convexPieceSingleTriangles[ key - m_convexPieceStream->m_convexPieceOffsets.getSize() ];
	}

	return m_displayMesh->getCollisionFilterInfo( firstTriangleKey );
}


void hkpConvexPieceMeshShape::getAabb(const hkTransform& localToWorld, hkReal tolerance, hkAabb& out) const
{
	HK_WARN(0x4d5b8621,"Use of hkpConvexPieceMeshShape is deprecated. Please use an alternate shape.");



	// warning: not all vertices might be used, so it is not enough to go through vertexarray !
	// potential optimization (same for hkpMeshShape): speedup by lazy evaluation and storing the cached version, having a modified flag

	out.m_min = hkVector4::getConstant<HK_QUADREAL_MAX>();
	out.m_max = hkVector4::getConstant<HK_QUADREAL_MINUS_MAX>();

	hkpShapeKey convexShapeKey = getFirstKey();

	hkpShapeBuffer buffer;
	hkAabb childAabb;

	while ( convexShapeKey != HK_INVALID_SHAPE_KEY )
	{
		getChildShape( convexShapeKey, buffer )->getAabb( localToWorld, tolerance, childAabb );

		out.m_min.setMin( out.m_min, childAabb.m_min );
		out.m_max.setMax( out.m_max, childAabb.m_max );

		convexShapeKey = getNextKey( convexShapeKey );
	}

	// the tolerance is included in the child's AABB.
}

//
// Get some statistics on the number of triangles and convex pieces in the mesh.
// The statsOut should be zero'd the first time it is used.
//

void hkpConvexPieceMeshShape::getStats( hkpConvexPieceMeshShape::Stats& statsOut )
{
	statsOut.m_numTriangles += m_convexPieceStream->m_convexPieceSingleTriangles.getSize();
	statsOut.m_numConvexPieces += m_convexPieceStream->m_convexPieceSingleTriangles.getSize();

	for ( int i = 0 ; i < m_convexPieceStream->m_convexPieceOffsets.getSize() ; i++)
	{
		int triSizei =  m_convexPieceStream->m_convexPieceStream[m_convexPieceStream->m_convexPieceOffsets[i]];

		statsOut.m_numTriangles += triSizei;
		statsOut.m_maxTrianglesPerConvexPiece = hkMath::max2( statsOut.m_maxTrianglesPerConvexPiece, triSizei );
	}

	statsOut.m_numConvexPieces += m_convexPieceStream->m_convexPieceOffsets.getSize();
	statsOut.m_avgNumTriangles = (hkReal)statsOut.m_numTriangles / statsOut.m_numConvexPieces;
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
