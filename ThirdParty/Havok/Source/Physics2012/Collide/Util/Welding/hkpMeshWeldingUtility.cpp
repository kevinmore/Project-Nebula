/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Util/Welding/hkpMeshWeldingUtility.h>
#include <Physics2012/Collide/Shape/Convex/Triangle/hkpTriangleShape.h>
#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/hkpMoppBvTreeShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/ExtendedMeshShape/hkpExtendedMeshShape.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>
#include <Physics2012/Collide/Util/hkpTriangleUtil.h>


struct hkEntry
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_COLLIDE, hkEntry);
	hkReal m_lo;
	hkReal m_hi;
	hkReal m_value;
};

hkBool operator<( const hkEntry& lhs, const hkEntry& rhs )
{
	return lhs.m_value < rhs.m_value;
}

hkBool operator!=( const hkEntry& lhs, const hkEntry& rhs )
{
	//return lhs.m_value != rhs.m_value;
	return !hkMath::equal( lhs.m_value, rhs.m_value, 1.0e-3f );
}



class hkHistogram
{
public:
	hkHistogram( bool weldOpenEdges )
	{
		m_freeAngle = weldOpenEdges ? 0.0f : HK_REAL_PI;
	}

	void addEntry( hkReal lo, hkReal hi, hkReal value )
	{
		int insertIndex = m_entries.getSize();
		for ( int i = 0; i < m_entries.getSize(); ++i )
		{
			const hkEntry& entry = m_entries[ i ];
			if ( lo > entry.m_lo )
			{
				insertIndex = i + 1;
				break;
			}
		}

		hkEntry entry;
		entry.m_lo = lo;
		entry.m_hi = hi;
		entry.m_value = value;

		m_entries.insertAt( insertIndex, entry );
	}


	hkReal evaluate( void ) 
	{
		// Is this a free edge
		if ( m_entries.getSize() == 0 )
		{
			return m_freeAngle;
		}

		// Sort by values
		hkSort( &m_entries[ 0 ], m_entries.getSize() );

		// Accumulate entries with same value
		hkArray< hkEntry > accumulatedEntries;
		for ( int rangeBegin = 0, rangeEnd = 1; rangeEnd <= m_entries.getSize(); ++rangeEnd )
		{
			if ( rangeEnd >= m_entries.getSize() || m_entries[ rangeBegin ] != m_entries[ rangeEnd ] )
			{
				int rangeSize = rangeEnd - rangeBegin;
				
				hkReal hi = hkReal(0);
				for ( int i = 0; i < rangeSize; ++i )
				{
					const hkEntry& entry = m_entries[ rangeBegin + i ];
					hi += ( entry.m_hi - entry.m_lo );
				}

				hkEntry newEntry;
				newEntry.m_value = m_entries[ rangeBegin ].m_value;
				newEntry.m_lo = hkReal(0);
				newEntry.m_hi = hkMath::clamp( hi, hkReal(0), hkReal(1) );

				accumulatedEntries.pushBack( newEntry );

				// Update range
				rangeBegin = rangeEnd;	
			}
		}


		// Find min angle
		hkReal minAngle = HK_REAL_PI;

		bool bFoundSignificantEntry = false;
		for ( int i = 0; i < accumulatedEntries.getSize(); ++i )
		{
			const hkEntry& entry = accumulatedEntries[ i ];
			if ( entry.m_hi - entry.m_lo < hkReal(0.6f) )
			{
				continue;
			}

			minAngle = hkMath::min2( minAngle, entry.m_value );
			bFoundSignificantEntry = true;
		}

		return bFoundSignificantEntry ? minAngle : hkReal(0);
	}


private:
	hkReal m_freeAngle;
	hkArray< hkEntry > m_entries;

};


HK_FORCE_INLINE bool hkAreTrianglesEqual( const hkpTriangleShape* triangle0, const hkpTriangleShape* triangle1, hkSimdRealParameter eps )
{
	return triangle0->getVertex<0>().allEqual<3>( triangle1->getVertex<0>(), eps ) && 
		   triangle0->getVertex<1>().allEqual<3>( triangle1->getVertex<1>(), eps ) && 
		   triangle0->getVertex<2>().allEqual<3>( triangle1->getVertex<2>(), eps );
}

HK_FORCE_INLINE bool hkTrianglesShareEdge( hkVector4Parameter v0, hkVector4Parameter v1, const hkpTriangleShape* triangle, hkSimdRealParameter eps ) 
{
	return ( triangle->getVertex<0>().allEqual<3>( v0, eps ) | triangle->getVertex<1>().allEqual<3>( v0, eps ) | triangle->getVertex<2>().allEqual<3>( v0, eps ) ) &&
		   ( triangle->getVertex<0>().allEqual<3>( v1, eps ) | triangle->getVertex<1>().allEqual<3>( v1, eps ) | triangle->getVertex<2>().allEqual<3>( v1, eps ) );
}


HK_FORCE_INLINE hkBool32 hkIsEdgeInTrianglePlane( hkVector4Parameter e, hkVector4Parameter n, hkSimdRealParameter eps )
{
	hkSimdReal absDot; absDot.setAbs(e.dot<3>(n));
	return absDot.isLess(eps);
}


void hkBuildClipPlanesForTriangle( hkVector4 out[ 5 ], const hkpTriangleShape* triangle, hkReal extraRadius )
{
	// Normal
	hkVector4 normal;
	hkpMeshWeldingUtility::calcAntiClockwiseTriangleNormal( triangle->getVertex<0>(), triangle->getVertex<1>(), triangle->getVertex<2>(), normal );

	// Clip edge against triangle 
	hkVector4 edge0, edge1, edge2;
	edge0.setSub( triangle->getVertex<1>(), triangle->getVertex<0>() );
	edge1.setSub( triangle->getVertex<2>(), triangle->getVertex<1>() );
	edge2.setSub( triangle->getVertex<0>(), triangle->getVertex<2>() );

	// Fill equations
	const hkSimdReal r = hkSimdReal::fromFloat(-extraRadius);
	hkVector4 a = normal;
	out[ 0 ].setXYZ_W(a, r - a.dot<3>( triangle->getVertex<0>() ));
	
	a.setNeg<3>( normal );
	out[ 1 ].setXYZ_W(a, r - a.dot<3>( triangle->getVertex<0>() ));
	
	a.setCross( edge0, normal );
	a.normalize<3>();
	out[ 2 ].setXYZ_W(a, r - a.dot<3>( triangle->getVertex<0>() ));

	a.setCross( edge1, normal );
	a.normalize<3>();
	out[ 3 ].setXYZ_W(a, r - a.dot<3>( triangle->getVertex<1>() ));

	a.setCross( edge2, normal );
	a.normalize<3>();
	out[ 4 ].setXYZ_W(a, r - a.dot<3>( triangle->getVertex<2>() ));
}


bool hkClipEdgeAgainstPlanes( hkVector4Parameter v0, hkVector4Parameter v1, int planeCount, const hkVector4* planeEquations, hkReal& lo, hkReal& hi )
{
	hkVector4 p0 = v0;
	hkVector4 p1 = v1;

	for ( int i = 0; i < planeCount; ++i )
	{
		hkVector4 plane = planeEquations[ i ];

		const hkSimdReal d0 = plane.dot<3>( p0 ) + plane.getW();
		const hkSimdReal d1 = plane.dot<3>( p1 ) + plane.getW();

		if ( d0.isGreaterEqualZero() && d1.isGreaterEqualZero() )
		{
			return false;
		}

		if ( (d0 * d1).isLessZero() )
		{
			const hkSimdReal t = d0 / ( d0 - d1 );

			hkVector4 x;
			x.setInterpolate( p0, p1, t );

			if ( d0.isGreaterZero() )
			{
				p0 = x;
			}
			else
			{
				p1 = x;
			}	
		}
	}


	// Compute the barycentric coordinates
	hkVector4 dv;
	dv.setSub( v1, v0 );

	const hkSimdReal L = dv.lengthSquared<3>();
	hkSimdReal invL; invL.setReciprocal(L);

	hkVector4 dp0, dp1;
	dp0.setSub( p0, v0 );
	dp1.setSub( p1, v0 );

	const hkSimdReal L1 = dp0.dot<3>( dv );
	const hkSimdReal L2 = dp1.dot<3>( dv );
	HK_ASSERT( 0xdd1298aa, L1.getReal() <= L2.getReal() );

	lo = (L1 * invL).getReal();
	hi = (L2 * invL).getReal();

	return true;
}


void hkpMeshWeldingUtility::computeWeldingInfo( hkpShapeCollection* collection, const hkpBvTreeShape* bvTree, hkpWeldingUtility::WeldingType weldingType, bool weldOpenEdges, bool disableEdges )
{
	HK_ON_DEBUG( const hkpShapeContainer* shapeContainer = bvTree->getContainer());
	HK_ASSERT2(0xaf6395ef, shapeContainer == collection, "The hkpMoppBvTreeShape you pass to computeWeldingInfo must be built referencing the input hkpShapeCollection.");

	hkpMeshWeldingUtility::ShapeInfo info;
	info.m_transform.setIdentity();
	info.m_shape = bvTree;
	hkLocalArray< hkpMeshWeldingUtility::ShapeInfo > shapes( 1 );
	shapes.pushBack( info );
	hkpMeshWeldingUtility::computeWeldingInfoMultiShape( hkTransform::getIdentity(), collection, weldingType, shapes, weldOpenEdges, disableEdges );
}


void hkpMeshWeldingUtility::computeWeldingInfoMultiShape( const hkTransform& meshTransform, hkpShapeCollection* mesh, hkpWeldingUtility::WeldingType weldingType, hkArray< ShapeInfo >& allShapes, bool weldOpenEdges, bool disableEdges )
{	

	if ( mesh->m_collectionType != hkpShapeCollection::COLLECTION_EXTENDED_MESH				&&
		 mesh->m_collectionType != hkpShapeCollection::COLLECTION_TRISAMPLED_HEIGHTFIELD	&&
		 mesh->m_collectionType != hkpShapeCollection::COLLECTION_COMPRESSED_MESH			&&
		 mesh->m_collectionType != hkpShapeCollection::COLLECTION_SIMPLE_MESH				&&
		 mesh->m_collectionType != hkpShapeCollection::COLLECTION_MESH_SHAPE
	   )
	{
		HK_WARN_ALWAYS(0x3f508c6a, "hkpShapeCollection must be a triangle collection that supports welding.");
		return;
	}

	mesh->initWeldingInfo( weldingType );
	if ( weldingType != hkpWeldingUtility::WELDING_TYPE_NONE )
	{
		mesh->m_disableWelding = true;
	}
	else
	{
		return;
	}

	hkpMeshWeldingUtility::WindingConsistency testConsistency = weldingType != hkpWeldingUtility::WELDING_TYPE_TWO_SIDED ? hkpMeshWeldingUtility::WINDING_TEST_CONSISTENCY : hkpMeshWeldingUtility::WINDING_IGNORE_CONSISTENCY;
	const hkReal degeneracyTolerance = 1e-5f;

	// Iterate all triangles
	for ( hkpShapeKey key = mesh->getFirstKey(); key != HK_INVALID_SHAPE_KEY; key = mesh->getNextKey( key ) )
	{
		hkpShapeBuffer buffer;
		const hkpShape* childShape = mesh->getChildShape( key, buffer );

		if ( childShape->getType() == hkcdShapeType::TRIANGLE )
		{
			const hkpTriangleShape* childTriangle = static_cast< const hkpTriangleShape* >( childShape );

			//if ( childTriangle->getWeldingType() != hkpWeldingUtility::WELDING_TYPE_TWO_SIDED )
			{
				hkInt16 combinedEdgeBitcodes = 0;

				hkVector4 triangleNormal;
				hkpMeshWeldingUtility::calcAntiClockwiseTriangleNormal( childTriangle->getVertex<0>(), childTriangle->getVertex<1>(), childTriangle->getVertex<2>(), triangleNormal );

				for ( int i = 0; i < 3; ++i )
				{
					// Create the histogram for this edge
					hkHistogram edgeHistogram( weldOpenEdges );

					// Get the edge vertices
					hkVector4 edgeVertex0 = childTriangle->getVertex( i );
					hkVector4 edgeVertex1 = childTriangle->getVertex( ( i + 1 ) % 3 );
					hkVector4 edgeVertex2 = childTriangle->getVertex( ( i + 2 ) % 3 );

					if ( hkpTriangleUtil::isDegenerate( edgeVertex0, edgeVertex1, edgeVertex2, degeneracyTolerance ) )
					{
						continue;
					}

					bool inconsistent = false;

					for ( int j = 0; j < allShapes.getSize(); ++j )
					{
						const hkTransform& transform = allShapes[ j ].m_transform;
						const hkpBvTreeShape* shape = allShapes[ j ].m_shape;

						// Get vertices, edge, and normal in this shape's local space
						hkTransform toLocalSpace;
						toLocalSpace._setMulInverseMul( transform, meshTransform );

						hkVector4 localTriangleNormal;
						localTriangleNormal._setRotatedDir( toLocalSpace.getRotation(), triangleNormal );

						hkVector4 localVertex0, localVertex1, localVertex2;
						localVertex0._setTransformedPos( toLocalSpace, edgeVertex0 );
						localVertex1._setTransformedPos( toLocalSpace, edgeVertex1 );
						localVertex2._setTransformedPos( toLocalSpace, edgeVertex2 );

						hkVector4 localEdge;
						localEdge.setSub( localVertex1, localVertex0 );
						localEdge.normalize<3>();

						// Query for shapekeys overlapping this edge
						hkInplaceArray< hkpShapeKey, 128 > collectedKeysAdjacentToEdge;
						{
							hkAabb aabb;
							aabb.setEmpty();
							aabb.includePoint( localVertex0 );
							aabb.includePoint( localVertex1 );

							shape->queryAabb( aabb, collectedKeysAdjacentToEdge );
						}

						// Iterate all resulting triangles
						for ( int k = 0; k < collectedKeysAdjacentToEdge.getSize(); ++k )
						{
							hkpShapeKey adjacentTriangleShapeKey = collectedKeysAdjacentToEdge[ k ];
							
							hkpShapeBuffer adjacentBuffer;
							const hkpShape* adjacentShape = shape->getContainer()->getChildShape( adjacentTriangleShapeKey, adjacentBuffer );
							
							if ( adjacentShape->getType() == hkcdShapeType::TRIANGLE )
							{
								const hkpTriangleShape* adjacentTriangle = static_cast< const hkpTriangleShape* >( adjacentShape );

								// Early outs
								if ( hkpTriangleUtil::isDegenerate( adjacentTriangle->getVertex<0>(), adjacentTriangle->getVertex<1>(), adjacentTriangle->getVertex<2>(), degeneracyTolerance ) )
								{
									continue;
								}

								const hkSimdReal eps = hkSimdReal::fromFloat(hkReal(0.001f));
								if( hkAreTrianglesEqual( childTriangle, adjacentTriangle, eps )                 || 
									hkTrianglesShareEdge( localVertex1, localVertex2, adjacentTriangle, eps )   ||
									hkTrianglesShareEdge( localVertex2, localVertex0, adjacentTriangle, eps )   )
								{
									continue;
								}

								hkVector4 adjacentTriangleNormal;
								hkpMeshWeldingUtility::calcAntiClockwiseTriangleNormal( adjacentTriangle->getVertex<0>(), adjacentTriangle->getVertex<1>(), adjacentTriangle->getVertex<2>(), adjacentTriangleNormal );

								// Early out
								const hkSimdReal epsEdge = hkSimdReal::fromFloat(hkReal(0.05f));
								if( !hkIsEdgeInTrianglePlane( localEdge, adjacentTriangleNormal, epsEdge ) )
								{
									continue;
								}

								// Compute angle
								hkReal angle;
								hkVector4 localEdgeOrtho;
								{
									localEdgeOrtho.setCross( localEdge, localTriangleNormal );
									localEdgeOrtho.normalize<3>();

									angle = hkpMeshWeldingUtility::calcAngleForEdge( localTriangleNormal, localEdgeOrtho, adjacentTriangleNormal );
								}

								bool edgeFound = false;

								// edge-edge check
								for ( int l = 0; l < 3; ++l )
								{
									const hkVector4& adjVertex0 = adjacentTriangle->getVertex( l );
									const hkVector4& adjVertex1 = adjacentTriangle->getVertex( ( l + 1 ) % 3 );

									hkVector4 adjEdge;
									adjEdge.setSub( adjVertex1 , adjVertex0 );
									adjEdge.normalize<3>();

									hkVector4 cross;
									cross.setCross( localEdge, adjEdge );
									hkReal sinEdges2 = cross.lengthSquared<3>().getReal();
									if ( sinEdges2 < (0.0001f*0.0001f) ) // if edges are parallel
									{										
										hkVector4 point0; point0.setSub( adjVertex0, localVertex0 );
										hkVector4 point1; point1.setSub( adjVertex1, localVertex0 );
										// check distance to edge
										cross.setCross( point0, localEdge );
										const hkSimdReal epsLen2 = hkSimdReal::fromFloat(hkReal(0.01f*0.01f));
										if ( cross.lengthSquared<3>().isGreater(epsLen2) )
										{
											continue;
										}
										cross.setCross( point1, localEdge );
										if ( cross.lengthSquared<3>().isGreater(epsLen2) )
										{
											continue;
										}
										// check for intersection
										hkVector4 segment; segment.setSub( localVertex1, localVertex0 );
										const hkSimdReal lenSqr = segment.lengthSquared<3>();
										hkSimdReal invLenSqr; invLenSqr.setReciprocal(lenSqr);
										hkSimdReal t0 = point0.dot<3>( segment ) * invLenSqr;
										hkSimdReal t1 = point1.dot<3>( segment ) * invLenSqr;
										t0.setMin(t0, hkSimdReal_1);
										t1.setMin(t1, hkSimdReal_1);
										t0.setMax(t0, hkSimdReal_0);
										t1.setMax(t1, hkSimdReal_0);
										hkSimdReal overlap; overlap.setAbs( t1 - t0 );
										if ( overlap.getReal() > hkReal(0.1f) )
										{											
											if ( testConsistency && (t1 > t0) )
											{
												HK_WARN_ALWAYS( 0xabba751e, "Inconsistent triangle winding between at least triangle " << key << " and triangle " << adjacentTriangleShapeKey << ". One sided welding will not work.");
												inconsistent = true;
												// mark the adjacent triangle as part of an inconsistent welding pair
												hkUint16 adjWeldingInfo = adjacentTriangle->getWeldingInfo();
												adjWeldingInfo |= 1 << 15;
												mesh->setWeldingInfo( adjacentTriangleShapeKey, adjWeldingInfo );												
											}
											else
											{
												edgeFound = true;
												hkSimdReal min01; min01.setMin(t0,t1);
												hkSimdReal max01; max01.setMax(t0,t1);
												edgeHistogram.addEntry( min01.getReal(), max01.getReal(), angle );
											}
										}
									}
								}

								if ( !edgeFound )
								{
									hkVector4 planeEquations[ 5 ];
									hkBuildClipPlanesForTriangle( planeEquations, adjacentTriangle, hkReal(0.05f) );
								
									hkReal lo, hi;
									if ( hkClipEdgeAgainstPlanes( localVertex0, localVertex1, 5, planeEquations, lo, hi ) )
									{
										edgeHistogram.addEntry( lo, hi, angle );
									}
								}
							}
						}
					}

					if ( disableEdges && inconsistent )
					{
						// mark the edge for bad welding
						combinedEdgeBitcodes = hkpMeshWeldingUtility::modifyCombinedEdgesBitcode( combinedEdgeBitcodes, i, hkpWeldingUtility::NUM_ANGLES );						
					}
					else
					{
						hkReal bestAngle = edgeHistogram.evaluate();

						int edgeBitcode = hkpWeldingUtility::calcEdgeAngleBitcode( bestAngle );
						combinedEdgeBitcodes = hkpMeshWeldingUtility::modifyCombinedEdgesBitcode( combinedEdgeBitcodes, i, edgeBitcode );
					}

					if ( inconsistent )
					{
						// mark the triangle as part of an inconsistent welding pair
						combinedEdgeBitcodes |= 1 << 15;
					}
				}

				// Save welding info per triangle
				mesh->setWeldingInfo( key, combinedEdgeBitcodes );
			}
		}
	}
}

hkResult hkpMeshWeldingUtility::testWindingConsistency( hkVector4Parameter localVertex0, hkVector4Parameter localEdgeOrtho, hkVector4Parameter localNormal, const hkpTriangleShape* adjacentTriangle, hkVector4Parameter adjacentNormal )
{
	{
		hkReal maxHeight = -HK_REAL_MAX;
		hkReal minHeight = HK_REAL_MAX;
		for (int i = 0; i < 3; ++i )
		{
			hkVector4 v; v.setSub( adjacentTriangle->getVertex(i), localVertex0 );
			hkReal height = localEdgeOrtho.dot<3>(v).getReal();
			maxHeight = hkMath::max2( height, maxHeight );
			minHeight = hkMath::min2( height, minHeight );
		}
		//minHeight = -minHeight;
		hkReal normTest;
		if ( ( maxHeight > .01f) || (minHeight < .01f ))
		{
			// Test if we have a T junction and if so do not check winding
			if ( (maxHeight > .2f) && (minHeight < .2f) )
			{
				return HK_SUCCESS;
			}
			normTest = localNormal.dot<3>( adjacentNormal ).getReal() * hkReal(hkMath::fabs(maxHeight) > hkMath::fabs(minHeight));
			if (normTest < hkReal(0)) 
				return HK_FAILURE;
		}
	}
	// If the triangles' normals are close to right angles, perform the same test using the triangle normal rather than the edge ortho normal
	{
		hkReal maxHeight = -HK_REAL_MAX;
		hkReal minHeight = HK_REAL_MAX;
		for (int i = 0; i < 3; ++i )
		{
			hkVector4 v; v.setSub( adjacentTriangle->getVertex(i), localVertex0 );
			hkReal height = localNormal.dot<3>(v).getReal();
			maxHeight = hkMath::max2( height, maxHeight );
			minHeight = hkMath::min2( height, minHeight );
		}
		hkReal normTest;
		if ( ( maxHeight > .01f) || (minHeight < .01f ))
		{
			// Test if we have a T junction and if so do not check winding
			if ( (maxHeight > .2f) && (minHeight < .2f) )
			{
				return HK_SUCCESS;
			}

			normTest = localEdgeOrtho.dot<3>( adjacentNormal ).getReal() * hkReal(hkMath::fabs(maxHeight) > hkMath::fabs(minHeight));
			if (normTest > hkReal(0))
				return HK_FAILURE;
		}
	}
	return HK_SUCCESS;
}


hkReal hkpMeshWeldingUtility::calcAngleForEdge( hkVector4Parameter edgeNormal, hkVector4Parameter edgeOrtho, hkVector4Parameter triangleNormal )
{
	hkSimdReal sinAngle = triangleNormal.dot<3>( edgeOrtho );
	hkSimdReal cosAngle = triangleNormal.dot<3>( edgeNormal );

	hkSimdReal a = hkVector4Util::atan2Approximation(sinAngle,cosAngle);
	HK_ASSERT (0xaf730363, (a >= -hkSimdReal_Pi) && (a <= hkSimdReal_Pi) );
	return a.getReal();
}


void hkpMeshWeldingUtility::calcAntiClockwiseTriangleNormal(const hkVector4& vertex0, const hkVector4& vertex1, const hkVector4& vertex2, hkVector4& normal)
{
	hkVector4 edges[2];
	{
		edges[0].setSub(vertex1, vertex0);
		edges[1].setSub(vertex2, vertex1);
	}

	normal.setCross(edges[0], edges[1]);
	normal.normalize<3>();
}


hkUint16 hkpMeshWeldingUtility::modifyCombinedEdgesBitcode(hkUint16 combinedBitcode, int edgeIndex, int bitcode)
{
	HK_ASSERT (0xaf730369, (bitcode >= 0) && (bitcode < hkpWeldingUtility::NUM_ANGLES + 1) );

	bitcode <<= (edgeIndex*5);

	int mask = ~(0x1f << (edgeIndex*5));

	combinedBitcode &= mask;
	combinedBitcode |= bitcode;

	return combinedBitcode;
}

int hkpMeshWeldingUtility::calcSingleEdgeBitcode(hkUint16 triangleEdgesBitcode, int edgeIndex )
{
	int edgeAngleBitcode = (triangleEdgesBitcode >> (edgeIndex*5)) & 0x1f; // filter 5bits
	return edgeAngleBitcode;
}



//
// DEPRECATED FUNCTIONS FROM HERE ON
//





hkResult hkpMeshWeldingUtility::calcWeldingInfoForTriangle( hkpShapeKey shapeKey, const hkpBvTreeShape* bvTreeShape, WindingConsistency testConsistency, hkUint16& info )
{
	hkpShapeBuffer buffer;
	const hkpShape* shape = bvTreeShape->getContainer()->getChildShape(shapeKey, buffer);
	HK_ASSERT2(0x86af9431, shape->getType() == hkcdShapeType::TRIANGLE, "calcWeldingInfoForTriangle() called on a shape that is not a triangle. Make sure the MoppShape references a mesh");
	const hkpTriangleShape* triangleShape = static_cast<const hkpTriangleShape*>(shape);

	// If the triangle is degenerate, return 0.  As degenerate triangles will not be included in the MOPP, this is OK.
	if ( hkpTriangleUtil::isDegenerate( triangleShape->getVertex<0>(), triangleShape->getVertex<1>(), triangleShape->getVertex<2>()) )
	{
		info = 0;
		return HK_FAILURE;
	}

	hkInt16 combinedEdgeBitcodes = 0;

	//
	// for all three edges find adjacent triangles and calculate welding info
	//
	hkResult result = HK_SUCCESS;
	{
		for (int edgeIndex = 0; edgeIndex < 3; edgeIndex++)
		{
			if ( calcBitcodeForTriangleEdge( bvTreeShape, triangleShape, shapeKey, edgeIndex, testConsistency, combinedEdgeBitcodes ) == HK_FAILURE )
			{
				result = HK_FAILURE;
			}
		}
	}

	info = combinedEdgeBitcodes;

	return result;
}

hkBool HK_CALL hkpMeshWeldingUtility::isTriangleWindingValid( hkpShapeKey shapeKey, const hkpBvTreeShape* bvTreeShape )
{
	hkpShapeBuffer buffer;
	const hkpShape* shape = bvTreeShape->getContainer()->getChildShape(shapeKey, buffer);
	HK_ASSERT2(0x86af9431, shape->getType() == hkcdShapeType::TRIANGLE, "calcWeldingInfoForTriangle() called on a shape that is not a triangle. Make sure the MoppShape references a mesh");
	const hkpTriangleShape* triangleShape = static_cast<const hkpTriangleShape*>(shape);
	
	// Disable winding assert in calcBitcodeForTriangleEdge
	hkBool isWindingAssertEnabled = hkError::getInstance().isEnabled(0xfe98751e);
	if( isWindingAssertEnabled )
	{
		hkError::getInstance().setEnabled( 0xfe98751e, false );
	}

	// validate winding info for all three edges
	hkBool isValid = true;
	{
		hkInt16 combinedEdgeBitcodes = 0;
		for (int edgeIndex = 0; edgeIndex < 3; edgeIndex++)
		{
			if( calcBitcodeForTriangleEdge( bvTreeShape, triangleShape, shapeKey, edgeIndex, WINDING_IGNORE_CONSISTENCY, combinedEdgeBitcodes ) == HK_FAILURE )
			{
				isValid = false;
				break;
			}
		}
	}

	if( isWindingAssertEnabled )
	{
		hkError::getInstance().setEnabled( 0xfe98751e, true );
	}

	return isValid;
}

//
// This method searches for the neighboring triangle that shares the supplied edge with the supplied 'original' triangle
// and then calculates the edge's angle-bitcode.
//
hkResult hkpMeshWeldingUtility::calcBitcodeForTriangleEdge( const hkpBvTreeShape* bvTreeShape, const hkpTriangleShape* triangleShape, hkpShapeKey triangleShapeKey, int edgeIndex, WindingConsistency testConsistency, hkInt16& combinedBitcodesOut )
{
	const hkVector4 *verticesTriangle0 = triangleShape->getVertices();

	//
	// AABB-query all triangles adjacent to edge
	//
	hkInplaceArray<hkpShapeKey,128> collectedKeysAdjacentToEdge;
	{
		hkReal queryRadius = 0.0001f;
		hkVector4 edgeVertex = verticesTriangle0[edgeIndex%3];
		hkAabb aabb;
		{
			hkVector4 min; min.set(-queryRadius, -queryRadius, -queryRadius);
			hkVector4 max; max.set( queryRadius,  queryRadius,  queryRadius);
			min.add(edgeVertex);
			max.add(edgeVertex);
			aabb.m_min = min;
			aabb.m_max = max;
		}
		bvTreeShape->queryAabb(aabb, collectedKeysAdjacentToEdge);
	}

	//
	// find the actual neighboring triangle from all reported triangles
	//
	bool isWindingConsistant = true;
	{
		int numNeighborsValid = 0;

		for (int hitlistIdx = collectedKeysAdjacentToEdge.getSize()-1; hitlistIdx >= 0; hitlistIdx--)
		{
			hkpShapeKey neighboringTriangleShapeKey = collectedKeysAdjacentToEdge[hitlistIdx];

			// skip the original triangle
			if ( neighboringTriangleShapeKey == triangleShapeKey )
			{
				continue;
			}

			//
			// get the vertices of the neighboring triangle
			//
			const hkVector4 *verticesTriangle1;
			{
				const hkpTriangleShape* adjacentTriangleShape;
				hkpShapeBuffer buffer;
				{
					const hkpShape* adjacentShape = bvTreeShape->getContainer()->getChildShape(neighboringTriangleShapeKey, buffer);
					if (adjacentShape->getType() != hkcdShapeType::TRIANGLE)
					{
						continue;
					}
					adjacentTriangleShape = static_cast<const hkpTriangleShape*>(adjacentShape);
				}
				verticesTriangle1 = adjacentTriangleShape->getVertices();
			}

			//
			// get the 4-element-vertex array of both neighboring triangles
			// (note: the array has to be able to store 6 elements!)
			//
			hkVector4 vertices[6];
			{
				int orderedEdgeVerticesOnTriangle1[2];
				int numSingularVertices = createSingularVertexArray(verticesTriangle0, verticesTriangle1, edgeIndex, vertices, orderedEdgeVerticesOnTriangle1);

				// skip all triangles that do not share our original edge
				if ( numSingularVertices != 4 )
				{
					continue;
				}
				
				// test consistent winding between the neighbors
				if ( testConsistency != WINDING_IGNORE_CONSISTENCY )
				{
					if( orderedEdgeVerticesOnTriangle1[0] != ( orderedEdgeVerticesOnTriangle1[1] + 1 ) % 3 )
					{
						if (isWindingConsistant)
						{
							HK_WARN_ALWAYS(0xabba751e, "Inconsistant triangle winding between at least triangle " << triangleShapeKey << " and triangle " << neighboringTriangleShapeKey << ". One sided welding will not work.");
						}
						isWindingConsistant = false;
					}
				}
			}

			int edgeBitcode = hkpMeshWeldingUtility::calcEdgeAngleBitcode(vertices);
			combinedBitcodesOut = hkpMeshWeldingUtility::modifyCombinedEdgesBitcode(combinedBitcodesOut, edgeIndex, edgeBitcode);

			numNeighborsValid++;
		}
	}

	return isWindingConsistant ? HK_SUCCESS : HK_FAILURE;
}

//
// This method builds a vertex array from the vertices of the supplied two triangles.
//
// The supplied array must at least have a size of 6 (i.e. there must be room for a
// total of 6 vectors).
// The vertices of triangle 0 go first (with the vertices of the supplied edge upfront),
// followed by the vertices of triangle 1. Identical vertices will only be added once,
// e.g. in case of two triangles sharing one edge this array will be filled with only
// 4 vertices. The value returned is the total number of singular vertices in the array.
// Note that in the case that both triangles are exactly the same (i.e. all 3 vertices
// of one triangle have an identical counterpart in the other triangle) we will still
// return 4 vertices in the array (as we only check for duplicates of the 'edge' vertices
// of triangle 0)!
//
int hkpMeshWeldingUtility::createSingularVertexArray(const hkVector4 *vertices0, const hkVector4 *vertices1, int edgeIndex, hkVector4* vertexArrayOut, int orderedEdgeVerticesOnTriangle1[2] ) 
{
	// first three vertices are the vertices from the first triangle, with the 'edge' vertices upfront
	vertexArrayOut[0] = vertices0[(edgeIndex+0)%3];
	vertexArrayOut[1] = vertices0[(edgeIndex+1)%3];
	vertexArrayOut[2] = vertices0[(edgeIndex+2)%3];

	int numSingularVertices = 3;

	{
		for (int i = 0; i < 3; i++)
		{
			bool vertexShared = false;

			for (int k = 0; k < 2; k++)
			{
				hkVector4 distance; distance.setSub(vertexArrayOut[k], vertices1[i]);

				if ( distance.lengthSquared<3>().isEqualZero() ) 
				{
					orderedEdgeVerticesOnTriangle1[k] = i;
					vertexShared = true;
					break;
				}
			}

			// only append vertex if it is no 'duplicate' of either one of the two edge vertices
			if ( vertexShared == false )
			{
				vertexArrayOut[numSingularVertices++] = vertices1[i];
			}

		}
	}

	return numSingularVertices;
}

//
// This method calculates the "bitcode" for the edge between the two supplied triangles (in 4-vertex array form).
// See calcAngleFromVertices() for a detailed description of the arrays structure.
// Note that for angles < 0 we will snap to the "lower" fixed angle whereas for angles > 0 we will snap to the "higher"
// fixed angle!
//
int hkpMeshWeldingUtility::calcEdgeAngleBitcode(const hkVector4* vertices)
{

	hkReal sinAngle;
	hkReal cosAngle;
	hkReal angle = hkpMeshWeldingUtility::calcAngleFromVertices(vertices, sinAngle, cosAngle);
	//hkReal debugAngleDegrees = (angle / HK_REAL_PI) * 180.0f;

	int i;
	for (i = 0; i < hkpWeldingUtility::NUM_ANGLES; i++)
	{
		// refAngle runs from -180 degree to +180 degree in 360/(NUM_ANGLES-1) degree steps
		hkReal refAngle = -HK_REAL_PI + ((2*HK_REAL_PI/hkReal(hkpWeldingUtility::NUM_ANGLES-1)) * hkReal(i));
		//hkReal debugRefAngleDegrees = (refAngle / HK_REAL_PI) * 180.0f;

		if ( angle <= refAngle )
		{
			if ( angle > hkReal(0) && angle != refAngle )
			{
				i--;
			}
			break;
		}

	}

	HK_ASSERT (0xaf730364, (i >= 0) && (i < hkpWeldingUtility::NUM_ANGLES) );

	return i;
}



//
// Calculates the real angle and its sine and cosine between the two supplied triangles (in 4-vertex array form).
// The vertices are expected to be in the following order:
//   v0(t0), v1(t0), v2(t0), v1(t1)
// with
//   v0(t0) == v0(t1) and v0(t0) == v2(t1)
// i.e. v0(t0) and v1(t0) is the edge between both supplied triangles!
//
hkReal hkpMeshWeldingUtility::calcAngleFromVertices(const hkVector4* vertices, hkReal& sinAngleOut, hkReal& cosAngleOut)
{

	hkVector4 normalTriangle0;
	hkpMeshWeldingUtility::calcAntiClockwiseTriangleNormal(vertices[0], vertices[1], vertices[2], normalTriangle0);

	hkVector4 normalTriangle1;
	hkpMeshWeldingUtility::calcAntiClockwiseTriangleNormal(vertices[1], vertices[0], vertices[3], normalTriangle1);

	//
	// calculate an orthogonal vector to the triangle's normal and its shared edge
	//
	hkVector4 orthNormalEdge;
#if 0
	{
		// this version simply works for corners of the triangle as well
		hkVector4 edge10; edge10.setSub4(vertices[1], vertices[0]); edge10.normalize4();
		hkVector4 edge20; edge20.setSub4(vertices[2], vertices[0]); edge20.normalize4();

		hkReal posProjected = edge10.dot3(edge20);
		edge10.mul4(posProjected);

		orthNormalEdge.setSub4(edge10, edge20);
		orthNormalEdge.normalize4();

	}
#else
	{
		hkVector4 edge10;
		edge10.setSub(vertices[1], vertices[0]);
		edge10.normalize<3>();
		orthNormalEdge.setCross(edge10, normalTriangle0);
	}
#endif

	hkSimdReal sinAngle = normalTriangle1.dot<3>(orthNormalEdge);
	hkSimdReal cosAngle = normalTriangle1.dot<3>(normalTriangle0);

	hkSimdReal a = hkVector4Util::atan2Approximation(sinAngle,cosAngle);
	sinAngle.store<1>(&sinAngleOut);
	cosAngle.store<1>(&cosAngleOut);
	HK_ASSERT (0xaf730363, (a >= -hkSimdReal_Pi) && (a <= hkSimdReal_Pi) );
	return a.getReal();
}


hkUint16 HK_CALL hkpMeshWeldingUtility::computeTriangleWeldingInfo(const hkVector4* triangle, 
																   const hkVector4* neighbors, int numNeighbors, 
																   hkBool weldOpenEdges, hkReal tolerance)
{
	if (numNeighbors == 0 || hkpTriangleUtil::isDegenerate(triangle[0], triangle[1], triangle[2]))
	{
		return 0;
	}

	hkVector4 triangleNormal;
	hkpMeshWeldingUtility::calcAntiClockwiseTriangleNormal(triangle[0], triangle[1], triangle[2], triangleNormal);	
	

	// Build histograms for each edge as we look for touching triangles
	hkHistogram edgeHistogram0(weldOpenEdges);
	hkHistogram edgeHistogram1(weldOpenEdges);
	hkHistogram edgeHistogram2(weldOpenEdges);
	hkHistogram* histograms[] = { &edgeHistogram0, &edgeHistogram1, &edgeHistogram2 };
	const hkSimdReal toleranceSr = hkSimdReal::fromFloat(tolerance);
	for (int i = 0; i < numNeighbors; ++i)
	{
		// Check for degeneracy
		const hkVector4* otherTriangle = neighbors + (i * 3);
		if (hkpTriangleUtil::isDegenerate(otherTriangle[0], otherTriangle[1], otherTriangle[2]))
		{
			continue;
		}

		// Calculate other triangle normal
		hkVector4 otherTriangleNormal;
		hkpMeshWeldingUtility::calcAntiClockwiseTriangleNormal(otherTriangle[0], otherTriangle[1], otherTriangle[2], otherTriangleNormal);		

		// Compare each edge to the other triangle
		for( int j = 0; j < 3; ++j )
		{
			const hkVector4& v0 = triangle[j];
			const hkVector4& v1 = triangle[(j + 1) % 3];			

			// Check if edge is ~coplanar with other triangle
			hkVector4 edge;
			edge.setSub(v1, v0);			
			edge.normalize<3>();
			const hkSimdReal eDot = edge.dot<3>(otherTriangleNormal);
			hkSimdReal absEDot; absEDot.setAbs(eDot);
			if (absEDot.isLess(toleranceSr))
			{
				// Clip edge against other triangle
				hkReal lowClip = hkReal(0);
				hkReal highClip = hkReal(1);
				if (clipLineAgainstTriangle(v0, v1, otherTriangle, otherTriangleNormal, tolerance, lowClip, highClip))
				{				
					// Compute angle
					hkVector4 edgeOrtho;
					edgeOrtho.setCross(edge, triangleNormal);
					edgeOrtho.normalize<3>();
					hkReal angle = hkpMeshWeldingUtility::calcAngleForEdge(triangleNormal, edgeOrtho, otherTriangleNormal);

					// Add it to relevant histogram
					histograms[j]->addEntry(lowClip, highClip, angle);					
				}
			}
		}
	}	

	// Convert "best" angles from histograms to welding info
	hkUint16 weldingInfo = 0;
	for (int i = 0; i < 3; ++i)
	{
		hkReal bestAngle = histograms[i]->evaluate();
		int edgeBitcode = hkpWeldingUtility::calcEdgeAngleBitcode(bestAngle);
		weldingInfo = hkpMeshWeldingUtility::modifyCombinedEdgesBitcode(weldingInfo, i, edgeBitcode);
	}
	return weldingInfo;
}


// Utility function

hkBool HK_CALL hkpMeshWeldingUtility::clipLineAgainstTriangle(hkVector4Parameter v0, hkVector4Parameter v1, 
															  const hkVector4* vertices, const hkVector4& normal, 
															  hkReal padding, hkReal& lo, hkReal& hi)
{
	// Clip the endpoints
	hkVector4 p0 = v0;
	hkVector4 p1 = v1;
	bool clipped = false;
	const hkSimdReal paddingSr = hkSimdReal::fromFloat(-padding);
	for ( int i = -2; i < 3; ++i )
	{
		// Create plane
		hkVector4 plane;
		hkSimdReal planeW;
		if( i == -2 )	// top
		{
			plane = normal;
			planeW = paddingSr - plane.dot<3>( vertices[0] );
		}
		else if( i == -1 )	// bottom
		{
			plane.setNeg<3>(normal);
			planeW = paddingSr - plane.dot<3>( vertices[0] );
		}
		else	// sides
		{
			hkVector4 edge;
			edge.setSub( vertices[(i+1)%3], vertices[i] );

			plane.setCross( edge, normal );
			plane.normalize<3>();
			planeW = paddingSr - plane.dot<3>( vertices[i] );
		}

		// Distances from the plane
		const hkSimdReal d0 = plane.dot<3>( p0 ) + planeW;
		const hkSimdReal d1 = plane.dot<3>( p1 ) + planeW;

		// Exit if both endpoints are "outside" of the plane
		if ( d0.isGreaterEqualZero() && d1.isGreaterEqualZero() )
		{
			return false;
		}

		// Clip if crossing the plane
		if ( (d0 * d1).isLessZero() )
		{
			const hkSimdReal t = d0 / ( d0 - d1 );

			hkVector4 x;
			x.setInterpolate( p0, p1, t );

			if ( d0.isGreaterZero() )
			{
				p0 = x;
			}
			else
			{
				p1 = x;
			}

			clipped = true;
		}
	}

	// Compute the barycentric coordinates
	if( clipped )
	{
		hkVector4 dv;
		dv.setSub( v1, v0 );

		hkVector4 dp0, dp1;
		dp0.setSub( p0, v0 );
		dp1.setSub( p1, v0 );

		const hkSimdReal L1 = dp0.dot<3>( dv );
		const hkSimdReal L2 = dp1.dot<3>( dv );
		HK_ASSERT( 0xdd1298aa, L1.getReal() <= L2.getReal() );

		const hkSimdReal L = dv.lengthSquared<3>();
		hkSimdReal invL; invL.setReciprocal(L);
		lo = (L1 * invL).getReal();
		hi = (L2 * invL).getReal();
	}
	else
	{
		lo = hkReal(0);
		hi = hkReal(1);
	}

	return true;
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
