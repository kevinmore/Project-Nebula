/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h>

#include <Common/Base/Algorithm/Collide/1AxisSweep/hk1AxisSweep.h>

// this
#include <Common/GeometryUtilities/Matching/hkGeometryMatchingUtils.h>

void HK_CALL hkGeometryMatchingUtils::matchTriangles( const hkArray<Geometry>& referenceTriangles, const hkArray<Geometry>& searchTriangles,  hkReal maxDistance, hkGeometryMatchingUtils::TriangleMap& triangleMapOut )
{
	int numReferenceTriangles = 0;
	{
		for (int i = 0; i < referenceTriangles.getSize(); i++)
		{
			numReferenceTriangles += referenceTriangles[i].m_numTriangles;
		}
	}

	int numSearchTriangles = 0;
	{
		for (int i = 0; i < searchTriangles.getSize(); i++)
		{
			numSearchTriangles += searchTriangles[i].m_numTriangles;
		}
	}

	const hkSimdReal aabbExpansion = hkSimdReal::fromFloat(maxDistance);

	//
	//	Build the AABBs for the reference triangles
	//
	hkArray<hk1AxisSweep::AabbInt>::Temp referenceAabbs(numReferenceTriangles+4);
	hkArray<TriangleMap::Hit>::Temp referenceMap(numReferenceTriangles);
	{
		int f = 0;
		for (int i = 0; i < referenceTriangles.getSize(); i++)
		{
			const Geometry& geo = referenceTriangles[i];
			const int* ip = geo.m_triangleIndices;
			for ( int ti =0; ti < geo.m_numTriangles; ip = hkAddByteOffsetConst<int>( ip, geo.m_triangleIndexStride ), ti++)
			{
				const hkVector4& r0 = *hkAddByteOffsetConst<hkVector4>( geo.m_vertices, ip[0] * geo.m_vertexStride );
				const hkVector4& r1 = *hkAddByteOffsetConst<hkVector4>( geo.m_vertices, ip[1] * geo.m_vertexStride );
				const hkVector4& r2 = *hkAddByteOffsetConst<hkVector4>( geo.m_vertices, ip[2] * geo.m_vertexStride );

				hkAabb aabb;
				aabb.m_min.setMin( r0, r1 );
				aabb.m_max.setMax( r0, r1 );
				aabb.includePoint( r2 );
				aabb.m_min.setSub(aabb.m_min,aabbExpansion);
				aabb.m_max.setAdd(aabb.m_max,aabbExpansion);

				referenceAabbs[f].set( aabb, f );
				referenceMap[f].m_geometryIndex = hkUint16(i);
				referenceMap[f].m_triangleIndex = ti;

				f++;
			}
		}
		while (f < numReferenceTriangles+4) { referenceAabbs[f++].setEndMarker(); }
	}


	//
	//	Build AABBs for the search triangles
	//
	triangleMapOut.m_foundReferenceTriangle.setSize(numSearchTriangles);
	triangleMapOut.m_startIndexPerGeometry.setSize( searchTriangles.getSize() );
	hkArray<hk1AxisSweep::AabbInt>::Temp searchAabbs(numSearchTriangles+4);
	hkArray<TriangleMap::Hit>::Temp searchMap(numSearchTriangles);
	{
		int f = 0;
		for (int i = 0; i < searchTriangles.getSize(); i++)
		{
			const Geometry& geo = searchTriangles[i];
			triangleMapOut.m_startIndexPerGeometry[i] = f;
			const int* ip = geo.m_triangleIndices;
			for ( int ti =0; ti < geo.m_numTriangles; ip = hkAddByteOffsetConst<int>(ip, geo.m_triangleIndexStride), ti++)
			{
				const hkVector4& r0 = *hkAddByteOffsetConst<hkVector4>( geo.m_vertices, ip[0] * geo.m_vertexStride );
				const hkVector4& r1 = *hkAddByteOffsetConst<hkVector4>( geo.m_vertices, ip[1] * geo.m_vertexStride );
				const hkVector4& r2 = *hkAddByteOffsetConst<hkVector4>( geo.m_vertices, ip[2] * geo.m_vertexStride );

				hkAabb aabb;
				aabb.m_min.setMin( r0, r1 );
				aabb.m_max.setMax( r0, r1 );
				aabb.includePoint( r2 );
				aabb.m_min.setSub(aabb.m_min,aabbExpansion);
				aabb.m_max.setAdd(aabb.m_max,aabbExpansion);

				searchAabbs[f].set( aabb, f );
				searchMap[f].m_geometryIndex = hkUint16(i);
				searchMap[f].m_triangleIndex = ti;

				f++;
			}
		}
		while (f < numSearchTriangles+4) { searchAabbs[f++].setEndMarker(); }
	}

	//
	//	sort
	//
	{
		hk1AxisSweep::sortAabbs(referenceAabbs.begin(), numReferenceTriangles);
		hk1AxisSweep::sortAabbs(searchAabbs.begin(), numSearchTriangles);
	}

	//
	//	run 1-axis sweep
	//
	hkArray<hkKeyPair>::Temp pairs;
	int numPairs;
	{
		int capacity = hkMath::max2( numReferenceTriangles, numSearchTriangles) * 20;
		while(1)
		{
			pairs.setSize(capacity);

			hkPadSpu<int> numPairsSkipped = 0;
			numPairs = hk1AxisSweep::collide(referenceAabbs.begin(), numReferenceTriangles, searchAabbs.begin(), numSearchTriangles, pairs.begin(), capacity, numPairsSkipped);
			if ( !numPairsSkipped )
			{
				break;
			}
			capacity += numPairsSkipped;
		}
	}

	//
	//	Find the closest match for each face
	//
	const hkSimdReal oneThird = hkSimdReal_Inv3;

	triangleMapOut.m_startIndexPerGeometry.setSize(searchTriangles.getSize());
	{
		hkArray<hkReal>::Temp bestDistance(numSearchTriangles);
		for ( int i =0; i < numSearchTriangles; i++)
		{
			bestDistance[i] = maxDistance;
			triangleMapOut.m_foundReferenceTriangle[i].m_geometryIndex = hkUint16(-1);
			triangleMapOut.m_foundReferenceTriangle[i].m_triangleIndex = -1;
			triangleMapOut.m_foundReferenceTriangle[i].m_indexOffset = 0;
		}

		for (int p=0; p < numPairs; p++ )
		{
			const hkKeyPair& pair = pairs[p];

			int rIndex = pair.m_keyA;
			TriangleMap::Hit& rMap = referenceMap[rIndex];
			const Geometry& refGeom = referenceTriangles[ rMap.m_geometryIndex ];
			const int* rip  = hkAddByteOffsetConst( refGeom.m_triangleIndices, refGeom.m_triangleIndexStride * rMap.m_triangleIndex );

			const hkVector4& r0 = *hkAddByteOffsetConst<hkVector4>( refGeom.m_vertices, rip[0] * refGeom.m_vertexStride );
			const hkVector4& r1 = *hkAddByteOffsetConst<hkVector4>( refGeom.m_vertices, rip[1] * refGeom.m_vertexStride );
			const hkVector4& r2 = *hkAddByteOffsetConst<hkVector4>( refGeom.m_vertices, rip[2] * refGeom.m_vertexStride );

			int sIndex = pair.m_keyB;
			TriangleMap::Hit& sMap = searchMap[sIndex];
			const Geometry& searchGeom = searchTriangles[ sMap.m_geometryIndex ];
			const int* sip  = hkAddByteOffsetConst( searchGeom.m_triangleIndices, searchGeom.m_triangleIndexStride * sMap.m_triangleIndex );

			const hkVector4& s0 = *hkAddByteOffsetConst<hkVector4>( searchGeom.m_vertices, sip[0] * searchGeom.m_vertexStride );
			const hkVector4& s1 = *hkAddByteOffsetConst<hkVector4>( searchGeom.m_vertices, sip[1] * searchGeom.m_vertexStride );
			const hkVector4& s2 = *hkAddByteOffsetConst<hkVector4>( searchGeom.m_vertices, sip[2] * searchGeom.m_vertexStride );

			TriangleMap::Hit& hit = triangleMapOut.m_foundReferenceTriangle[sIndex];

			hkSimdReal bestDistance_sIndex; bestDistance_sIndex.load<1>(&bestDistance[sIndex]);

			//////////////////////////
			// Cyclic
			//////////////////////////
			{
				const hkSimdReal d = oneThird * (r0.distanceTo( s0 ) + r1.distanceTo( s1 ) + r2.distanceTo( s2 ));
				if ( d.isLess(bestDistance_sIndex) )
				{
					bestDistance_sIndex = d;
					hit.m_geometryIndex = rMap.m_geometryIndex;
					hit.m_indexOffset = 0;
					hit.m_triangleIndex = rMap.m_triangleIndex;
					hit.m_flippedWinding = false;
				}
			}
			{
				const hkSimdReal d = oneThird * (r0.distanceTo( s1 ) + r1.distanceTo( s2 ) + r2.distanceTo( s0 ));
				if ( d.isLess(bestDistance_sIndex) )
				{
					bestDistance_sIndex = d;
					hit.m_geometryIndex = rMap.m_geometryIndex;
					hit.m_indexOffset = 2;
					hit.m_triangleIndex = rMap.m_triangleIndex;
					hit.m_flippedWinding = false;
				}
			}
			{
				const hkSimdReal d = oneThird * (r0.distanceTo( s2 ) + r1.distanceTo( s0 ) + r2.distanceTo( s1 ));
				if ( d.isLess(bestDistance_sIndex) )
				{
					bestDistance_sIndex = d;
					hit.m_geometryIndex = rMap.m_geometryIndex;
					hit.m_indexOffset = 1;
					hit.m_triangleIndex = rMap.m_triangleIndex;
					hit.m_flippedWinding = false;
				}
			}

			//////////////////////////
			// Anti-cyclic
			//////////////////////////
			{
				const hkSimdReal d = oneThird * (r0.distanceTo( s0 ) + r1.distanceTo( s2 ) + r2.distanceTo( s1 ));
				if ( d.isLess(bestDistance_sIndex) )
				{
					bestDistance_sIndex = d;
					hit.m_geometryIndex = rMap.m_geometryIndex;
					hit.m_indexOffset = 0;
					hit.m_triangleIndex = rMap.m_triangleIndex;
					hit.m_flippedWinding = true;
				}
			}
			{
				const hkSimdReal d = oneThird * (r0.distanceTo( s1 ) + r1.distanceTo( s0 ) + r2.distanceTo( s2 ));
				if ( d.isLess(bestDistance_sIndex) )
				{
					bestDistance_sIndex = d;
					hit.m_geometryIndex = rMap.m_geometryIndex;
					hit.m_indexOffset = 1;
					hit.m_triangleIndex = rMap.m_triangleIndex;
					hit.m_flippedWinding = true;
				}
			}
			{
				const hkSimdReal d = oneThird * (r0.distanceTo( s2 ) + r1.distanceTo( s1 ) + r2.distanceTo( s0 ));
				if ( d.isLess(bestDistance_sIndex) )
				{
					bestDistance_sIndex = d;
					hit.m_geometryIndex = rMap.m_geometryIndex;
					hit.m_indexOffset = 2;
					hit.m_triangleIndex = rMap.m_triangleIndex;
					hit.m_flippedWinding = true;
				}
			}

			bestDistance_sIndex.store<1>(&bestDistance[sIndex]);
		}
	}
}

//
// Full match
//

void HK_CALL hkGeometryMatchingUtils::matchGeometries ( const hkArray<Geometry>& referenceGeometries, const hkArray<Geometry>& searchGeometries, hkReal maxDistance, FullMap& fullMapOut)
{
	// First do a map between search vertices and search triangles
	{
		int totalNumSearchVertices = 0;
		for (int sg=0; sg<searchGeometries.getSize(); sg++)
		{
			const Geometry& sGeom = searchGeometries[sg];
			fullMapOut.m_startEntryPerGeometry.pushBack(totalNumSearchVertices);

			totalNumSearchVertices += sGeom.m_numVertices;
		}

		fullMapOut.m_searchTrianglePerSearchVertex.setSize (totalNumSearchVertices);

		for (int sg=0; sg<searchGeometries.getSize(); sg++)
		{
			const Geometry& sGeom = searchGeometries[sg];

			for (int tri=0; tri<sGeom.m_numTriangles; tri++)
			{
				for (hkUint8 e=0; e<3; ++e)
				{
					int vIndex = *( hkAddByteOffsetConst<int> ( sGeom.m_triangleIndices,  tri*sGeom.m_triangleIndexStride) + e );

					hkUint32 entryIndex = fullMapOut.m_startEntryPerGeometry[sg] + vIndex;
					FullMap::VertexTriangleEntry& vtEntry = fullMapOut.m_searchTrianglePerSearchVertex [entryIndex];
					vtEntry.m_triangleIndex = tri;
					vtEntry.m_trianglePos = e;
				}
			}
		}
	}

	// Then build the triangle map
	matchTriangles (referenceGeometries, searchGeometries, maxDistance, fullMapOut.m_triangleMap);
}

/*
 * Havok SDK - Base file, BUILD(#20130912)
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
