/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h>
#include <Common/GeometryUtilities/Misc/hkGeometryUtils.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Common/Base/Math/Vector/hkIntVector.h>

template<typename T>
inline void hkGeometryUtils::Node::Triangle::_sort3(T& a, T& b, T& c)
{
	if (b < a) hkAlgorithm::swap(b, a);
	if (c < b) hkAlgorithm::swap(c, b);
	if (b < a) hkAlgorithm::swap(b, a);
}

hkGeometryUtils::Node::Triangle::Triangle(hkUint32 a, hkUint32 b, hkUint32 c)
{
	m_indices[0] = a; m_sortedIndices[0] = a;
	m_indices[1] = b; m_sortedIndices[1] = b;
	m_indices[2] = c; m_sortedIndices[2] = c;
	_sort3(m_sortedIndices[0], m_sortedIndices[1], m_sortedIndices[2]);
}

hkGeometryUtils::Node::Node( hkUint32 vertexIndex ) : m_vertexIndex(vertexIndex) {}

hkGeometryUtils::Node::Node(const hkGeometryUtils::Node& other)
{
	m_vertexIndex = other.m_vertexIndex;
	m_edges = other.m_edges;
}

hkGeometryUtils::Node& hkGeometryUtils::Node::operator= (const hkGeometryUtils::Node& other) 
{ 
	m_vertexIndex = other.m_vertexIndex;
	m_edges = other.m_edges;
	return *this; 
}

hkGeometryUtils::Node::Edge::Edge(hkUint32 endpointIndex, const Triangle& triangle, hkUint32 triangleIndex) : 

m_endpointIndex(endpointIndex), 
m_numIncoming(0),
m_numOutgoing(0),
m_nonManifold(false),
m_inconsistentWinding(false),
m_processed(false)

{ 
	m_triangles.pushBack(triangle);
	m_triangleIndices.pushBack(triangleIndex);
}

hkGeometryUtils::Node::Edge::Edge(const Edge& other)
{
	m_endpointIndex = other.m_endpointIndex;
	m_triangles = other.m_triangles;
	m_triangleIndices = other.m_triangleIndices;
	m_numIncoming = other.m_numIncoming;
	m_numOutgoing = other.m_numOutgoing;
	m_nonManifold = other.m_nonManifold;
	m_inconsistentWinding = other.m_inconsistentWinding;
	m_processed = other.m_processed;
}

hkGeometryUtils::Node::Edge& hkGeometryUtils::Node::Edge::operator= (const hkGeometryUtils::Node::Edge& other) 
{ 
	m_endpointIndex = other.m_endpointIndex;
	m_triangles = other.m_triangles;
	m_triangleIndices = other.m_triangleIndices;
	m_numIncoming = other.m_numIncoming;
	m_numOutgoing = other.m_numOutgoing;
	m_nonManifold = other.m_nonManifold;
	m_inconsistentWinding = other.m_inconsistentWinding;
	m_processed = other.m_processed;
	return *this; 
}

bool hkGeometryUtils::Node::Edge::hasTriangleSameWinding(const Triangle& triangle, int& triIndex)
{
	triIndex = -1;
	for (int t=0; t<m_triangles.getSize(); ++t)
	{
		if ( ( triangle.m_indices[0] == m_triangles[t].m_indices[0] ) &&
			 ( triangle.m_indices[1] == m_triangles[t].m_indices[1] ) &&
			 ( triangle.m_indices[2] == m_triangles[t].m_indices[2] ) )
		{
			triIndex = (int)m_triangleIndices[t];
			return true;
		}
	}
	return false;
}

bool hkGeometryUtils::Node::Edge::hasTriangleIgnoreWinding(const Triangle& triangle, int& triIndex)
{
	triIndex = -1;
	for (int t=0; t<m_triangles.getSize(); ++t)
	{
		if ( ( triangle.m_sortedIndices[0] == m_triangles[t].m_sortedIndices[0] ) &&
			 ( triangle.m_sortedIndices[1] == m_triangles[t].m_sortedIndices[1] ) &&
			 ( triangle.m_sortedIndices[2] == m_triangles[t].m_sortedIndices[2] ) )
		{
			triIndex = (int)m_triangleIndices[t];
			return true;
		}
	}
	return false;
}

hkGeometryUtils::Node::Edge* hkGeometryUtils::Node::findEdge(hkUint32 endpointIndex)
{
	for (int e=0; e<m_edges.getSize(); ++e)
	{
		if (m_edges[e].m_endpointIndex == endpointIndex) return &m_edges[e];
	}
	return HK_NULL;
}

void hkGeometryUtils::Node::addEdge(hkUint32 endpointIndex, const Triangle& triangle, hkUint32 triangleIndex, bool incoming)
{
	hkGeometryUtils::Node::Edge* edge = findEdge(endpointIndex);
	if (edge)
	{
		if (incoming) edge->m_numIncoming++;
		else edge->m_numOutgoing++;

		// Flag edge as having inconsistent winding if it comes in or out of this node more than once
		if ( edge->m_numIncoming>1 || edge->m_numOutgoing>1) edge->m_inconsistentWinding = true;

		edge->m_triangleIndices.pushBack(triangleIndex); 
		edge->m_triangles.pushBack(triangle);

		// Flag edge as non-manifold if more than 2 triangles share it
		if (edge->m_triangleIndices.getSize()>2) edge->m_nonManifold = true;
	}
	else
	{
		hkGeometryUtils::Node::Edge e(endpointIndex, triangle, triangleIndex);
		if (incoming) e.m_numIncoming++;
		else e.m_numOutgoing++;
		m_edges.pushBack(e);
	}
}

bool hkGeometryUtils::Node::checkForNonManifoldGeometry() const
{
	for (int ei=0; ei<m_edges.getSize(); ++ei)
	{
		const hkGeometryUtils::Node::Edge& edge = m_edges[ei];
		if (edge.m_nonManifold) return false;
	}
	return true;
}

void hkGeometryUtils::Node::warnAboutInconsistentWinding(int e) const
{
	const hkGeometryUtils::Node::Edge& edge = m_edges[e];
	if (edge.m_inconsistentWinding && edge.m_triangles.getSize()>1)
	{
		HK_WARN_ALWAYS(0xabba1daf, "Edge (" << edge.m_endpointIndex << "," << m_vertexIndex << ") has inconsistent winding in triangles " << edge.m_triangleIndices[0] << "and" << edge.m_triangleIndices[1] << ".");
	}
}


/******************************   hkGeometryUtils::weldVertices  ******************************/
namespace weldVerticesVirtualNs
{
	struct VertexRef
	{
		HK_FORCE_INLINE	bool operator<(const VertexRef& other) const { return m_value < other.m_value; }
		hkSimdReal	m_value;
		int			m_index;
	};
}

using namespace weldVerticesVirtualNs;

int HK_CALL hkGeometryUtils::weldVerticesVirtual (const IVertices* vertices, hkArray<int>& remap, hkReal thr)
{	
	const hkSimdReal threshold = hkSimdReal::fromFloat(thr);
	hkSimdReal maxDistanceSquared; maxDistanceSquared.setMul(threshold,threshold);
	const int			numVertices = vertices->getNumVertices();
	hkArray<VertexRef>	sortedVertices(numVertices);
	int					numUnique = 0;
	remap.setSize(numVertices);
	for (int i=0; i<numVertices; i++)
	{
		hkVector4	x; vertices->getVertex(i, x);
		sortedVertices[i].m_index	=	i;
		sortedVertices[i].m_value	=	x.getComponent<0>();
	}
	
	hkSort(sortedVertices.begin(), numVertices);

	for(int i=0; i<numVertices; ++i)
	{
		VertexRef&	vri = sortedVertices[i];
		
		if(vri.m_index < 0) continue;

		remap[vri.m_index]	=	vri.m_index;

		hkVector4	xi; vertices->getVertex(vri.m_index,xi);
		for(int j=i+1; j<numVertices; ++j)
		{
			VertexRef&	vrj = sortedVertices[j];
			
			if(vrj.m_index < 0) continue;
			if((vrj.m_value - vri.m_value) > threshold) break;
			
			hkVector4	xj; vertices->getVertex(vrj.m_index,xj);
			if(xi.distanceToSquared(xj).isLessEqual(maxDistanceSquared))
			{
				if(vertices->isWeldingAllowed(vrj.m_index, vri.m_index))
				{
					remap[vrj.m_index]	=	vri.m_index;
					vrj.m_index			=	-1;
				}
			}
		}
		vri.m_index	= -1;
		++numUnique;
	}

	return numUnique;
}

void HK_CALL hkGeometryUtils::weldVertices( struct hkGeometry& meshGeometry, hkReal threshold )
{
	hkArray<int> vertexRemap;
	hkArray<int> triangleRemap;
	hkArray<hkVector4> uniqueVerts;
	hkArray<hk1AxisSweep::AabbInt> sweepAabbs( meshGeometry.m_vertices.getSize() + 4 );
	hkArray<hkRadixSort::SortData32> sortData( meshGeometry.m_vertices.getSize() + 4 );
	hkArray<hk1AxisSweep::AabbInt> sortedAabbs( meshGeometry.m_vertices.getSize() + 4 );
	_weldVertices( meshGeometry, threshold, false, vertexRemap, triangleRemap, uniqueVerts, sweepAabbs, sortData, sortedAabbs );
}

void HK_CALL hkGeometryUtils::weldVertices( struct hkGeometry& meshGeometry, hkReal threshold, hkBool keepVertexOrder, hkArray<int>& vertexRemapOut, hkArray<int>& triangleRemapOut )
{
	hkArray<hkVector4> uniqueVerts;
	hkArray<hk1AxisSweep::AabbInt> sweepAabbs( meshGeometry.m_vertices.getSize() + 4 );
	hkArray<hkRadixSort::SortData32> sortData( meshGeometry.m_vertices.getSize() + 4 );
	hkArray<hk1AxisSweep::AabbInt> sortedAabbs( meshGeometry.m_vertices.getSize() + 4 );
	_weldVertices(meshGeometry, threshold, keepVertexOrder, vertexRemapOut, triangleRemapOut, uniqueVerts, sweepAabbs, sortData, sortedAabbs );
}

namespace 
{
	// Weld the vertices inplace returning the number of new verts
	void hkWeldVertices( const hkArray<hkVector4>& vertsIn, hkArray<hkVector4>& uniqueVerts, hkArray<int>& remap, hkArrayBase<hk1AxisSweep::AabbInt>& vertAabbs, hkArrayBase<hkRadixSort::SortData32>& sortData,hkArrayBase<hk1AxisSweep::AabbInt>& tempAabbs, hkReal tol )
	{
		const hkSimdReal tolerance = hkSimdReal::fromFloat(tol);
		const int numVerts = vertsIn.getSize();
		remap.setSize( numVerts );

		// Create AABBs for each vertex
		HK_ASSERT(0x4be7d0ae, vertAabbs.getCapacity() >= numVerts + 4);
		{
			const hkSimdReal halfTol = tolerance * hkSimdReal_Half;
			vertAabbs.setSizeUnchecked(numVerts + 4);
			for (int v = 0; v < numVerts; v++)
			{
				hkAabb aabb;
				aabb.m_min.setSub( vertsIn[v], halfTol );
				aabb.m_max.setAdd( vertsIn[v], halfTol );
				vertAabbs[v].set(aabb, v);
			}

			hk1AxisSweep::AabbInt terminator; terminator.setEndMarker();
			// I can do this because I made space for 4 terminators above
			hk1AxisSweep::AabbInt* end = vertAabbs.begin() + numVerts;
			end[0] = terminator;
			end[1] = terminator;
			end[2] = terminator;
			end[3] = terminator;
		}

		// Sort
		hk1AxisSweep::sortAabbs( vertAabbs.begin(), numVerts, sortData, tempAabbs );

		// Find all unique verts
		hkSimdReal tolSqrd; tolSqrd.setMul(tolerance, tolerance);
		for (int current=0; current < numVerts; current++) 
		{
			const hkUint32 currentKey = vertAabbs[current].getKey();

			// This vertex is a duplicate
			if (currentKey == 0xFFFFFFFF)
				continue;

			// Store this vertex it is unique
			const hkVector4& currentPos = vertsIn[ currentKey ];
			remap[ currentKey ] = uniqueVerts.getSize();
			uniqueVerts.pushBack( currentPos );

			// For each overlapping AABB
			for ( int potential = current+1;
				  (potential < numVerts) && ( vertAabbs[potential].m_min[0] <= vertAabbs[current].m_max[0] );
				  potential++ )
			{
				const hkUint32 potentialKey = vertAabbs[potential].getKey();

				// Skip duplicates
				if (potentialKey == 0xFFFFFFFF)
					continue;

				const hkVector4& potentialPos = vertsIn[ potentialKey ];

				if ( currentPos.distanceToSquared( potentialPos ).isLessEqual(tolSqrd) )
				{
					// Mark vertex as a duplicate
					remap[ potentialKey ] = remap[ currentKey ];
					vertAabbs[potential].getKey() = 0xFFFFFFFF;
				}
			}
		} 
	}

	bool hkTriangle_isDegenerate( const hkGeometry::Triangle& a )
	{
		return (a.m_a == a.m_b) || (a.m_a == a.m_c) || (a.m_b == a.m_c);		   
	}
}

static inline bool _sameTriangleIgnoringWinding(const hkGeometry::Triangle& triA, const hkGeometry::Triangle& triB)
{

	{
		// Some assumptions we are making re. hkGeometryTriangle
		HK_ASSERT(0xd52ff74, hkUlong(&(triA.m_a)) == hkUlong(&triA));
		HK_ASSERT(0x64b63cb4, hkUlong(&(triA.m_b))-hkUlong(&(triA.m_a)) == hkUlong(sizeof(int)));
	}

	const int* tA = reinterpret_cast<const int*> (&triA);
	const int* tB = reinterpret_cast<const int*> (&triB);

	// We assume neither triangle is degenerate
	for (int i=0; i<3; ++i)
	{
		bool found = false;
		for (int j=0; j<3; ++j)
		{
			if (tA[i]==tB[j]) found = true;
		}
		if (!found) return false;
	}

	return true;
}

static inline bool _sameTriangleConsideringWinding(const hkGeometry::Triangle& triA, const hkGeometry::Triangle& triB)
{

	{
		// Some assumptions we are making re. hkGeometryTriangle
		HK_ASSERT(0xd52ff74, hkUlong(&(triA.m_a)) == hkUlong(&triA));
		HK_ASSERT(0x64b63cb4, hkUlong(&(triA.m_b))-hkUlong(&(triA.m_a)) == hkUlong(sizeof(int)));
	}

	const int* tA = reinterpret_cast<const int*> (&triA);
	const int* tB = reinterpret_cast<const int*> (&triB);

	// If triangle A is {0, 1, 2}, then any of
	//		{0, 1, 2}
	//		{1, 2, 0}
	//		{2, 0, 1}
	// is considered the "same" triangle
	return ((tA[0] == tB[0]) && (tA[1] == tB[1]) && (tA[2] == tB[2]))
		|| ((tA[0] == tB[1]) && (tA[1] == tB[2]) && (tA[2] == tB[0]))
		|| ((tA[0] == tB[2]) && (tA[1] == tB[0]) && (tA[2] == tB[1])) ;
}

hkUint64 hashTriangle( const hkGeometry::Triangle& t1 )
{
	const int mask = (1<<21) - 1;
	hkUint64 x = t1.m_a & (mask);
	hkUint64 y = t1.m_b & (mask);
	hkUint64 z = t1.m_c & (mask);

	return (x<<42) | (y<<21) | z ;
}

struct triInfo
{
	hkUint64 m_key;
	hkInt32 m_index;

};

struct sortByKey
{
public:

	HK_FORCE_INLINE hkBool32 operator() ( const triInfo& t1, const triInfo& t2 )
	{
		// Sort by key first, then index is the tie-breaker
		return (t1.m_key < t2.m_key) || ((t1.m_key == t2.m_key) && (t1.m_index < t2.m_index));
	}
};

struct sortByIndex
{
public:

	HK_FORCE_INLINE hkBool32 operator() ( const triInfo& t1, const triInfo& t2 )
	{
		return t1.m_index < t2.m_index;
	}
};

hkResult HK_CALL hkGeometryUtils::removeDuplicateTrianglesFast(struct hkGeometry& meshGeometry, hkArray<hkGeometry::Triangle>& newTriangles )
{
	int size = meshGeometry.m_triangles.getSize();
	if (size == 0)
	{
		return HK_SUCCESS;
	}

	hkArray<triInfo>::Temp sortArray(size);
	if (sortArray.begin() == HK_NULL)
	{
		return HK_FAILURE;
	}
	
	// Go through the triangles, sort the indices, and overwrite the material with the actual index
	for (int t=0; t<meshGeometry.m_triangles.getSize(); t++)
	{
		// Bubble sort!
		hkGeometry::Triangle triangle = meshGeometry.m_triangles[t];
		if (triangle.m_a > triangle.m_b)	{ hkAlgorithm::swap( triangle.m_a, triangle.m_b ); }
		if (triangle.m_a > triangle.m_c)	{ hkAlgorithm::swap( triangle.m_a, triangle.m_c ); }
		if (triangle.m_b > triangle.m_c)	{ hkAlgorithm::swap( triangle.m_b, triangle.m_c ); }

		HK_ASSERT(0x53429a52, triangle.m_a <= triangle.m_b);
		HK_ASSERT(0x53429a52, triangle.m_a <= triangle.m_c);
		HK_ASSERT(0x53429a52, triangle.m_b <= triangle.m_c);

		triInfo& entry = sortArray[t];

		entry.m_key = hashTriangle(triangle);
		entry.m_index = t;
	}

	// Sort by key. This will clump potentially identical triangles together
	hkSort( sortArray.begin(), sortArray.getSize(), sortByKey() );

	// Walk backwards, removing duplicates
	for (int t=sortArray.getSize()-1; t>=1; t--)
	{
		const hkUint64 key1 = sortArray[t].m_key;

		int lastMatch = -1;
		// Find the first triangle that matches this one
		for (int t2 = t-1; t2>=0 && key1 == sortArray[t2].m_key; t2--)
		{
			hkGeometry::Triangle& tri1 = meshGeometry.m_triangles[ sortArray[t].m_index ];
			hkGeometry::Triangle& tri2 = meshGeometry.m_triangles[ sortArray[t2].m_index ];

			if ( _sameTriangleConsideringWinding( tri1, tri2 ) )
			{
				lastMatch = t2;
			}
		}

		// If we found a match, remove it
		if (lastMatch != -1)
		{
			sortArray.removeAt(t);
		}
	}

	// Resort by original index
	hkSort( sortArray.begin(), sortArray.getSize(), sortByIndex() );

	newTriangles.clear();

	for (int t=0; t<sortArray.getSize(); t++)
	{
		int originalIndex = sortArray[t].m_index;
		newTriangles.pushBack( meshGeometry.m_triangles[ originalIndex ] );
	}

	meshGeometry.m_triangles = newTriangles;
	return HK_SUCCESS;
}


void HK_CALL hkGeometryUtils::_weldVertices( struct hkGeometry& meshGeometry, hkReal threshold, hkBool keepVertexOrder, hkArray<int>& vertexRemapOut, hkArray<int>& triangleRemapOut, hkArray<hkVector4>& uniqueVerts, hkArrayBase<hk1AxisSweep::AabbInt>& sweepAabbs, hkArrayBase<hkRadixSort::SortData32>& sortData, hkArrayBase<hk1AxisSweep::AabbInt>& sortedAabbs )
{
	const int originalNumVertices = meshGeometry.m_vertices.getSize();
	const int originalNumTriangles = meshGeometry.m_triangles.getSize();

	// Delete any duplicate vertices, building the map from old to new indices
	{
		// Initialize the index map
		vertexRemapOut.setSize( originalNumVertices );
		hkWeldVertices( meshGeometry.m_vertices, uniqueVerts, vertexRemapOut, sweepAabbs, sortData, sortedAabbs, threshold );

		if (keepVertexOrder)
		{
			// HCL-1174 
			hkArray< hkVector4 > orderedVerts; orderedVerts.reserve(uniqueVerts.getSize());
			hkArray< int > extraMap;
			extraMap.setSize(uniqueVerts.getSize(),-1);

			for (int i=0; i<vertexRemapOut.getSize(); ++i)
			{
				const int originalIndex = i;
				const int remapedIndex = vertexRemapOut[i];
				int newIndex = extraMap [remapedIndex];
				if (newIndex == -1) // new vert
				{
					newIndex = orderedVerts.getSize();
					extraMap [remapedIndex] = newIndex;
					orderedVerts.pushBack(uniqueVerts[remapedIndex]);
				}
				vertexRemapOut[originalIndex] = newIndex;
			}

			HK_ASSERT(0x36b08746, orderedVerts.getSize() == uniqueVerts.getSize());

			meshGeometry.m_vertices = orderedVerts;
		}
		else
		{
			meshGeometry.m_vertices = uniqueVerts;
		}
	}

	// Remap each triangle's vertex indices
	{
		for( int i=0; i < originalNumTriangles; ++i )
		{
			hkGeometry::Triangle& triangle = meshGeometry.m_triangles[i];
			triangle.m_a = vertexRemapOut[triangle.m_a];
			triangle.m_b = vertexRemapOut[triangle.m_b];
			triangle.m_c = vertexRemapOut[triangle.m_c];
		}
	}

	triangleRemapOut.setSize( originalNumTriangles );

	// Copy down nondegenerate triangles in place
	{
		hkGeometry::Triangle* nonDegenerate = meshGeometry.m_triangles.begin();
		for( int i=0; i < originalNumTriangles; ++i )
		{
			const hkGeometry::Triangle& triangle = meshGeometry.m_triangles[i];

			int remappedIndex = -1;	// -1 implies the triangle is degenerate

			if ( !hkTriangle_isDegenerate( triangle ) )
			{
				// Copy if not degenerate
				remappedIndex = (int) ( nonDegenerate - meshGeometry.m_triangles.begin() );
				*nonDegenerate++ = triangle;
			}

			triangleRemapOut[i] = remappedIndex;

		}
		meshGeometry.m_triangles.setSize( (int) ( nonDegenerate - meshGeometry.m_triangles.begin() ) );
	}
}



/******************************   hkGeometryUtils::removeDuplicateTriangles  ******************************/


void HK_CALL hkGeometryUtils::removeDuplicateTriangles (struct hkGeometry& meshGeometry, hkArray<int>& triangleMapOut, bool ignoreWinding)
{
	hkArray<hkGeometry::Triangle> newTriangles;
	hkArray<Node>::Temp nodes;
	
	for (hkUint32 vi=0; vi<(hkUint32)meshGeometry.m_vertices.getSize(); ++vi)
	{
		Node node(vi);
		nodes.pushBack(node);
	}
	
	int numDuplicates = 0;

	if (ignoreWinding)
	{
		for (hkUint32 ti=0; ti<(hkUint32)meshGeometry.m_triangles.getSize(); ++ti)
		{
			hkGeometry::Triangle& geomTriangle = meshGeometry.m_triangles[ti];
			hkGeometryUtils::Node::Triangle tri(geomTriangle.m_a, geomTriangle.m_b, geomTriangle.m_c);

			Node::Edge* AB = nodes[geomTriangle.m_a].findEdge(geomTriangle.m_b);
			Node::Edge* BC = nodes[geomTriangle.m_b].findEdge(geomTriangle.m_c);
			Node::Edge* CA = nodes[geomTriangle.m_c].findEdge(geomTriangle.m_a);

			int triIndex;
			if ( (AB && AB->hasTriangleIgnoreWinding(tri, triIndex)) || 
				 (BC && BC->hasTriangleIgnoreWinding(tri, triIndex)) || 
				 (CA && CA->hasTriangleIgnoreWinding(tri, triIndex)) )
			{
				HK_ASSERT(0x5c6b5be9, triIndex>=0);
				triangleMapOut.pushBack(triIndex);
				numDuplicates++;
			}
			else
			{
				int indices[3] = {geomTriangle.m_a, geomTriangle.m_b, geomTriangle.m_c};
				int newTriangleIndex = newTriangles.getSize();

				Node::Triangle triangle(geomTriangle.m_a, geomTriangle.m_b, geomTriangle.m_c);

				for (int a=0; a<3; ++a)
				{
					int b = (a+1) % 3;
					hkUint32 va = indices[a];
					hkUint32 vb = indices[b];

					nodes[va].addEdge(vb, triangle, newTriangleIndex, false);
					nodes[vb].addEdge(va, triangle, newTriangleIndex, true);
				}

				triangleMapOut.pushBack(newTriangleIndex);
				newTriangles.pushBack(meshGeometry.m_triangles[ti]);
			}
		}
	}

	else
	{
		for (hkUint32 ti=0; ti<(hkUint32)meshGeometry.m_triangles.getSize(); ++ti)
		{
			hkGeometry::Triangle& geomTriangle = meshGeometry.m_triangles[ti];
			hkGeometryUtils::Node::Triangle tri(geomTriangle.m_a, geomTriangle.m_b, geomTriangle.m_c);

			Node::Edge* AB = nodes[geomTriangle.m_a].findEdge(geomTriangle.m_b);
			Node::Edge* BC = nodes[geomTriangle.m_b].findEdge(geomTriangle.m_c);
			Node::Edge* CA = nodes[geomTriangle.m_c].findEdge(geomTriangle.m_a);

			int triIndex;
			if ( (AB && AB->hasTriangleSameWinding(tri, triIndex)) || 
				 (BC && BC->hasTriangleSameWinding(tri, triIndex)) || 
				 (CA && CA->hasTriangleSameWinding(tri, triIndex)) )
			{
				HK_ASSERT(0x3aaf0b78, triIndex>=0);
				triangleMapOut.pushBack(triIndex);
				numDuplicates++;
			}
			else
			{
				int indices[3] = {geomTriangle.m_a, geomTriangle.m_b, geomTriangle.m_c};
				int newTriangleIndex = newTriangles.getSize();

				Node::Triangle triangle(geomTriangle.m_a, geomTriangle.m_b, geomTriangle.m_c);

				for (int a=0; a<3; ++a)
				{
					int b = (a+1) % 3;
					hkUint32 va = indices[a];
					hkUint32 vb = indices[b];

					nodes[va].addEdge(vb, triangle, newTriangleIndex, false);
					nodes[vb].addEdge(va, triangle, newTriangleIndex, true);
				}

				triangleMapOut.pushBack(newTriangleIndex);
				newTriangles.pushBack(meshGeometry.m_triangles[ti]);
			}
		}
	}

	if (numDuplicates>0)
	{
		HK_REPORT("Removed " << numDuplicates << " duplicate triangles out of a total of " << meshGeometry.m_triangles.getSize() << " triangles.");
	}
	
	meshGeometry.m_triangles = newTriangles;
}


/******************************   debug display geometry  ******************************/

void HK_CALL hkGeometryUtils::createCapsuleGeometry(const hkVector4& start, const hkVector4& end, hkReal radius, int heightSamples, int thetaSamples, const hkTransform& transform, hkGeometry& geometryOut)
{

	const int phiSamples = thetaSamples >> 1;

	hkArray<hkVector4> verts;

		// Create "transform" from start, end.
	hkTransform capsuleToLocal;

	const hkVector4& zAxis = hkVector4::getConstant<HK_QUADREAL_0010>();
	hkVector4 axis;
	axis.setSub(end, start);
	hkSimdReal height = axis.length<3>();
	if(height.isGreaterZero())
	{
		axis.normalize<3>();

		hkSimdReal axisDot; axisDot.setAbs(axis.dot<3>(zAxis));
		if(axisDot < hkSimdReal::fromFloat(1.0f - 1e-5f))
		{
			hkRotation rotation;
			hkVector4& col0 = rotation.getColumn(0);
			hkVector4& col1 = rotation.getColumn(1);
			hkVector4& col2 = rotation.getColumn(2);

			col2 = axis;
			col1.setCross( col2, zAxis); 
			col1.normalize<3>();
			col0.setCross( col1, col2 );
			capsuleToLocal.setRotation(rotation);
		}
		else
		{
			capsuleToLocal.setIdentity();	
		}

	}
	else
	{
		capsuleToLocal.setIdentity();
	}

			// Now recenter
	{
		hkVector4 toCentre;
		toCentre.setAdd(start, end);
		toCentre.mul(hkSimdReal_Half);
		capsuleToLocal.setTranslation(toCentre);
	}

	// We'll sweep around the axis of the deflector, from top to bottom, using the original
	// sample directions and data to define the vertices. We'll tessellate in the obvious way.
	// N.B. Top and bottom vertices are added to cap the object. 

	int i,j;

	hkVector4 vert;

	hkVector4 bottomInGeom; bottomInGeom.setMul(zAxis,-height*hkSimdReal_Half);
	hkVector4 topInGeom; topInGeom.setNeg<4>(bottomInGeom);
	hkVector4 axisInGeom;
	axisInGeom.setSub(topInGeom, bottomInGeom);
	hkVector4 axisNInGeom = zAxis;
	hkVector4 normalInGeom = hkVector4::getConstant<HK_QUADREAL_1000>();
	hkVector4 binormalInGeom; binormalInGeom.setMul(hkVector4::getConstant<HK_QUADREAL_0100>(),hkSimdReal_Minus1);

	// top capsule = phiSamples (segments) = phiSamples + 1 vert rings but top ring is vert so phiSamples * thetaSamples + 1 verts
	// This contains the top ring of the cylinder, top cap = phiSamples rings of faces
	// bottom capsule = phiSamples (segments) = phiSamples + 1 rings but bottom ring is vert so phiSamples * thetaSamples + 1 verts
	// This contains the bottom ring of the cylinder, bottom cap = phiSamples rings of faces
	// cylinder body = heightSamples (segments) = heightSamples + 1 vert rings but bottom and top caps already create 2 so (heightSamples -1) * thetaSamples verts
	// cylinder body = heightSamples rings of faces
	// total number of face rings = 2 * phiSamples + heightSamples
	verts.reserveExactly(2 * phiSamples * thetaSamples + 2 + (heightSamples-1) * thetaSamples);

	//
	// GET TOP VERTICES
	//
	const hkSimdReal radiusSr = hkSimdReal::fromFloat(radius);

	vert.setMul(zAxis,height*hkSimdReal_Half+radiusSr);
	vert._setTransformedPos(capsuleToLocal, vert);
	verts.pushBack(vert);

	const hkSimdReal invPhiSamples = hkSimdReal::fromInt32(phiSamples).reciprocal() * hkSimdReal_PiOver2;
	const hkSimdReal invThetaSamples = hkSimdReal::fromInt32(thetaSamples).reciprocal() * hkSimdReal_TwoPi;
	for (i = phiSamples-1 ; i >= 0; i--)
	{
		hkQuaternion qTop; qTop.setAxisAngle(binormalInGeom, hkSimdReal::fromInt32(i) * invPhiSamples);
		hkVector4 topElevation;
		topElevation._setRotatedDir(qTop, normalInGeom);

		for (j = 0; j < thetaSamples; j++)
		{
			hkQuaternion rotationTop; rotationTop.setAxisAngle(axisNInGeom, hkSimdReal::fromInt32(j) * invThetaSamples);			
			hkVector4 topDirection;
			topDirection._setRotatedDir(rotationTop, topElevation);

			vert.setAddMul(topInGeom, topDirection, radiusSr);

			vert._setTransformedPos(capsuleToLocal, vert);

			//push back the rest of the vertices
			verts.pushBack(vert);

		}
	}

	//
	// GET MIDDLE VERTICES
	//
	const hkSimdReal invHeightSamples = hkSimdReal::fromInt32(heightSamples).reciprocal();
	for (j = heightSamples-1; j > 0; j--)
	{
	
		for (i = 0; i < thetaSamples; i++)
		{	
		//
		// Calculate direction vector for this angle
		//

			hkQuaternion q; q.setAxisAngle(axisNInGeom, hkSimdReal::fromInt32(i) * invThetaSamples);
			hkVector4 direction;
			direction._setRotatedDir(q, normalInGeom);
			
			hkVector4 s;
			s.setAddMul(bottomInGeom, axisInGeom, hkSimdReal::fromInt32(j) * invHeightSamples);

			vert.setAddMul(s, direction, radiusSr);

			vert._setTransformedPos(capsuleToLocal, vert);

			verts.pushBack(vert);

		}
	}

	//
	// GET BOTTOM VERTICES
	//
	for (i = 0; i < phiSamples; i++)
	{
		hkQuaternion qBottom; qBottom.setAxisAngle(binormalInGeom, -hkSimdReal::fromInt32(i) * invPhiSamples);
		hkVector4 bottomElevation;
		bottomElevation._setRotatedDir(qBottom, normalInGeom);

		for (j = 0; j < thetaSamples; j++)
		{
			hkQuaternion rotationBottom; rotationBottom.setAxisAngle(axisNInGeom, hkSimdReal::fromInt32(j) * invThetaSamples);			
			hkVector4 bottomDirection;
			bottomDirection._setRotatedDir(rotationBottom, bottomElevation);

			vert.setAddMul(bottomInGeom, bottomDirection, radiusSr);
			vert._setTransformedPos(capsuleToLocal, vert);
			verts.pushBack(vert);
		}
	}

	vert.setMul(zAxis,-(height*hkSimdReal_Half+radiusSr));
	vert._setTransformedPos(capsuleToLocal, vert);

		// Push back bottom vertex
	verts.pushBack(vert);

	// Transform all the points by m_transform.
	// TODO: take all these calcs out of these functions, and put into graphics handler
	// currently what comes out is wrong - i.e. a different graphics handler might assume
	// the transform should be taken into account, so it would be doubly counted.
	{
		for ( int vi = 0; vi < verts.getSize(); ++vi )
		{
			verts[vi]._setTransformedPos(transform, verts[vi]);
		}
	}

	//
	// CONSTRUCT FACE DATA
	//

	// Right, num samples AROUND axis is thetaSamples.

	// First off, we have thetaSamples worth of faces connected to the top
	hkGeometry::Triangle tr;
	hkArray<hkGeometry::Triangle> tris;
	tr.m_material=-1;

	int currentBaseIndex = 1;
	for (i = 0; i < thetaSamples; i++)
	{
		tr.m_a = 0;
		tr.m_b = currentBaseIndex + i;
		tr.m_c = currentBaseIndex + (i+1)%(thetaSamples);

		tris.pushBack(tr);
	}

	// Next we have phi-1 + height + phi-1 lots of thetaSamples*2 worth of faces connected to the previous row
	for(j = 0; j < 2*(phiSamples-1) + heightSamples; j++)
	{
		for (i = 0; i < thetaSamples; i++)
		{
			tr.m_a = currentBaseIndex + i;
			tr.m_b = currentBaseIndex + thetaSamples + i;
			tr.m_c = currentBaseIndex + thetaSamples + (i+1)%(thetaSamples);

			tris.pushBack(tr);

			tr.m_b = currentBaseIndex + i;
			tr.m_a = currentBaseIndex + (i+1)%(thetaSamples);
			tr.m_c = currentBaseIndex + thetaSamples + (i+1)%(thetaSamples);
		
			tris.pushBack(tr);

		}
		currentBaseIndex += thetaSamples;
	}

	// Finally, we have thetaSamples worth of faces connected to the bottom
	for (i = 0; i < thetaSamples; i++)
	{
		tr.m_b = currentBaseIndex + i;
		tr.m_a = currentBaseIndex + (i+1)%(thetaSamples);
		tr.m_c = currentBaseIndex + thetaSamples;

		tris.pushBack(tr);
	}

	geometryOut.m_vertices.swap( verts );
	geometryOut.m_triangles.swap( tris );
}


void HK_CALL hkGeometryUtils::createTaperedCapsuleGeometry(const hkVector4& start, const hkVector4& end, hkReal startRadius, hkReal endRadius, int heightSamples, int thetaSamples, const hkTransform& transform, hkGeometry& geometryOut)
{
	hkSimdReal bigRadius, smallRadius;
	hkVector4 big, small;

#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
	hkSimdReal endRadiusSr = hkSimdReal::fromFloat(endRadius);
	hkSimdReal startRadiusSr = hkSimdReal::fromFloat(startRadius);

	const hkVector4Comparison reverse = startRadiusSr.greater(endRadiusSr);
	small.setSelect(reverse, end, start);
	big.setSelect(reverse, start, end);
	smallRadius.setSelect(reverse, endRadiusSr, startRadiusSr);
	bigRadius.setSelect(reverse, startRadiusSr, endRadiusSr);
#else
	if (startRadius > endRadius)
	{
		big = start; 
		small = end; 
		bigRadius.setFromFloat(startRadius);
		smallRadius.setFromFloat(endRadius); 
	}
	else
	{
		big = end; 
		small = start; 
		bigRadius.setFromFloat(endRadius);
		smallRadius.setFromFloat(startRadius); 
	}
#endif
	const int phiSamples = thetaSamples >> 1;
	hkArray<hkVector4> verts;

	hkVector4 axis;
	axis.setSub(big, small);
	hkSimdReal height = axis.normalizeWithLength<3>() + hkSimdReal_Eps;

	const hkVector4& xAxis = hkVector4::getConstant<HK_QUADREAL_1000>();
	const hkVector4& yAxis = hkVector4::getConstant<HK_QUADREAL_0100>();
	const hkVector4& zAxis = hkVector4::getConstant<HK_QUADREAL_0010>();

	// Create "transform" from start, end.
	hkTransform capsuleToLocal;
	{
		if(height > (hkSimdReal_Eps+hkSimdReal_Eps))
		{
			axis.normalize<3>();

			hkSimdReal axisDot; axisDot.setAbs(axis.dot<3>(zAxis));
			if(axisDot < hkSimdReal::fromFloat(1.0f - 1e-5f))
			{
				hkRotation rotation;
				hkVector4& col0 = rotation.getColumn(0);
				hkVector4& col1 = rotation.getColumn(1);
				hkVector4& col2 = rotation.getColumn(2);

				col2 = axis;
				col1.setCross( col2, zAxis); 
				col1.normalize<3>();
				col0.setCross( col1, col2 );
				capsuleToLocal.setRotation(rotation);
			}
			else
			{
				if (axis.dot<3>(zAxis).isLessZero())
				{
					// Capsule oriented along -z
					hkRotation rotation;
					rotation.setAxisAngle(xAxis, hkSimdReal_Pi);
					capsuleToLocal.setRotation(rotation);
				}
				else
				{
					capsuleToLocal.setIdentity();	
				}
			}
		}
		else
		{
			capsuleToLocal.setIdentity();
		}

		hkVector4 center;
		center.setAdd(start, end);
		center.mul(hkSimdReal_Half);
		capsuleToLocal.setTranslation(center);
	}

	hkVector4 localBig;   localBig.setMul(zAxis, height*hkSimdReal_Half);
	hkVector4 localSmall; localSmall.setNeg<4>(localBig);

	verts.reserveExactly(2 * phiSamples * thetaSamples + 2 + (heightSamples-1) * thetaSamples);

	hkVector4 vert;

	//
	// GET BIG SPHERE VERTICES
	//
	vert.setMul(zAxis, height*hkSimdReal_Half + bigRadius);
	vert._setTransformedPos(capsuleToLocal, vert);
	verts.pushBack(vert);

	hkSimdReal sinTheta; sinTheta.setClamped( (bigRadius - smallRadius) / height, hkSimdReal_Minus1, hkSimdReal_1 );
	hkSimdReal cosTheta = (hkSimdReal_1 - sinTheta*sinTheta).sqrt();
	hkSimdReal theta; theta.setFromFloat( hkMath::asin(sinTheta.getReal()) );

	hkSimdReal invPhiSamples; invPhiSamples.setReciprocal(hkSimdReal::fromInt32(phiSamples));
	hkSimdReal invThetaSamples; invThetaSamples.setReciprocal(hkSimdReal::fromInt32(thetaSamples)); invThetaSamples.mul(hkSimdReal_TwoPi);
	for (int i = phiSamples-1 ; i >= 0; i--)
	{
		hkSimdReal frac = hkSimdReal::fromInt32(i) * invPhiSamples;
		hkSimdReal angle = -theta * (hkSimdReal_1 - frac) + hkSimdReal_PiOver2 * frac;

		hkQuaternion qTop; qTop.setAxisAngle(yAxis, -angle);
		hkVector4 topElevation;
		topElevation._setRotatedDir(qTop, xAxis);

		for (int j = 0; j < thetaSamples; j++)
		{
			hkQuaternion rotationTop; rotationTop.setAxisAngle(zAxis, hkSimdReal::fromInt32(j) * invThetaSamples);			
			hkVector4 topDirection;
			topDirection._setRotatedDir(rotationTop, topElevation);

			vert.setAddMul(localBig, topDirection, bigRadius);
			vert._setTransformedPos(capsuleToLocal, vert);
			verts.pushBack(vert);
		}
	}

	//
	// GET CONE VERTICES
	//
	hkSimdReal sMin = -smallRadius * sinTheta;
	hkSimdReal sMax = height - bigRadius * sinTheta;

	hkSimdReal rMin = smallRadius * cosTheta;
	hkSimdReal rMax = bigRadius * cosTheta;

	hkSimdReal invHeightSamples; invHeightSamples.setReciprocal(hkSimdReal::fromInt32(heightSamples));
	for (int j = heightSamples-1; j > 0; j--)
	{
		for (int i = 0; i < thetaSamples; i++)
		{	
			//
			// Calculate direction vector for this angle
			//
			hkQuaternion q; q.setAxisAngle(zAxis, hkSimdReal::fromInt32(i) * invThetaSamples);
			hkVector4 direction;
			direction._setRotatedDir(q, xAxis);

			hkSimdReal axisFrac = hkSimdReal::fromInt32(j) * invHeightSamples;

			hkSimdReal s = (hkSimdReal_1 - axisFrac) * sMin + axisFrac * sMax;

			hkVector4 coneZ;
			coneZ.setAddMul(localSmall, zAxis, s);

			hkSimdReal rad = (hkSimdReal_1 - axisFrac) * rMin + axisFrac * rMax;

			vert.setAddMul(coneZ, direction, rad);
			vert._setTransformedPos(capsuleToLocal, vert);
			verts.pushBack(vert);
		}
	}

	//
	// GET SMALL SPHERE VERTICES
	//
	for (int i = 0; i < phiSamples; i++)
	{
		hkSimdReal frac = hkSimdReal::fromInt32(i) * invPhiSamples;
		hkSimdReal angle = theta * (hkSimdReal_1 - frac) + hkSimdReal_PiOver2 * frac;

		hkQuaternion qBottom; qBottom.setAxisAngle(yAxis, angle);
		hkVector4 bottomElevation;
		bottomElevation._setRotatedDir(qBottom, xAxis);

		for (int j = 0; j < thetaSamples; j++)
		{
			hkQuaternion rotationBottom; rotationBottom.setAxisAngle(zAxis, hkSimdReal::fromInt32(j) * invThetaSamples);			
			hkVector4 bottomDirection;
			bottomDirection._setRotatedDir(rotationBottom, bottomElevation);

			vert.setAddMul(localSmall, bottomDirection, smallRadius);
			vert._setTransformedPos(capsuleToLocal, vert);
			verts.pushBack(vert);
		}
	}

	vert.setMul(zAxis, -(height*hkSimdReal_Half + smallRadius));
	vert._setTransformedPos(capsuleToLocal, vert);

	// Push back bottom vertex
	verts.pushBack(vert);

	{
		for ( int vi = 0; vi < verts.getSize(); ++vi )
		{
			verts[vi]._setTransformedPos(transform, verts[vi]);
		}
	}
	//
	// CONSTRUCT FACE DATA
	//

	// Num. samples AROUND axis is thetaSamples.
	// First off, we have thetaSamples worth of faces connected to the top
	hkGeometry::Triangle tr;
	hkArray<hkGeometry::Triangle> tris;
	tr.m_material=-1;

	int currentBaseIndex = 1;
	for (int i = 0; i < thetaSamples; i++)
	{
		tr.m_a = 0;
		tr.m_b = currentBaseIndex + i;
		tr.m_c = currentBaseIndex + (i+1)%(thetaSamples);
		tris.pushBack(tr);
	}

	// Next we have phi-1 + height + phi-1 lots of thetaSamples*2 worth of faces connected to the previous row
	for(int j = 0; j < 2*(phiSamples-1) + heightSamples; j++)
	{
		for (int i = 0; i < thetaSamples; i++)
		{
			tr.m_a = currentBaseIndex + i;
			tr.m_b = currentBaseIndex + thetaSamples + i;
			tr.m_c = currentBaseIndex + thetaSamples + (i+1)%(thetaSamples);
			tris.pushBack(tr);

			tr.m_b = currentBaseIndex + i;
			tr.m_a = currentBaseIndex + (i+1)%(thetaSamples);
			tr.m_c = currentBaseIndex + thetaSamples + (i+1)%(thetaSamples);
			tris.pushBack(tr);

		}
		currentBaseIndex += thetaSamples;
	}

	// Finally, we have thetaSamples worth of faces connected to the bottom
	for (int i = 0; i < thetaSamples; i++)
	{
		tr.m_b = currentBaseIndex + i;
		tr.m_a = currentBaseIndex + (i+1)%(thetaSamples);
		tr.m_c = currentBaseIndex + thetaSamples;
		tris.pushBack(tr);
	}

	geometryOut.m_vertices.swap(verts);
	geometryOut.m_triangles.swap(tris);
}

void HK_CALL hkGeometryUtils::createCylinderGeometry(const hkVector4& top, const hkVector4& bottom, hkReal radius, int numSides, int numHeightSegments, hkGeometry& geometryOut)
{
	// Generate points in the order: bottom, top & then around the axis
	{
		// center point on the base of the cylinder
		const hkVector4& base = bottom;
		// axis lies along OZ + the length of a single heightSegment
		hkVector4 axis;
		axis.setSub(top, bottom);
		hkSimdReal ihs; ihs.setFromFloat(1.0f / numHeightSegments);
		axis.mul( ihs );

		hkVector4 unitAxis = axis;
		unitAxis.normalize<3>();
		//( 0.0f, 0.0f, 1.0f );
		// vector perpendicular to the axis of the clinder; choose OX
		hkVector4 radiusVec;
		{
			hkVector4 tmp;
			if ( hkMath::fabs(unitAxis(0)) > 0.5f )
			{
				tmp = hkVector4::getConstant(HK_QUADREAL_0100);
			}
			else if ( hkMath::fabs(unitAxis(1)) > 0.5f )
			{
				tmp = hkVector4::getConstant(HK_QUADREAL_0010);
			}
			else //if ( hkMath::fabs(unitAxis(2)) > 0.5f )
			{
				tmp = hkVector4::getConstant(HK_QUADREAL_1000);
			}

			radiusVec.setCross(unitAxis, tmp);
			radiusVec.normalize<3>();
			hkSimdReal sradius; sradius.setFromFloat(radius);
			radiusVec.mul(sradius);
		}
		// quaternion used to rotate radius to generate next set of (axis aligned) points
		hkQuaternion stepQuat; stepQuat.setAxisAngle( unitAxis, HK_REAL_PI * 2.0f / numSides );
		hkQuaternion currQuat; currQuat.setAxisAngle( unitAxis, 0 );

		*(geometryOut.m_vertices.expandBy(1)) = base;
		{
			hkVector4 axisEnd;
			hkSimdReal hSeg; hSeg.setFromFloat(hkReal(numHeightSegments));
			axisEnd.setMul(hSeg, axis);
			axisEnd.add(base);
			*(geometryOut.m_vertices.expandBy(1)) = axisEnd;
		}

		// Generate all points on the side of the cylinder
		for (int s = 0; s < numSides; ++s)
		{
			hkVector4 point;
			{
				// Set start position at the bottom of the cylinder
				hkVector4 modRadius;
				modRadius.setRotatedDir(currQuat, radiusVec);
				point.setAdd(base, modRadius);
			}

			// loop for numHeightSegments and add one extra ending after the loop
			for (int h = 0; h < numHeightSegments; ++h)
			{
				*(geometryOut.m_vertices.expandBy(1)) = point;
				point.add(axis);
			}
			*(geometryOut.m_vertices.expandBy(1)) = point;

			hkQuaternion tmp = currQuat;
			currQuat.setMul(tmp, stepQuat);
		}
		// The above loop does one unneeded quat multiplication at the end.
	}

	//
	// Generate triangle indices: Draw triangles in a right-hand axes space in CCW direction
	//
	{
		// Store number of vertices in every column along the cylinder axis
		const int vertColLen = numHeightSegments + 1;
		const int allVertsCnt = 2 + numSides * (1 + numHeightSegments);
		// Store indices to the first two triangles; 
		// then increase all by the same number to get following triangles
		int indices[4] = { 2+0,
			2+vertColLen,
			2+1,
			2+vertColLen+1
		};

		// Generate trinagles on cylinder side
		for (int s = 0; s < numSides; ++s)
		{
			for (int h = 0; h < numHeightSegments; ++h)
			{
				geometryOut.m_triangles.expandBy(1)->set(indices[0], indices[1], indices[2]);
				geometryOut.m_triangles.expandBy(1)->set(indices[2], indices[1], indices[3]);
				++indices[0];
				++indices[1];
				++indices[2];
				++indices[3];
			}
			// Increase indices again, since we have one less triangles/segments then vertices in a column
			++indices[0];
			++indices[1];
			++indices[2];
			++indices[3];
			if (indices[3] >= allVertsCnt)
			{
				indices[1] += 2 - allVertsCnt;
				indices[3] += 2 - allVertsCnt;
			}
		}

		// Generate both bases
		for (int base = 0; base < 2; ++base)
		{
			int start = 2 + (base ? numHeightSegments : 0);
			for (int s = 0; s < numSides; ++s)
			{
				int next = start + vertColLen;
				if (next >= allVertsCnt)
				{
					next += 2 - allVertsCnt;
				}

				if (base)
				{
					geometryOut.m_triangles.expandBy(1)->set(base, start, next );
				}
				else
				{
					geometryOut.m_triangles.expandBy(1)->set(base, next, start );
				}
				start = next;
			}
		}
	}
}

void HK_CALL hkGeometryUtils::createPlaneGeometry(const hkVector4& normal, const hkVector4& perpToNormal, const hkVector4& center, const hkVector4& extents, hkGeometry& geometryOut)
{
	geometryOut.m_vertices.setSize(5);
	geometryOut.m_triangles.setSize(4);

	hkVector4 otherDir;
	otherDir.setCross(normal, perpToNormal);

	hkVector4 offset1;
	offset1.setMul(extents, otherDir);
	
	hkVector4 offset2;
	offset2.setMul(extents, perpToNormal);

	geometryOut.m_vertices[0].setAdd(center, offset1);
	geometryOut.m_vertices[0].add(offset2);

	geometryOut.m_vertices[1].setAdd(center, offset1);
	geometryOut.m_vertices[1].sub(offset2);

	geometryOut.m_vertices[2].setSub(center, offset1);
	geometryOut.m_vertices[2].add(offset2);

	geometryOut.m_vertices[3].setSub(center, offset1);
	geometryOut.m_vertices[3].sub(offset2);

	geometryOut.m_vertices[4] = center;

	geometryOut.m_triangles[0].set(2,4,3);
	geometryOut.m_triangles[1].set(0,4,2);
	geometryOut.m_triangles[2].set(1,4,0);
	geometryOut.m_triangles[3].set(3,4,1);
}

static inline hkSimdReal _determinant3x3 (hkVector4Parameter col0, hkVector4Parameter col1, hkVector4Parameter col2)
{
	hkVector4 r0; r0.setCross( col1, col2 );

	return col0.dot<3>(r0);
}

/*static*/ void HK_CALL hkGeometryUtils::computeVolume (const hkGeometry& geometry, hkSimdReal& volume)
{
	volume.setZero();

	const hkArray<hkVector4>& verts = geometry.m_vertices;
	const hkArray<hkGeometry::Triangle>& tris = geometry.m_triangles;

	for(int i=0; i < geometry.m_triangles.getSize(); i++) // for each triangle
	{

		volume.add( _determinant3x3(verts[tris[i].m_a],verts[tris[i].m_b],verts[tris[i].m_c]) ); 
	}

	volume.mul(hkSimdReal_Inv6); // since the determinant give 6 times tetra volume
}

//
//	Computes the AABB of the given geometry

void HK_CALL hkGeometryUtils::computeAabb(const hkGeometry& geomIn, hkAabb& aabbOut)
{
	hkAabb aabb;	aabb.setEmpty();
	
	const hkVector4* HK_RESTRICT vptr = geomIn.m_vertices.begin();
	for (int k = geomIn.m_vertices.getSize() - 1; k >= 0; k--)
	{
		aabb.includePoint(vptr[k]);
	}

	aabbOut = aabb;
}

void HK_CALL hkGeometryUtils::computeAabbFromTriangles( const hkGeometry& geomIn, hkAabb& aabbOut )
{
	hkAabb aabb;	aabb.setEmpty();
	for (int t=0; t<geomIn.m_triangles.getSize(); t++)
	{
		hkVector4 triVerts[3];
		geomIn.getTriangle(t, triVerts);
		aabb.includePoint(triVerts[0]);
		aabb.includePoint(triVerts[1]);
		aabb.includePoint(triVerts[2]);
	}

	aabbOut = aabb;
}


/// Transform a geometry
void HK_CALL hkGeometryUtils::transformGeometry (const hkMatrix4& transform, hkGeometry& geometryInOut)
{
	for (int v=0; v<geometryInOut.m_vertices.getSize(); ++v)
	{
		transform.transformPosition (geometryInOut.m_vertices[v], geometryInOut.m_vertices[v]);
	}
}


void HK_CALL hkGeometryUtils::appendGeometry( const hkGeometry& input, hkGeometry& geomInOut )
{
	int vertexOffset = geomInOut.m_vertices.getSize();
	geomInOut.m_vertices.append( input.m_vertices.begin(), input.m_vertices.getSize() );
	hkGeometry::Triangle* newTris = geomInOut.m_triangles.expandBy( input.m_triangles.getSize() );

	for (int i=0, n = input.m_triangles.getSize(); i<n; i++)
	{
		newTris[i] = input.m_triangles[i];
		newTris[i].m_a += vertexOffset;
		newTris[i].m_b += vertexOffset;
		newTris[i].m_c += vertexOffset;
	}
}



void HK_CALL hkGeometryUtils::quantize( hkGeometry& mesh, int resolution )
{
	hkAabb aabb; aabb.setEmpty();

	for (int v=0; v < mesh.m_vertices.getSize(); v++)
	{
		aabb.includePoint( mesh.m_vertices[v] );
	}

	// Add a small (hardcoded) tolerance for the 
	// <ng.todo.aa> Review this quantization for:
	// 1. Error for large values (ask DG)
	// 2. Issues with ranges being different (hence quantization grid being different) in adjacent navmesh sections.
	// 3. Hardcoded #bits is 16
	const hkSimdReal tol = hkSimdReal::fromFloat( hkReal(0.01f) );
	aabb.m_min.setSub( aabb.m_min,tol );
	aabb.m_max.setAdd( aabb.m_max,tol );

	hkVector4 range; range.setSub( aabb.m_max , aabb.m_min );

	hkVector4 domain; domain.setAll( hkSimdReal::fromInt32(resolution - 1) );
	hkVector4 stepSize; stepSize.setDiv( range, domain );
	hkVector4 invStepSize; invStepSize.setDiv( domain, range );

	for (int v=0; v < mesh.m_vertices.getSize(); v++)
	{
		hkVector4 vec;
		vec.setSub( mesh.m_vertices[v], aabb.m_min );
		vec.mul( invStepSize );

		// Convert
		hkIntVector c; 
		c.setConvertF32toU32( vec );
		c.convertU32ToF32( vec );

		// Clamp vec
		vec.setClamped( vec, hkVector4::getZero(), domain );

		vec.mul( stepSize );

		mesh.m_vertices[v].setAdd( vec, aabb.m_min );
	}
}


hkResult HK_CALL hkGeometryUtils::getGeometryInsideAabb( const hkGeometry& geometryIn, hkGeometry& geometryOut, const hkAabb& aabb, GetGeometryInsideAabbMode mode /*= MODE_COPY_DATA*/ )
{
	hkArray<int>::Temp vertexRemap;
	hkResult vertexRemapRes = vertexRemap.reserve( geometryIn.m_vertices.getSize() );
	if (vertexRemapRes != HK_SUCCESS)
	{
		return HK_FAILURE;
	}
		
	vertexRemap.setSize( geometryIn.m_vertices.getSize(), -1);

	int totalTriangles = 0;
	int totalVertices = 0;

	for (int t=0; t<geometryIn.m_triangles.getSize(); t++)
	{
		const hkGeometry::Triangle& tri = geometryIn.m_triangles[t];
		hkVector4 verts[3];
		geometryIn.getTriangle(t, verts);

		hkAabb triAabb;
		triAabb.setFromTriangle(verts[0], verts[1], verts[2]);

		if( !aabb.overlaps(triAabb) )
		{
			continue;
		}

		totalTriangles++;
		const int* triIndices = &tri.m_a;

		// For each triangle vertex, check if we've seen it before
		// If not, update the map and add a vertex to the output vertex array.
		for (int i=0; i<3; i++)
		{
			int index = triIndices[i];
			if (vertexRemap[index] == -1)
			{
				vertexRemap[index] = totalVertices;
				totalVertices++;

				if (mode == MODE_COPY_DATA)
				{
					geometryOut.m_vertices.pushBack( verts[i] );
				}
			}
		}

		if (mode == MODE_COPY_DATA)
		{
			hkGeometry::Triangle& newTri = geometryOut.m_triangles.expandOne();
			newTri.m_a = vertexRemap[tri.m_a];
			newTri.m_b = vertexRemap[tri.m_b];
			newTri.m_c = vertexRemap[tri.m_c];
			newTri.m_material = tri.m_material;
		}
	}

	if (mode == MODE_PRESIZE_ARRAYS)
	{
		hkResult triRes = geometryOut.m_triangles.reserve( totalTriangles );
		if (triRes != HK_SUCCESS)
			return HK_FAILURE;

		hkResult vertRes = geometryOut.m_vertices.reserve( totalVertices );
		if (vertRes != HK_SUCCESS)
			return HK_FAILURE;
	}

	return HK_SUCCESS;
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
