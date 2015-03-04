/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Internal/hkInternal.h>
#include <Common/Internal/GeometryProcessing/Topology/hkgpVertexTriangleTopology.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>

int hkgpVertexTriangleTopologyBase::Triangle::findVertexIndex(VertexIndex index) const
{
	for (int i = 0; i < 3; i++)
	{
		if (m_vertexIndices[i] == index)
		{
			return i;
		}
	}
	return -1;
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                            hkgpVertexTriangleTopology

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

hkgpVertexTriangleTopologyBase::hkgpVertexTriangleTopologyBase(hk_size_t triangleSize):
    m_triangleFreeList(triangleSize, sizeof(void*), 4096)
{
	HK_ASSERT(0x324a3423, triangleSize >= sizeof(Triangle));
}

hkBool32 hkgpVertexTriangleTopologyBase::isClosed() const
{
#if 1
	hkPointerMap<hkUint64, int> map;

	// Add all of the edges. An edge should only appear once
	const Triangle*const* cur = m_triangles.begin();
	const Triangle*const* end = m_triangles.end();
	for (; cur != end; cur++)
	{
		const Triangle* tri = *cur;
		for (int i = 0; i < 3; i++)
		{
			const int next = NextTriIndex(i);
			VertexIndex a = tri->m_vertexIndices[i];
			VertexIndex b = tri->m_vertexIndices[next];

			int inc = 1;
			if ( b < a)
			{
				hkAlgorithm::swap(a, b);
				inc = (1 << 16);
			}

			HK_ASSERT(0x3242a423, a <= b);
			hkUint64 key = (hkUint64(a) << 32) | b;

			hkPointerMap<hkUint64, int>::Iterator iter = map.findKey(key);
			if (map.isValid(iter))
			{
				map.setValue(iter, map.getValue(iter) + inc);
			}
			else
			{
				map.insert(key, inc);
			}
		}
	}

	{
		hkPointerMap<hkUint64, int>::Iterator iter = map.getIterator();
		for (; map.isValid(iter); iter = map.getNext(iter))
		{
			//hkUint64 key = map.getKey(iter);

			const int value = map.getValue(iter);
			const int f = value & 0xffff;
			const int b = value >> 16;

			//HK_ON_DEBUG(hkUint64 key = map.getKey(iter));
			//HK_ON_DEBUG(int start = int(key >> 32); )
			//HK_ON_DEBUG(int end = int(key); )

			// Just to stop the variable not used warning....
			//HK_ASSERT(0x3243423, map.hasKey(key));

			//HK_ASSERT(0x324432, f > 0 && b > 0);

			if (f != b)
			{
				// They must pair to be closed
				return false;
			}
		}
	}

	return true;

#else
	hkPointerMap<hkUint64, int> edgeMap;

	// Add all of the edges. An edge should only appear once
    const Triangle*const* cur = m_triangles.begin();
    const Triangle*const* end = m_triangles.end();
    for (; cur != end; cur++)
    {
		const Triangle* tri = *cur;
		for (int i = 0; i < 3; i++)
		{
			const int next = NextTriIndex(i);
			const VertexIndex a = tri->m_vertexIndices[i];
			const VertexIndex b = tri->m_vertexIndices[next];

			hkUint64 key = (hkUint64(a) << 32) | b;

			// If it has the key -> then the edge has appeared more than once
			if (edgeMap.hasKey(key))
			{
				return false;
			}
			edgeMap.insert(key, 1);
		}
    }

	// Every edge should have a pair
	cur = m_triangles.begin();
	for (; cur != end; cur++)
	{
		const Triangle* tri = *cur;
		for (int i = 0; i < 3; i++)
		{
			const int next = NextTriIndex(i);
			const VertexIndex a = tri->m_vertexIndices[i];
			const VertexIndex b = tri->m_vertexIndices[next];

			hkUint64 key = (hkUint64(b) << 32) | a;

			// If it has the key -> then the edge has appeared more than once
			if (!edgeMap.hasKey(key))
			{
				return false;
			}
		}
	}

    return true;
#endif
}

void hkgpVertexTriangleTopologyBase::deleteTriangle(Triangle* tri)
{
    // Make sure it belongs to it
    HK_ASSERT(0x42423897, m_triangles[tri->m_triangleIndex] == tri);

	// Remove the vertex to edge in map
	for (int i = 0; i < 3; i++)
	{
		EdgeId edgeId = Edge::getEdgeId(tri, i);
		// Remove the entry
		m_vertexEdgeMap.remove(tri->m_vertexIndices[i], edgeId);
	}

    const int numTris = m_triangles.getSize();
    const int triangleIndex = tri->m_triangleIndex;

    // Fix up the last one (if this is the last one, then thats fine too)
    m_triangles[numTris - 1]->m_triangleIndex = triangleIndex;

    m_triangles.removeAt(triangleIndex);
    m_triangleFreeList.free(tri);
}

hkgpVertexTriangleTopologyBase::Triangle* hkgpVertexTriangleTopologyBase::createTriangle(const VertexIndex* indices)
{
    const int numTris = m_triangles.getSize();
    Triangle* tri = (Triangle*)m_triangleFreeList.alloc();

    tri->m_triangleIndex = numTris;

	for (int i = 0; i < 3; i++)
	{
		tri->m_vertexIndices[i] = indices[i];
		const EdgeId edgeId = Edge::getEdgeId(tri, i);
		m_vertexEdgeMap.insert(indices[i], edgeId);
	}

    m_triangles.pushBack(tri);
	return tri;
}

void hkgpVertexTriangleTopologyBase::remapVertexIndices(const hkArray<int>& remap)
{
	m_vertexEdgeMap.clear();

	// Remap all of the triangles
	const int numTris = m_triangles.getSize();
	for (int i = 0; i < numTris; i++)
	{
		Triangle* tri = m_triangles[i];
		for (int j = 0; j < 3; j++)
		{
			int index = remap[tri->m_vertexIndices[j]];
			HK_ASSERT(0x324a3243, index >= 0);
			tri->m_vertexIndices[j] = VertexIndex(index);

			// Add to the map
			const EdgeId edgeId = Edge::getEdgeId(tri, j);
			m_vertexEdgeMap.insert(index, edgeId);
		}
	}
}

hkBool32 hkgpVertexTriangleTopologyBase::isOk() const
{
	VertexEdgeMapType map;

	for (int i = 0; i < m_triangles.getSize(); i++)
	{
		Triangle* tri = m_triangles[i];

		for (int j = 0; j < 3; j++)
		{
			EdgeId edgeId = Edge::getEdgeId(tri, j);
			map.insert(tri->m_vertexIndices[j], edgeId);
		}
	}

	// Okay lets go through both and make sure they are the same
	{
		if (map.getSize() != m_vertexEdgeMap.getSize())
		{
			return false;
		}

		VertexEdgeMapType::Iterator iter = map.getIterator();

		for (; map.isValid(iter); iter = map.getNext(iter))
		{
			int vertexIndex = map.getKey(iter);
			EdgeId edgeId = map.getValue(iter);

			int count = map.findNumEntries(vertexIndex, edgeId);
			int count2 = m_vertexEdgeMap.findNumEntries(vertexIndex, edgeId);

			if (count != count2)
			{
				return false;
			}
		}
	}

	return true;
}


hkgpVertexTriangleTopologyBase::Triangle* hkgpVertexTriangleTopologyBase::createTriangle(const int* indices)
{
	const int numTris = m_triangles.getSize();
	Triangle* tri = (Triangle*)m_triangleFreeList.alloc();

	tri->m_triangleIndex = numTris;

	for (int i = 0; i < 3; i++)
	{
		HK_ASSERT(0x324a234a, indices[i] >= 0);
		VertexIndex index = VertexIndex(indices[i]);

		tri->m_vertexIndices[i] = index;
		const EdgeId edgeId = Edge::getEdgeId(tri, i);
		m_vertexEdgeMap.insert(index, edgeId);
	}

	m_triangles.pushBack(tri);
	return tri;
}

void hkgpVertexTriangleTopologyBase::findAllEdges(int start, int end, hkArray<EdgeId>& edgeIds) const
{
	edgeIds.clear();

	// Find all of the edges from start
	{
		VertexEdgeMapType::Iterator iter = m_vertexEdgeMap.findKey(start);
		for (; m_vertexEdgeMap.isValid(iter); iter = m_vertexEdgeMap.getNext(iter, start))
		{
			const EdgeId edgeId = m_vertexEdgeMap.getValue(iter);
			Edge edge(edgeId);
			if (edge.getEnd() == VertexIndex(end))
			{
				edgeIds.pushBack(edgeId);
			}
		}
	}

	// Find all of the edges from the end
	if (start != end)
	{
		VertexEdgeMapType::Iterator iter = m_vertexEdgeMap.findKey(end);
		for (; m_vertexEdgeMap.isValid(iter); iter = m_vertexEdgeMap.getNext(iter, end))
		{
			const EdgeId edgeId = m_vertexEdgeMap.getValue(iter);
			Edge edge(edgeId);
			if (edge.getEnd() == VertexIndex(start))
			{
				edgeIds.pushBack(edgeId);
			}
		}
	}
}

int hkgpVertexTriangleTopologyBase::checkTopology(int ignoreFlags) const
{
	int flags = 0;
    const int numTris = m_triangles.getSize();

    for (int i = 0; i < numTris; i++)
	{
        Triangle* tri = m_triangles[i];

        for (int j = 0; j < 3; j++)
		{
            Edge edge(tri, j);

            const VertexIndex start = edge.getStart();
            const VertexIndex end = edge.getEnd();

            //
            if (start == end)
            {
				flags |= CHECK_FLAG_NULL_EDGE;
            }
		}
	}

    return flags & (~ignoreFlags);
}

void hkgpVertexTriangleTopologyBase::findVertexLeavingEdges(int vertexIndex, hkArray<EdgeId>& edges) const
{
	edges.clear();
	VertexEdgeMapType::Iterator iter = m_vertexEdgeMap.findKey(vertexIndex);

	for (; m_vertexEdgeMap.isValid(iter); iter = m_vertexEdgeMap.getNext(iter, vertexIndex))
	{
		edges.pushBack(m_vertexEdgeMap.getValue(iter));
	}
}

void hkgpVertexTriangleTopologyBase::findVertexReturningEdges(int vertexIndex, const hkArray<EdgeId>& leavingEdges, hkArray<EdgeId>& returningEdges) const
{
	returningEdges.clear();

	const int numLeaving = leavingEdges.getSize();
	returningEdges.setSize(numLeaving);

	for (int i = 0; i < numLeaving; i++)
	{
		Edge edge(leavingEdges[i]);
		HK_ASSERT(0x3424a234, edge.getStart() == VertexIndex(vertexIndex));
		edge.prev();
		returningEdges[i] = edge.getEdgeId();
	}
}

void hkgpVertexTriangleTopologyBase::findAllVertexEdges(int vertexIndex, hkArray<EdgeId>& edges) const
{
	edges.clear();

	VertexEdgeMapType::Iterator iter = m_vertexEdgeMap.findKey(vertexIndex);
	for (; m_vertexEdgeMap.isValid(iter); iter = m_vertexEdgeMap.getNext(iter, vertexIndex))
	{
		const EdgeId edgeId = m_vertexEdgeMap.getValue(iter); 
		edges.pushBack(edgeId);
		Edge edge(edgeId);
		HK_ASSERT(0x3443aa24, edge.getStart() == VertexIndex(vertexIndex));
		edge.prev();
		HK_ASSERT(0x3443aa24, edge.getEnd() == VertexIndex(vertexIndex));
		edges.pushBack(edge.getEdgeId());
	}
}

int hkgpVertexTriangleTopologyBase::calcNumVertexLeavingEdges(int vertexIndex) const
{
	return m_vertexEdgeMap.findNumEntries(vertexIndex);
}

void hkgpVertexTriangleTopologyBase::findVertexTriangles(int vertexIndex, hkArray<Triangle*>& tris) const
{
	tris.clear();
	VertexEdgeMapType::Iterator iter = m_vertexEdgeMap.findKey(vertexIndex);

	for (; m_vertexEdgeMap.isValid(iter); iter = m_vertexEdgeMap.getNext(iter, vertexIndex))
	{
		tris.pushBack(Edge::getTriangle(m_vertexEdgeMap.getValue(iter)));
	}
	uniqueTriangles(tris);
}

/* static */void hkgpVertexTriangleTopologyBase::uniqueTriangles(hkArray<Triangle*>& tris)
{
	// Sort 
	hkSort(tris.begin(), tris.getSize());

	// Remove any repeats
	Triangle** cur = tris.begin();
	Triangle** end = tris.end();
	Triangle** back = cur;

	// Remove all runs
	Triangle* last = HK_NULL;
	for (; cur < end; cur++)
	{
		Triangle* tri = *cur;
		// Look for a transition
		if (*cur != last)
		{
			// Copy back
			*back++ = tri;
			// This is now the last
			last = tri;
		}
	}
	// Set the size
	tris.setSizeUnchecked(int(back - tris.begin()));
}

/* static */void hkgpVertexTriangleTopologyBase::uniqueTriangles(hkArray<EdgeId>& edges, hkArray<Triangle*>& tris)
{
	tris.clear();
	for (int i = 0; i < edges.getSize(); i++)
	{
		tris.pushBack(Edge::getTriangle(edges[i]));
	}
	uniqueTriangles(tris);
}

void hkgpVertexTriangleTopologyBase::disconnectVertex(int vertexIndex)
{
	hkInplaceArray<Triangle*, 16> tris;
	findVertexTriangles(vertexIndex, tris);

    for (int i = 0; i < tris.getSize(); i++)
    {
		deleteTriangle(tris[i]);
	}
}

void hkgpVertexTriangleTopologyBase::setTriangleVertexIndex(Triangle* tri, int index, int to)
{
	HK_ASSERT(0x32432432, index >= 0 && index < 3);
	int from = tri->m_vertexIndices[index];

	const EdgeId edgeId = Edge::getEdgeId(tri, index);
	if (from != to)
	{
		// Remove from the 
		m_vertexEdgeMap.remove(from, edgeId);
		m_vertexEdgeMap.insert(to, edgeId);

		tri->m_vertexIndices[index] = to;
	}
}

int hkgpVertexTriangleTopologyBase::reindexTriangle(Triangle* tri, int from, int to)
{
	HK_ASSERT(0x324aa234, from != to);

	int numReindexed = 0;
	for (int i = 0; i < 3; i++)
	{
		int curIndex = tri->m_vertexIndices[i];
		if (curIndex == from)
		{
			const EdgeId edgeId = Edge::getEdgeId(tri, i);

			m_vertexEdgeMap.remove(from, edgeId);
			m_vertexEdgeMap.insert(to, edgeId);

			tri->m_vertexIndices[i] = to;

			numReindexed++;
		}
	}

	return numReindexed;
}

void hkgpVertexTriangleTopologyBase::reindexVertexIndex(int from, int to)
{
	HK_ASSERT(0x32432432, from != to);
	hkInplaceArray<EdgeId, 32> edges;
	findVertexLeavingEdges(from, edges);

	// Just remove all edges from the from list
	m_vertexEdgeMap.removeAll(VertexIndex(from));

	// Add all of them to the 'to' list
	for (int i = 0; i < edges.getSize(); i++)
	{
		EdgeId edgeId = edges[i];

		// Update the vertex index
		Edge edge(edgeId);
		edge.setStart(to);

		// Add it
		m_vertexEdgeMap.insert(to, edgeId);
	}
}

hkgpVertexTriangleTopologyBase::Edge hkgpVertexTriangleTopologyBase::getOppositeEdge(const Edge& edge) const
{
	int start = edge.getStart();
	int end = edge.getEnd();

	VertexEdgeMapType::Iterator iter = m_vertexEdgeMap.findKey(end);
	for (; m_vertexEdgeMap.isValid(iter); iter = m_vertexEdgeMap.getNext(iter, end))
	{
		EdgeId edgeId = m_vertexEdgeMap.getValue(iter);
		Edge checkEdge(edgeId);

		if (checkEdge.getEnd() == VertexIndex(start))
		{
			// Okay this is an opposite edge
			return checkEdge;
		}
	}

	// Return an invalid edge
	return Edge();
}


#ifdef HK_DEBUG

/* static */void hkgpVertexTriangleTopologyBase::selfTest()
{
    const hkUint8 indices[] = { 0, 1, 3,
					   1, 2, 3,
					   0, 2, 1,
					   0, 3, 2 };

    hkgpVertexTriangleTopologyBase topology;
	{
        int numTris = HK_COUNT_OF(indices) / 3;

        for (int i = 0; i < numTris; i++)
		{
			const hkUint8* ind = indices + i * 3;

			VertexIndex vertIndices[3] = { ind[0], ind[1], ind[2] };
			topology.createTriangle(vertIndices);
		}
	}
	HK_ASSERT(0x43242389, topology.isClosed());

	{
		Triangle* tri = topology.getTriangle(0);
		// Delete
		topology.deleteTriangle(tri);
		HK_ASSERT(0x2332432, !topology.isClosed());
	}
}

#endif

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
