/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Common/GeometryUtilities/hkGeometryUtilities.h>
#include <Common/GeometryUtilities/Mesh/Simplifiers/QemSimplifier/hkQemSimplifier.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/Container/String/hkStringBuf.h>
#include <Common/GeometryUtilities/Mesh/Simplifiers/QemSimplifier/hkQuadricMetric.h>
#include <Common/Visualize/hkDebugDisplay.h>
#include <Geometry/Collide/Algorithms/Triangle/hkcdTriangleUtil.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

							hkQemSimplifier

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

void hkQemSimplifier::EdgeMap::addTriangle(const Triangle& tri)
{
	for (int i = 0; i < 3; i++)
	{
		Edge edge(const_cast<Triangle*>(&tri), i);
		int start = edge.getStart();
		int end = edge.getEnd();

		m_edgeMap.insert((hkUint64(start) << 32) | end, edge.getEdgeId());
	}
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

					hkQemSimplifier::Group

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

int hkQemSimplifier::Group::createAttribute()
{
	hkVector4* attrib = (hkVector4*)m_attributeFreeList.alloc();
	hkString::memClear16(attrib, (m_attributeVecSize + m_quadricVecSize) * (sizeof(hkVector4)>>4));

	if (m_availableAttributeIndices.getSize() > 0)
	{
		const int attribIndex = m_availableAttributeIndices.back();
		m_availableAttributeIndices.popBack();
		HK_ASSERT(0x432a3242, m_attributes[attribIndex] == HK_NULL);

		m_attributes[attribIndex] = attrib;
		return attribIndex;
	}
	else
	{
		m_attributes.pushBack(attrib);
		return m_attributes.getSize() - 1;
	}
}

void hkQemSimplifier::Group::deleteAttribute(int index)
{
	hkVector4* attrib = m_attributes[index];
	HK_ASSERT(0x2423a432, attrib);
	// Free the memory
	m_attributeFreeList.free(attrib);
	// Mark as available
	m_attributes[index] = HK_NULL;
	// This index is now available
	m_availableAttributeIndices.pushBack(index);
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

							hkQemSimplifier

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

hkQemSimplifier::hkQemSimplifier()
:	m_contractionFreeList(sizeof(EdgeContraction), sizeof(void*), 4096)
,	m_thresholds(hkReal(1.0e-6f))
{
	clear();
	m_openEdgeScale					= hkSimdReal_1;
	m_enableOpenEdgeGeometry		= true;
	m_discardInvalidAttributeMoves	= false;
	m_preserveMaterialBoundary		= false;
	m_materialBoundaryTolerance		= hkSimdReal_Max;

	m_simplifyCoplanar				= false;
	m_coplanarTolerance				. setZero();

	m_scaleCalc.setAndDontIncrementRefCount(new SizeScaleCalculator);
}

void hkQemSimplifier::clear()
{
	m_contractionFreeList.freeAll();
	m_contractions.clear();
	m_groups.clear();

	m_edgeContractionMap.clear();

	m_positions.clear();
	m_topology.clear();

	m_isFinalized = false;
	m_inGroup = false;
	m_inMesh = false;
}

int hkQemSimplifier::_addGroup(int attribSize, const AttributeFormat& fmt, hkSimdRealParameter positionToAttribScale)
{
	Group& group = m_groups.expandOne();

	group.m_positionToAttribScale = positionToAttribScale;

	group.m_attributeSize = attribSize; 
	group.m_attributeVecSize = (group.m_attributeSize + 3) >> 2;

	// Work out quadric size
	group.m_quadricSize = hkQuadricMetric::calcStoreSize(group.m_attributeSize);
	group.m_quadricVecSize = (group.m_quadricSize + 3) >> 2;

	// Set up the attribute allocator
	group.m_attributeFreeList.init((group.m_quadricVecSize + group.m_attributeVecSize) * sizeof(hkVector4), HK_REAL_ALIGNMENT, 8192);

	// Compute position offset
	int positionOffset = 0;
	for (int k = 0; k < fmt.m_entries.getSize(); k++)
	{
		const AttributeEntry& entry = fmt.m_entries[k];
		if ( entry.m_type == ATTRIB_TYPE_POSITION )
		{
			break;
		}
		positionOffset += entry.m_size;
	}
	group.m_positionOffset	= positionOffset;
	group.m_fmt				= fmt;
	
	// Return the index
	return m_groups.getSize() - 1;
}

void hkQemSimplifier::startGroup(int attribSize, const AttributeFormat& fmt, hkSimdRealParameter positionToAttribScale)
{
	HK_ASSERT(0x3453443a, m_inMesh);
	HK_ASSERT(0x26546546, !m_inGroup);
	_addGroup(attribSize, fmt, positionToAttribScale);
	m_inGroup = true;
}

bool hkQemSimplifier::_areGroupIndicesValid(int groupIndex) const
{
	const Group& group = m_groups[groupIndex];

	const int numAttribs = group.m_attributes.getSize();
	const int numPositions = m_positions.getSize();

	
	for (int i = 0; i < m_topology.getNumTriangles(); i++)
	{
		const Triangle* tri = m_topology.getTriangle(i);
		if (tri->m_groupIndex == groupIndex)
		{
			for (int j = 0; j < 3; j++)
			{
				const int positionIndex = tri->m_vertexIndices[j];
				const int attribIndex = tri->m_attributeIndices[j];

				if (positionIndex < 0 || positionIndex >= numPositions)
				{
					return false;
				}

				if (attribIndex < 0 || attribIndex >= numAttribs)
				{
					return false;
				}
			}
		}
	}

	return true;
}

void hkQemSimplifier::endGroup()
{
	HK_ASSERT(0x26546546, m_inGroup);

	if( !_areGroupIndicesValid(m_groups.getSize() - 1))
	{
		HK_WARN(0x2888002, "Mismatched group indices between start/end.");
	}
	m_inGroup = false;
}

int hkQemSimplifier::addAttribute(hkVector4* attribute)
{
	Group& group = m_groups.back();
	hkVector4* dst = addAttribute();
	hkString::memCpy16(dst, attribute, group.m_attributeSize * (sizeof(hkVector4)>>4));
	return group.m_attributes.getSize();
}

hkVector4* hkQemSimplifier::addAttribute()
{
	HK_ASSERT(0x2432432, m_inGroup);

	Group& group = m_groups.back();

	int index = group.createAttribute();
	return group.m_attributes[index];
}

hkVector4* hkQemSimplifier::addAttribute(int& indexOut)
{
	HK_ASSERT(0x2432432, m_inGroup);
	hkVector4* attrib = addAttribute();
	Group& group = m_groups.back();
	indexOut = group.m_attributes.getSize() - 1;

	return attrib;
}

void hkQemSimplifier::addTriangle(const int positionIndices[], const int attributeIndices[], hkReal weight)
{
	HK_ASSERT(0x2432432, m_inGroup);

	const int groupIndex = m_groups.getSize() - 1;

	Triangle* tri = m_topology.createTriangle(positionIndices);

	for (int i = 0; i < 3; i++)
	{
		const int positionIndex = positionIndices[i];
		const int attribIndex = attributeIndices[i];

		// Check in range
		HK_ASSERT(0x2423432, positionIndex >= 0 && positionIndex < m_positions.getSize());
		
		tri->m_vertexIndices[i] = positionIndex;
		tri->m_attributeIndices[i] = attribIndex;	
	}

	tri->m_groupIndex = groupIndex;
	tri->m_weight = weight;
}

void hkQemSimplifier::getPosition01(int index, hkVector4& pos) const
{
	pos.setSub(m_positions[index], m_positionsAabb.m_min);
	pos.mul(m_aabbScale);
}

void hkQemSimplifier::getPositionAttrib(int groupIndex, int positionIndex, hkVector4& pos) const
{
	const Group& group = m_groups[groupIndex];
	pos.setSub(m_positions[positionIndex], m_positionsAabb.m_min);
	pos.mul(m_aabbScale);
	pos.mul(group.m_positionToAttribScale);
}

void hkQemSimplifier::getAttribPosition(int groupIndex, int attribIndex, hkVector4& pos) const
{
	const Group& group = m_groups[groupIndex];
	pos.load<3,HK_IO_NATIVE_ALIGNED>(group.getAttributePosition(attribIndex));
	hkSimdReal invPosScale; invPosScale.setReciprocal<HK_ACC_FULL,HK_DIV_IGNORE>(group.m_positionToAttribScale);
	pos.mul(invPosScale);

	hkVector4 scale; scale.setSub(m_positionsAabb.m_max, m_positionsAabb.m_min);
	pos.mul(scale);
	pos.add(m_positionsAabb.m_min);
}

void hkQemSimplifier::startMesh(const hkAabb& aabb, const hkVector4* positions, int numPositions, bool unitScalePosition)
{
	HK_ASSERT(0x24234a32, !m_inMesh);
	clear();

	m_positions.clear();
	m_positions.insertAt(0, positions, numPositions);
		
	m_inMesh = true;
	m_inGroup = false;

	m_positionsAabb = aabb;

	if ( unitScalePosition )
	{
		m_aabbScale.setSub(aabb.m_max, aabb.m_min);
		m_aabbScale.setReciprocal(m_aabbScale); 
		m_aabbScale.setComponent<3>(hkSimdReal_1);
	}
	else
	{
		m_aabbScale = hkVector4::getConstant<HK_QUADREAL_1>();
	}
}

hkResult hkQemSimplifier::endMesh()
{
	HK_ASSERT(0x324a3423, !m_inGroup);
	HK_ASSERT(0x324a3423, m_inMesh);

	m_inMesh = false;

	if (m_enableOpenEdgeGeometry)
	{
	}
	else
	{
		// If 
		if (!m_topology.isClosed())
		{
			return HK_FAILURE;
		} 
	}

	// Calculate all of the quadrics at vertices
	if (!_calcQuadrics())
	{
		return HK_FAILURE;
	}

	// Calculate the material boundaries
	findMaterialBoundaries();

	// Calculate all of the contractions
	if (!_calcContractions())
	{
		return HK_FAILURE;
	}

	if(!isOk())
	{
		return HK_FAILURE;
	}

	return HK_SUCCESS;
}

void hkQemSimplifier::_accumulateTriangleQem(const Triangle& tri, const hkVector4* triQem)
{
	Group& group = m_groups[tri.m_groupIndex];

	const int quadricVecSize = group.m_quadricVecSize;
	for (int i = 0; i < 3; i++)
	{
		int attribIndex = tri.m_attributeIndices[i];

		hkVector4* dstQuadric = group.getQuadric(attribIndex);

		// Accumulate the triangles quadric into the attribute which indexes that triangle
		for (int j = 0; j < quadricVecSize; j++)
		{
			dstQuadric[j].add(triQem[j]);
		}
	}
}

void hkQemSimplifier::_accumulateScaleTriangleQem(const Triangle& tri, const hkVector4* triQem)
{
	Group& group = m_groups[tri.m_groupIndex];

	const int quadricVecSize = group.m_quadricVecSize;
	for (int i = 0; i < 3; i++)
	{
		hkSimdReal scale = m_scaleCalc->calcScale(*this, tri, i);

		int attribIndex = tri.m_attributeIndices[i];
		hkVector4* dstQuadric = group.getQuadric(attribIndex);

		// Accumulate the triangles quadric into the attribute which indexes that triangle
		for (int j = 0; j < quadricVecSize; j++)
		{
			dstQuadric[j].addMul( scale, triQem[j]);
		}
	}
}


bool hkQemSimplifier::_oppositeTriangleExists(Triangle* tri) const
{
	// I want to look for another triangle that has exactly the same positions, but faces in the opposite direction
	
	hkInplaceArray<EdgeId, 16> edgeIds;
	const VertexIndex apexIndex = tri->m_vertexIndices[2]; 
	m_topology.findAllEdges(tri->m_vertexIndices[1], tri->m_vertexIndices[0], edgeIds);

	for (int i = 0; i < edgeIds.getSize(); i++)
	{
		const Edge edge(edgeIds[i]);

		if (edge.getApex() == apexIndex)
		{
			return true;
		}
	}

	return false;
}

HK_DISABLE_OPTIMIZATION_VS2008_X64
bool hkQemSimplifier::_calcQuadrics()
{
	const int numTris = m_topology.getNumTriangles();

	hkVectorN a;
	hkVectorN b;
	hkVectorN c;

	hkArray<hkVector4> triQuadrics;
	// Maps a triangle index to a tri quadric
	hkArray<int> triangleMap;
	hkArray<Triangle*> groupTris;

	EdgeMap edgeMap;

	if (m_enableOpenEdgeGeometry)
	{
		for (int i = 0; i < numTris; i++)
		{
			Triangle* tri = m_topology.getTriangle(i);
			edgeMap.addTriangle(*tri);
		}
	}

	hkQuadricMetric quadricMetric;

	const int numGroups = m_groups.getSize();
	for (int i = 0; i < numGroups; i++)
	{
		Group& group = m_groups[i];

		// Set all to -1
		triangleMap.clear();
		triangleMap.setSize(numTris, -1);

		const int attribSize = group.m_attributeSize;

		// What is the size going to be
		const int quadricVecSize = group.m_quadricVecSize;

		triQuadrics.clear();
		int numTriQuadrics = 0;

		groupTris.clear();

		for (int j = 0; j < numTris; j++)
		{
			Triangle* tri = m_topology.getTriangle(j);

			if (tri->m_groupIndex == i)
			{
				// Set up the vertex indices
				a.alias(group.getAttribute(tri->m_attributeIndices[0]), attribSize);
				b.alias(group.getAttribute(tri->m_attributeIndices[1]), attribSize);
				c.alias(group.getAttribute(tri->m_attributeIndices[2]), attribSize);

				if (! (a.isOk() && b.isOk() && c.isOk()) ) 
				{
					return false;
				}

				// Work out the quadric
				quadricMetric.setFromPlane(a, b, c);

#if 0 && defined(HK_DEBUG)
				{
					const hkReal epsilon = 1e-2f;
					hkReal dist = quadricMetric.calcDistanceSquared(a);
					HK_ASSERT(0x4324a32a, hkMath::fabs(dist) < epsilon);
					dist = quadricMetric.calcDistanceSquared(b);
					HK_ASSERT(0x4324a32a, hkMath::fabs(dist) < epsilon);
					dist = quadricMetric.calcDistanceSquared(c);
					HK_ASSERT(0x4324a32a, hkMath::fabs(dist) < epsilon);

				}
#endif

				// Set up the map
				triangleMap[j] = numTriQuadrics;

				hkVector4* dstQuadric = triQuadrics.expandBy(quadricVecSize);
				dstQuadric[quadricVecSize - 1].setZero();

				// Store the quadric
				quadricMetric.store((hkReal*)dstQuadric);

				hkSimdReal scale = hkSimdReal_1;
				if (_oppositeTriangleExists(tri))
				{
					// Its an internal face... make its strength less. Half should do it.
					scale = hkSimdReal_Inv2;
				}

				// Scale by weighting
				scale = scale * hkSimdReal::fromFloat(tri->m_weight);

				// Scale the metric - taking into account the triangle scale, and if there is an opposite
				for (int k = 0; k < quadricVecSize; k++)
				{
					dstQuadric[k].mul(scale);
				}

				// Another tri in this group
				groupTris.pushBack(tri);

				// We added a quadric
				numTriQuadrics++;
			}
		}	

		// At this point all the quadrics have been zeroed when they were initially setup
		// Accumulate the quadrics from each plane (triangle) that touches the attribute
		for (int j = 0; j < numTris; j++)
		{
			const int triIndex = triangleMap[j];
			if (triIndex < 0)
			{
				continue;
			}

			Triangle* tri = m_topology.getTriangle(j);
			const hkVector4* triQem = &triQuadrics[triIndex * quadricVecSize];

			if (m_scaleCalc)
			{
				_accumulateScaleTriangleQem(*tri, triQem);
			}
			else
			{
				_accumulateTriangleQem(*tri, triQem);
			}
		}	


		if (m_enableOpenEdgeGeometry && m_openEdgeScale.isGreaterZero())
		{
			_accumulateOpenEdgeQem(groupTris, edgeMap);
		}
	}

	// the distance from the quadric to the attrib should be close to 0

#if 0
	def HK_DEBUG
	{
		hkQuadricMetric qm;
		hkVectorN v;
		for (int i = 0; i < m_groups.getSize(); i++)
		{
			Group& group = m_groups[i];

			for (int j = 0; j < group.m_attributes.getSize(); j++)
			{
				v.alias(group.getAttribute(j), group.m_attributeSize);
				qm.load(group.m_attributeSize, (hkReal*)group.getQuadric(j));

				hkReal dist = qm.calcDistanceSquared(v).getReal();
				dist = hkMath::fabs(dist);

				HK_ASSERT(0x2432a432, dist < 1e-2f);
			}
		}
	}
#endif


	return true;
}
HK_RESTORE_OPTIMIZATION_VS2008_X64
	
void hkQemSimplifier::_accumulateOpenEdgeQem(const hkArray<Triangle*>& groupTris, const EdgeMap& edgeMap)
{
	if (groupTris.getSize() <= 0)
	{
		return;
	}

	hkVectorN a;
	hkVectorN b;
	hkVectorN c;

	const int groupIndex = groupTris[0]->m_groupIndex;
	Group& group = m_groups[groupIndex];

	const int quadricVecSize = group.m_quadricVecSize;
	const int attribSize = group.m_attributeSize;

	// Set up the buffer to store to
	hkInplaceArray<hkVector4, 8> quadricBuffer;
	quadricBuffer.setSize(quadricVecSize);
	quadricBuffer.back().setZero();

	const hkSimdReal scaleFactor = m_openEdgeScale;

	hkVector4* workQuadric = quadricBuffer.begin();
	hkQuadricMetric quadricMetric;

	for (int i = 0; i < groupTris.getSize(); i++)
	{
		Triangle* tri = groupTris[i];

		HK_ASSERT(0x423432a4, tri->m_groupIndex == groupIndex);

		// Work out if it has opposite edges
		hkBool32 hasOpposite[3];
		for (int j = 0; j < 3; j++)
		{
			Edge edge(tri, j);
			// Record if it has a back edge
			hasOpposite[j] = edgeMap.hasEdge(edge.getEnd(), edge.getStart());
		}

		// If they all have opposites then it is contained
		if (hasOpposite[0] && hasOpposite[1] && hasOpposite[2])
		{
			continue;
		}

		hkVector4 attribPos[3];			/// Positions in attrib space
		hkVector4 pos01[3];			/// Positions in 0-1 space

		for (int j = 0; j < 3; j++)
		{
			// Load the position in attribute space
			attribPos[j].load<3,HK_IO_NATIVE_ALIGNED>(group.getAttributePosition(tri->m_attributeIndices[j]));
			// Load the position in 01 space
			getPosition01(tri->m_vertexIndices[j], pos01[j]);
		}

		// Calc normal
		hkVector4 normal;
		_calcNormal(attribPos[0], attribPos[1], attribPos[2], normal);

		for (int j = 0; j < 3; j++)
		{
			// Get the edge and see if its open or not
			Edge edge(tri, j);

			// Count the amount which have opposite edges
			if (!hasOpposite[j])
			{
				// This edge is open
				const int nextIndex = edge.getNextIndex();

				// Set up the vertex indices
				a.alias(group.getAttribute(tri->m_attributeIndices[j]), attribSize);
				b.alias(group.getAttribute(tri->m_attributeIndices[nextIndex]), attribSize);

				// Copy a to c
				c = a;
				// Store the new position for c
				hkVector4 pos; pos.setAdd(attribPos[j], normal);
				pos.store<3,HK_IO_NATIVE_ALIGNED>(c.getElements() + group.m_positionOffset);
				
				// Work out the quadric from the plane
				quadricMetric.setFromPlane(a, b, c);

				// Store
				quadricMetric.store((hkReal*)workQuadric);

				// Add it to the attrib for this and the next
				hkVector4* dstQuadric0 = group.getQuadric(tri->m_attributeIndices[j]);
				hkVector4* dstQuadric1 = group.getQuadric(tri->m_attributeIndices[nextIndex]);

				// Work out the edge length. Leave squared - to approximate a reasonably shaped triangle
				hkVector4 t; t.setSub(pos01[nextIndex], pos01[j]);
				hkSimdReal scale = t.lengthSquared<3>() * scaleFactor;

				// Accumulate the triangles quadric into the attribute which indexes that triangle
				for (int k = 0; k < quadricVecSize; k++)
				{
					hkVector4 s; s.setMul(scale, workQuadric[k]);

					dstQuadric0[k].add(s);
					dstQuadric1[k].add(s);
				}
			}
		}
	}
}


void hkQemSimplifier::_calcSumQems(int groupIndex, int start, int end, hkQuadricMetric& qm)
{
	Group& group = m_groups[groupIndex];

	const int quadricVecSize = group.m_quadricVecSize;

	hkLocalArray<hkVector4> sum(quadricVecSize);
	sum.setSize(quadricVecSize);

	const hkVector4* startQem = group.getQuadric(start);
	const hkVector4* endQem = group.getQuadric(end);

	// Accumulate the quadric error
	for (int i = 0; i < quadricVecSize; i++)
	{
		sum[i].setAdd(startQem[i], endQem[i]);
	}

	qm.load(group.m_attributeSize, (const hkReal*)sum.begin());
}

hkBool32 hkQemSimplifier::_allEdgesHaveSameAttribs(const hkArray<EdgeId>& edgeIds)
{
	if (edgeIds.getSize() < 0)
	{
		return false;
	}
	Edge edge(edgeIds[0]);

	const Triangle* tri = edge.getTriangle();
	const int groupIndex = tri->m_groupIndex;

	const int startIndex = edge.getStart();

	const int startAttrib = tri->m_attributeIndices[edge.getIndex()];
	const int endAttrib = tri->m_attributeIndices[edge.getNextIndex()];

	for (int i = 1; i < edgeIds.getSize(); i++)
	{
		edge.setEdgeId(edgeIds[i]);
		tri = edge.getTriangle();

		if (tri->m_groupIndex != groupIndex)
		{
			return false;
		}

		int curStartAttrib = tri->m_attributeIndices[edge.getIndex()];
		int curEndAttrib = tri->m_attributeIndices[edge.getNextIndex()];

		if (edge.getStart() != VertexIndex(startIndex))
		{
			hkAlgorithm::swap(curStartAttrib, curEndAttrib);
			HK_ASSERT(0x442a3423, edge.getEnd() == VertexIndex(startIndex));
		}
		
		// The attribs don't match
		if (startAttrib != curStartAttrib || endAttrib != curEndAttrib)
		{
			return false;
		}
	}
	return true;
}

hkBool32 hkQemSimplifier::_findMove(const Attribute& attrib, const hkArray<EdgeId>& contractionEdgeIds, Attribute& move)
{
	// Find in the contraction edges, an edge going from the start
	for (int i = 0; i < contractionEdgeIds.getSize(); i++)
	{
		EdgeId edgeId = contractionEdgeIds[i];
		Edge edge(edgeId);

		Triangle* tri = edge.getTriangle();

		// If not in the same group then cannot be a match
		if (tri->m_groupIndex != attrib.m_groupIndex)
		{
			continue;
		}

		const int startAttribIndex = tri->m_attributeIndices[edge.getIndex()];
		const int endAttribIndex = tri->m_attributeIndices[edge.getNextIndex()];

		if (startAttribIndex == attrib.m_attributeIndex)
		{
			move.m_attributeIndex = endAttribIndex;
			move.m_groupIndex = tri->m_groupIndex;
			return true;
		}

		if (endAttribIndex == attrib.m_attributeIndex)
		{
			move.m_attributeIndex = startAttribIndex;
			move.m_groupIndex = tri->m_groupIndex;
			return true;
		}
	}

	return false;
}

void hkQemSimplifier::_calcVertexUniqueAttributes(int vertexIndex, hkArray<Attribute>& attribs)
{
	attribs.clear();
	{
		hkInplaceArray<EdgeId, 16> edgeIds;

		m_topology.findVertexLeavingEdges(vertexIndex, edgeIds);

		for (int i = 0; i < edgeIds.getSize(); i++)
		{
			const EdgeId edgeId = edgeIds[i];
			Edge edge(edgeId);

			Triangle* tri = edge.getTriangle();

			// We are finding attributes at the vertex index
			HK_ASSERT(0x324243aa, tri->m_vertexIndices[edge.getIndex()] == VertexIndex(vertexIndex));

			Attribute attrib;
			attrib.m_attributeIndex = tri->m_attributeIndices[edge.getIndex()];
			attrib.m_groupIndex = tri->m_groupIndex;

			// If doesn't have it, then add it
			int index = attribs.indexOf(attrib);
			if (index < 0)
			{
				attribs.pushBack(attrib);
			}
		}
	}
}

void hkQemSimplifier::_calcChooseError(int startIndex, int endIndex, const hkArray<EdgeId>& contractionEdgeIds, hkSimdReal& error)
{
	// I want to only add once. I need to find the unique attribute/group indices
	hkInplaceArray<Attribute, 16> attribs;
	hkInplaceArray<hkVector4, 16> attribBuffer;
	_calcVertexUniqueAttributes(startIndex, attribs);
	
	error.setZero();

	hkQuadricMetric qm;
	hkVectorN targetAttrib;

	// Okay I have all of the unique attribs, I can now see if its a replacement or a move
	for (int i = 0; i < attribs.getSize(); i++)
	{
		const Attribute& attrib = attribs[i];
		Group& group = m_groups[attrib.m_groupIndex];

		Attribute moveTo;
		if (_findMove(attrib, contractionEdgeIds, moveTo))
		{
			// If I move there, then the qem is the sum, and the new
			// attribute value will be the target
			HK_ASSERT(0x32423423, attrib.m_groupIndex == moveTo.m_groupIndex);

			// Work out the error
			_calcSumQems(attrib.m_groupIndex, attrib.m_attributeIndex, moveTo.m_attributeIndex, qm);

			// Load the target attrib
			targetAttrib.alias(group.getAttribute(moveTo.m_attributeIndex), group.m_attributeSize);

			// Add the distance
			error.add(qm.calcDistanceSquared(targetAttrib));
		}
		else if ( !m_discardInvalidAttributeMoves )
		{
			// In this case there is no qem target, but the position changes
			// Arguably other values should change too, such as normals could be recalculated.... but this is all kinda dicey
			// For now I'll just calculate what the error is just with the position changing

			attribBuffer.setSize(group.m_attributeVecSize);
			// Copy over into the attrib buffer
			hkString::memCpy16(attribBuffer.begin(), group.getAttribute(attrib.m_attributeIndex), group.m_attributeVecSize);

			// Get the new position
			//const hkVector4& position = m_positions[endIndex];

			hkVector4 position;
			getPositionAttrib(attrib.m_groupIndex, endIndex, position);
			// Store the new position
			position.store<3,HK_IO_NATIVE_ALIGNED>(((hkReal*)attribBuffer.begin()) + group.m_positionOffset);

			// Load the qem where its at
			qm.load(group.m_attributeSize, (hkReal*)group.getQuadric(attrib.m_attributeIndex));

			// Calc the error
			targetAttrib.alias(attribBuffer.begin(), group.m_attributeSize);
			error.add(qm.calcDistanceSquared(targetAttrib));
		}
		else
		{
			error = hkSimdReal_Max;
		}
	}
}

// The vertex fromIndex is merged into the targetIndex

void hkQemSimplifier::_applyChooseContraction(int fromIndex, int targetIndex)
{
	hkInplaceArray<EdgeId, 16> edgeIds;
	m_topology.findVertexLeavingEdges(fromIndex, edgeIds);

	hkInplaceArray<Attribute, 16> attribs;
	// This is set up as one for one with the unique attribs found... So a lookup on the attribs
	// array, can be used to lookup the target attrib index here
	hkInplaceArray<int, 16> dstAttribIndices;

	// Find all of the edges used in the contraction
	hkInplaceArray<EdgeId, 16> contractionEdgeIds;
	m_topology.findAllEdges(fromIndex, targetIndex, contractionEdgeIds);

	for (int i = 0; i < edgeIds.getSize(); i++)
	{
		const EdgeId edgeId = edgeIds[i];
		Edge edge(edgeId);

		const int triAttribIndex = edge.getIndex(); 
		Triangle* tri = edge.getTriangle();

		Attribute attrib;
		attrib.m_attributeIndex = tri->m_attributeIndices[triAttribIndex];
		attrib.m_groupIndex = tri->m_groupIndex;

		// If doesn't have it, then add it
		const int index = attribs.indexOf(attrib);

		int dstAttribIndex;
		if (index >= 0)
		{
			// Has already been calculated
			dstAttribIndex = dstAttribIndices[index];
		}
		else
		{
			// We need to work out what the new index etc should be
			Group& group = m_groups[attrib.m_groupIndex];

			Attribute moveTo;
			if (_findMove(attrib, contractionEdgeIds, moveTo))
			{
				// Okay there is an attribute it moved to. So make the to attribute the target
				dstAttribIndex = moveTo.m_attributeIndex;

				// We need to accumulate the quadric
				hkVector4* dstQuadric = group.getQuadric(dstAttribIndex);
				const hkVector4* srcQuadric = group.getQuadric(attrib.m_attributeIndex);
				for (int j = 0; j < group.m_quadricVecSize; j++)
				{
					dstQuadric[j].add(srcQuadric[j]);
				}

				group.deleteAttribute(attrib.m_attributeIndex);
			}
			else
			{
				// Couldn't find attribute to move to... soooo just need to update the 
				// quadrics attribute position value
				//const hkVector4& position = m_positions[targetIndex];
				hkVector4 position;
				getPositionAttrib(attrib.m_groupIndex, targetIndex, position );

				// Store the new position... hopefully the quadric will keep subsequent stupid moves
				position.store<3,HK_IO_NATIVE_ALIGNED>(((hkReal*)group.getAttribute(attrib.m_attributeIndex)) + group.m_positionOffset);

				// Set the target attribute that is being used
				dstAttribIndex = attrib.m_attributeIndex;
			}

			// Add the attrib
			attribs.pushBack(attrib);
			dstAttribIndices.pushBack(dstAttribIndex);
		}

		// Set the target index
		tri->m_attributeIndices[triAttribIndex] = dstAttribIndex;
	}

	m_topology.reindexVertexIndex(fromIndex, targetIndex);		
	HK_ASSERT(0xabc5645a, m_topology.calcNumVertexLeavingEdges(fromIndex) == 0);

	// We need to update the boundary edges
	applyContractionOnBoundary(fromIndex, targetIndex);
}

hkQemSimplifier::EdgeContraction* hkQemSimplifier::_createChooseContraction(int start, int end, const hkArray<EdgeId>& contractionEdgeIds)
{
	// Calc the error if end is moved to start
	hkSimdReal startError; _calcChooseError(end, start, contractionEdgeIds, startError);
	// Calc the error if start is moved to the end
	hkSimdReal endError;   _calcChooseError(start, end, contractionEdgeIds, endError);

	// Material border preservation
	if ( m_preserveMaterialBoundary )
	{
		if ( startError.isLess(hkSimdReal_Max) && !contractionPreservesMaterialBoundary(end, start) )	{	startError	= hkSimdReal_Max;	}
		if ( endError.isLess(hkSimdReal_Max)   && !contractionPreservesMaterialBoundary(start, end) )	{	endError	= hkSimdReal_Max;	}
	}

	// Co-planarity check
	if ( m_simplifyCoplanar )
	{
		// Positions must remain co-planar after the contraction
		if ( startError.isLess(hkSimdReal_Max) && !contractionPreservesCoplanarity(end, start) )		{	startError	= hkSimdReal_Max;	}
		if ( endError.isLess(hkSimdReal_Max)   && !contractionPreservesCoplanarity(start, end) )		{	endError	= hkSimdReal_Max;	}

		// Attribute gradients should not change
		if ( startError.isLess(hkSimdReal_Max) && !contractionPreservesAttributes(end, start) )		{	startError	= hkSimdReal_Max;	}
		if ( endError.isLess(hkSimdReal_Max)   && !contractionPreservesAttributes(start, end) )		{	endError	= hkSimdReal_Max;	}
	}

	EdgeContraction* contraction = (EdgeContraction*)m_contractionFreeList.alloc();

	if (startError < endError)
	{
		// Okay we go to the start
		contraction->m_type = EdgeContraction::TYPE_SELECT_START;
		contraction->m_error = startError;
	}
	else
	{
		// We go to the end
		contraction->m_type = EdgeContraction::TYPE_SELECT_END;
		contraction->m_error = endError;
	}

	// Save the indices 
	contraction->m_start = start;
	contraction->m_end = end;	

	// There is no attrib index, as we are just selecting start or end
	contraction->m_attributeIndex = -1;
	contraction->m_groupIndex = -1;
	return contraction;
}

hkQemSimplifier::EdgeContraction* hkQemSimplifier::_createContraction(int start, int end)
{
	// Ensure the start <= end
	if (end < start)
	{
		hkAlgorithm::swap(start, end);
	}

	HK_ASSERT(0x5524a353, start <= end);

	hkUint64 key = (hkUint64(start) << 32) | end;
	if(m_edgeContractionMap.hasKey(key)) 
	{
		return HK_NULL;
	}

	hkInplaceArray<EdgeId, 16> edgeIds;
	m_topology.findAllEdges(start, end, edgeIds);

	// Best I can do is a contraction, as I can't use an inverse to choose the start or the end
	EdgeContraction* contraction = _createChooseContraction(start, end, edgeIds);

	// Add the contraction to the map
	m_edgeContractionMap.insert(key, contraction);
	// Add the contraction to the min error heap
	m_contractions.addEntry(contraction);

	// Done
	return contraction;
}

void hkQemSimplifier::_deleteContraction(EdgeContraction* contraction)
{
	if (contraction->m_type == EdgeContraction::TYPE_NEW)
	{
		// Delete the attribute if needs be
		Group& group = m_groups[contraction->m_groupIndex];
		group.deleteAttribute(contraction->m_attributeIndex);
	}	

	// Remove from the edge -> contraction map
	hkUint64 key = (hkUint64(contraction->m_start) << 32) | contraction->m_end;
	m_edgeContractionMap.remove(key);

	// Remove from the contraction list
	m_contractions.removeEntry(contraction->m_contractionIndex);

	// Mark as invalid.. 
	HK_ON_DEBUG(contraction->m_type = EdgeContraction::TYPE_INVALID;)

	// get rid of its allocation
	m_contractionFreeList.free(contraction);
}

void hkQemSimplifier::discardContraction(EdgeContraction* conn)
{
	_deleteContraction(conn);
}

hkQemSimplifier::EdgeContraction* hkQemSimplifier::findContraction(int start, int end) const
{
	if (end < start)
	{
		hkAlgorithm::swap(start, end);
	}

	hkUint64 key = (hkUint64(start) << 32) | end;
	EdgeContraction* conn = m_edgeContractionMap.getWithDefault(key, HK_NULL);
	return conn;
}

bool hkQemSimplifier::_calcContractions()
{
	const int numTris = m_topology.getNumTriangles();

	for (int i = 0; i < numTris; i++)
	{
		Triangle* tri = m_topology.getTriangle(i);
		for (int j = 0; j < 3; j++)
		{
			Edge edge(tri, j);

			int start = edge.getStart();
			int end = edge.getEnd();

			if (!findContraction(start, end))
			{
				EdgeContraction* contraction = _createContraction(start, end);
				if (contraction && m_controller && !m_controller->allowContraction(*contraction))
				{
					discardContraction(contraction);
				}
			}
		}
	}

	return m_contractions.isOk() ? true : false;
}

hkQemSimplifier::EdgeContraction* hkQemSimplifier::getTopContraction() const
{
	return m_contractions.isEmpty() ? HK_NULL : m_contractions.getTop();
}

bool hkQemSimplifier::doesTopContractionFlipTriangle()
{
	EdgeContraction* contraction = getTopContraction();
	return contraction && _doesContractionFlipTriangle(*contraction);
}

void hkQemSimplifier::discardTopContraction()
{
	EdgeContraction* contraction = getTopContraction();
	HK_ASSERT(0x12123213, contraction);
	if (contraction)
	{
		_deleteContraction(contraction);
	}
}

/* static */void hkQemSimplifier::_calcNormal(hkVector4Parameter a, hkVector4Parameter b, hkVector4Parameter c, hkVector4& normal)
{
	hkVector4 e0; e0.setSub(b, a);
	hkVector4 e1; e1.setSub(c, a);
	normal.setCross(e0, e1);
	if (normal.lengthSquared<3>().isLess(hkSimdReal::fromFloat(hkReal(1e-10f))) )
	{
		normal = hkVector4::getConstant<HK_QUADREAL_1000>();
	}
	else
	{
		normal.normalize<3>();
	}
}

bool hkQemSimplifier::_doesFlipTriangle(int vertexIndex, const EdgeContraction& contraction, hkVector4Parameter newPosition)
{
	hkInplaceArray<hkgpVertexTriangleTopologyBase::Triangle*, 16> tris;
	m_topology.findVertexTriangles(vertexIndex, tris);

	for (int i = 0; i < tris.getSize(); i++)
	{
		Triangle* tri = static_cast<Triangle*>(tris[i]);
		if (tri->hasVertexIndex(contraction.m_start) && tri->hasVertexIndex(contraction.m_end))
		{
			// This tri would be removed so not interesting
			continue;
		}

		// Calculate the normal
		hkVector4 prevNormal;
		_calcNormal(m_positions[tri->m_vertexIndices[0]], m_positions[tri->m_vertexIndices[1]], m_positions[tri->m_vertexIndices[2]], prevNormal);

		// Replace the vertex with the vertex thats being moved
		const int index = tri->findVertexIndex(vertexIndex);
		HK_ASSERT(0x2423423, index >= 0);

		const int v1 = tri->m_vertexIndices[m_topology.NextTriIndex(index)];
		const int v2 = tri->m_vertexIndices[m_topology.PrevTriIndex(index)];

		hkVector4 newNormal;
		_calcNormal(newPosition, m_positions[v1], m_positions[v2], newNormal);
		
		if ( prevNormal.dot<3>(newNormal).isLessZero() )
		{
			return true;
		}
	}

	return false;
}

bool hkQemSimplifier::_doesContractionFlipTriangle(const EdgeContraction& contraction)
{
	hkVector4 newPosition; newPosition.setZero();

	switch (contraction.m_type)
	{
		case EdgeContraction::TYPE_SELECT_END:
		{
			newPosition = m_positions[contraction.m_end];
			break;
		}
		case EdgeContraction::TYPE_SELECT_START:
		{
			newPosition = m_positions[contraction.m_start];
			break;
		}
		case EdgeContraction::TYPE_NEW:
		{
			getAttribPosition(contraction.m_groupIndex, contraction.m_attributeIndex, newPosition);
			break;
		}
		default:
			HK_ASSERT(0x6488ab97, false); 
			break;
	}

	// Returns true if the triangle is flipped
	return _doesFlipTriangle(contraction.m_start, contraction, newPosition) || _doesFlipTriangle(contraction.m_end, contraction, newPosition);
}


void hkQemSimplifier::applyTopContraction()
{
	EdgeContraction* contraction = getTopContraction();
	if (contraction)
	{
		_applyContraction(contraction);
	}
}

hkSimdReal hkQemSimplifier::getTopContractionError() const
{
	EdgeContraction* contraction = getTopContraction();
	if (contraction) 
		return contraction->m_error;
	return hkSimdReal_Max;
}

void hkQemSimplifier::_deleteVertexContractions(VertexIndex index)
{
	// Delete all of the contractions
	hkInplaceArray<EdgeId, 16> edges;
	m_topology.findAllVertexEdges(index, edges);

	for (int i = 0; i < edges.getSize(); i++)
	{
		Edge edge(edges[i]);
		EdgeContraction* contraction = findContraction(edge.getStart(), edge.getEnd());
		if (contraction)
		{
			_deleteContraction(contraction);
		}
	}
}

void hkQemSimplifier::_applyContraction(EdgeContraction* contractionIn)
{
	EdgeContraction contractionCopy = *contractionIn;

	// Mark as used so when we delete all, we won't delete the attribute created
	contractionIn->m_type = EdgeContraction::TYPE_USED;

	_deleteVertexContractions(contractionCopy.m_start);
	_deleteVertexContractions(contractionCopy.m_end);

	// I shouldn't be able to find the contraction in the heap anymore
	HK_ASSERT(0x425ba134, m_contractions.getContents().indexOf(contractionIn) < 0);

	// Find the contraction edges
	hkInplaceArray<EdgeId, 16> contractionEdges;
	m_topology.findAllEdges(contractionCopy.m_start, contractionCopy.m_end, contractionEdges);

	// Work space for edges
	hkInplaceArray<EdgeId, 16> edges;

	int positionIndex;
	switch (contractionCopy.m_type)
	{
		case EdgeContraction::TYPE_SELECT_END:
		{
			_applyChooseContraction(contractionCopy.m_start, contractionCopy.m_end);
			positionIndex = contractionCopy.m_end;
			break;
		}
		case EdgeContraction::TYPE_SELECT_START:
		{
			_applyChooseContraction(contractionCopy.m_end, contractionCopy.m_start);
			positionIndex = contractionCopy.m_start;
			break;
		}
		case EdgeContraction::TYPE_NEW:
		{
			// We are going to merge into the start
			positionIndex = contractionCopy.m_start; 

			getAttribPosition(contractionCopy.m_groupIndex, contractionCopy.m_attributeIndex, m_positions[positionIndex] );
			// Reindex
			m_topology.reindexVertexIndex(contractionCopy.m_end, contractionCopy.m_start);	

			HK_ASSERT(0xabc5645a, m_topology.calcNumVertexLeavingEdges(contractionCopy.m_end) == 0);

			// make all use the new attribute 
			m_topology.findVertexLeavingEdges(contractionCopy.m_start, edges);
			for (int i = 0; i < edges.getSize(); i++)
			{
				Edge edge(edges[i]);
				Triangle* tri = edge.getTriangle();

				// Only alter if they are in the same group as the contraction
				if (tri->m_groupIndex == contractionCopy.m_groupIndex)
				{
					tri->m_attributeIndices[edge.getIndex()] = contractionCopy.m_attributeIndex;
				}
			}
			break;
		}
		default:
		{
			HK_WARN(0x243a3242, "Invalid type");
			return;
		}
	}

	// Delete the triangles that share the contraction edge
	{
		hkInplaceArray<hkgpVertexTriangleTopologyBase::Triangle*, 16> tris;
		// Delete all the degenerate triangles
		m_topology.findAllEdges(positionIndex, positionIndex, edges);
		m_topology.uniqueTriangles(edges, tris);

		//m_topology.uniqueTriangles(contractionEdges, tris);
		for (int i = 0; i < tris.getSize(); i++)
		{
			m_topology.deleteTriangle(tris[i]);
		}
	}

	// Recreate the contractions
	{
		m_topology.findVertexLeavingEdges(positionIndex, edges);

		for (int i = 0; i < edges.getSize(); i++)
		{
			Edge edge(edges[i]);

			// Force the attribs to be at the right position
			Triangle* tri = edge.getTriangle();
			int attribIndex = tri->m_attributeIndices[edge.getIndex()];

			// Work out the position in attrib space
			hkVector4 position;
			getPositionAttrib(tri->m_groupIndex, positionIndex, position);

			Group& group = m_groups[tri->m_groupIndex];
			group.setAttributePosition(attribIndex, position);

			// Handle the contraction
			const int start = edge.getStart();
			const int end = edge.getEnd();
			EdgeContraction* contraction = findContraction(start, end);
			HK_ASSERT(0x234234, contraction == HK_NULL || contraction->m_type != EdgeContraction::TYPE_USED);
			if (!contraction)
			{
				contraction = _createContraction(start, end);

				if (contraction && m_controller && !m_controller->allowContraction(*contraction))
				{
					discardContraction(contraction);
				}
			}
		}

		// Handle any returning edges
		{
			hkInplaceArray<EdgeId, 16> returningEdges;
			m_topology.findVertexReturningEdges(positionIndex, edges, returningEdges);
			// Find any returning edges
			for (int i = 0; i < returningEdges.getSize(); i++)
			{
				Edge edge(returningEdges[i]);

				// Handle the contraction
				const int start = edge.getStart();
				const int end = edge.getEnd();
				EdgeContraction* contraction = findContraction(start, end );
				HK_ASSERT(0x234234, contraction == HK_NULL || contraction->m_type != EdgeContraction::TYPE_USED);

				if (!contraction)
				{
					contraction = _createContraction(start, end);

					if (contraction && m_controller && !m_controller->allowContraction(*contraction))
					{
						discardContraction(contraction);
					}
				}
			}
		}
	}

	// Check all is okay
	if(! isOk())
	{
		HK_WARN(0x23444111, "Inconsistent contractions");
	}
}

void hkQemSimplifier::finalize()
{
	HK_ASSERT(0x1321321, !m_isFinalized);
	if (m_isFinalized) 
	{
		return;
	}

	const int numGroups = m_groups.getSize();
	hkArray<int> attributeMap;
	// For positions
	hkArray<hkVector4> positions;
	hkArray<int> positionMap(m_positions.getSize(), -1);

	const int numTris = m_topology.getNumTriangles();

	for (int i = 0; i < numGroups; i++)
	{
		Group& group = m_groups[i];

		hkArray<hkVector4*>& srcAttributes = group.m_attributes;

		attributeMap.clear();
		attributeMap.setSize(srcAttributes.getSize(), -1);

        hkArray<hkVector4*> dstAttributes;

		for (int j = 0; j < numTris; j++)
		{
			Triangle* tri = m_topology.getTriangle(j);
			if (tri->m_groupIndex != i)
			{
				continue;
			}

			for (int k = 0; k < 3; k++)
			{
				const int srcPositionIndex = tri->m_vertexIndices[k];
				int dstPositionIndex = positionMap[srcPositionIndex];
				if (dstPositionIndex < 0)
				{
					dstPositionIndex = positions.getSize();
					positions.pushBack(m_positions[srcPositionIndex]);
					positionMap[srcPositionIndex] = dstPositionIndex;
				}

				const int srcAttribIndex = tri->m_attributeIndices[k];
				int dstAttribIndex = attributeMap[srcAttribIndex];
				if (dstAttribIndex < 0)
				{
                    dstAttribIndex = dstAttributes.getSize();
                    dstAttributes.pushBack(srcAttributes[srcAttribIndex]);

					// Copy to the destination
					attributeMap[srcAttribIndex] = dstAttribIndex;
				}

				// I can do this - because attributes aren't handled by the topology
				tri->m_attributeIndices[k] = dstAttribIndex;

				// I CAN'T do this otherwise the topologies mapping will be broken, so I do with the remap on the topology after the table is all set up
				//tri->m_vertexIndices[k] = dstPositionIndex;
			}

            // Okay we need to free any attributes that were not indexed
		}

        for (int j = 0; j < srcAttributes.getSize(); j++)
        {
            if (attributeMap[j] < 0 && srcAttributes[j])
            {
                group.m_attributeFreeList.free(srcAttributes[j]);
            }
        }

        group.m_availableAttributeIndices.clear();
		// All the attributes for the group have been remapped -> can use the newly built list
		srcAttributes.swap(dstAttributes);
	}

	// Remap all of the vertex indices
	m_topology.remapVertexIndices(positionMap);

	m_positions.swap(positions);

	// The attributes are just what is now indexed
	m_contractions.clear();
	m_contractionFreeList.freeAll();
	m_edgeContractionMap.clear();

	m_isFinalized = true;
}

void hkQemSimplifier::applyContractions(int numContractions, bool allowFlip)
{
	for (int j = 0; j < numContractions && getTopContraction(); j++)
	{
		if (allowFlip)
		{
			applyTopContraction();
			continue;
		}

		if (doesTopContractionFlipTriangle())
		{
			discardTopContraction();
			j--;
		}
		else
		{
			applyTopContraction();
		}
	}
}

void hkQemSimplifier::simplify(hkSimdRealParameter percentageRemoveTris, bool allowFlip)
{
	HK_ASSERT(0x423a4324, percentageRemoveTris.isGreaterEqualZero() && percentageRemoveTris.isLessEqual(hkSimdReal_1));
	hkSimdReal nT; nT.setFromInt32(m_topology.getNumTriangles());
	hkSimdReal targetN = nT * (hkSimdReal_1 - percentageRemoveTris) + hkSimdReal_Half;
	int targetTris; targetN.storeSaturateInt32(&targetTris);

	while (getTopContraction() && m_topology.getNumTriangles() > targetTris)
	{
		if (allowFlip || !doesTopContractionFlipTriangle())
		{
			applyTopContraction();		
		}
		else
		{
			discardTopContraction();
		}
	}
}

void hkQemSimplifier::calcNumTrianglesPerGroup(hkArray<int>& trisPerGroup) const
{
	trisPerGroup.clear();
	trisPerGroup.setSize(m_groups.getSize(), 0);

	const int numTris = m_topology.getNumTriangles();

	for (int i = 0; i < numTris; i++)
	{
		const Triangle* tri = m_topology.getTriangle(i);
		trisPerGroup[tri->m_groupIndex]++;
	}
}

hkBool32 hkQemSimplifier::_areAttributesOk() const
{
	{
		hkArray<int> attribMap;
		// Lets make sure an attribute index only maps to a single position
		for (int i = 0; i < m_groups.getSize(); i++)
		{
			const Group& group = m_groups[i];
			attribMap.clear();
			attribMap.setSize(group.m_attributes.getSize(), -1);

			for (int j = 0; j < m_topology.getNumTriangles(); j++)
			{
				const Triangle* tri = m_topology.getTriangle(j);
				if (tri->m_groupIndex != i)
				{
					continue;
				}

				for (int k = 0; k < 3; k++)
				{
					const int attribIndex = tri->m_attributeIndices[k];

					int posIndex = attribMap[attribIndex];
					if (posIndex >= 0)
					{
						if (tri->m_vertexIndices[k] != VertexIndex(posIndex))
						{
							// There can be lots of attributes - but they must map down to a single vertex
							return false;
						}
					}
					else
					{
						// Store the vertex index associated with this attribute
						attribMap[attribIndex] = tri->m_vertexIndices[k];
					}
				}
			}
		}
	}
	
#if 0
	{
		// Check the positions + attribute positions are in the same place
		const int numTris = m_topology.getNumTriangles();
		for (int i = 0; i < numTris; i++)
		{
			const Triangle* tri = m_topology.getTriangle(i);

			const Group& group = m_groups[tri->m_groupIndex];

			hkVector4 attribPos;
			
			for (int j = 0; j < 3; j++)
			{
				attribPos.load<3>(group.getAttributePosition(tri->m_attributeIndices[j]));
				const hkVector4& pos = m_positions[tri->m_vertexIndices[j]];

				if (!pos.equals3(attribPos, 0.0f))
				{
					return false;
				}
			}
		}
	}
#endif

	for (int i = 0; i < m_groups.getSize(); i++)
	{
		const Group& group = m_groups[i];
		const hkArray<hkVector4*>& attribs = group.m_attributes;

		// Check all of the free ones are zero'd
		for (int j = 0; j < group.m_availableAttributeIndices.getSize(); j++)
		{
			int index = group.m_availableAttributeIndices[j];
			if (attribs[index] != HK_NULL)
			{
				return false;
			}
		}
		
		for (int j = 0; j < m_topology.getNumTriangles(); j++)
		{
			const Triangle* tri = m_topology.getTriangle(j);
			if (tri->m_groupIndex != i)
			{
				continue;
			}

			// See if all the attributes are okay
			for (int k = 0; k < 3; k++)
			{
				if (attribs[tri->m_attributeIndices[k]] == HK_NULL)
				{
					return false;
				}
			}
		}
	}

	{
		// Make sure all the contractions with new attributes are valid
		const hkArray<EdgeContraction*>& contractions = m_contractions.getContents();
		for (int i = 0; i < contractions.getSize(); i++)
		{
			EdgeContraction* contraction = contractions[i];

			if (contraction->m_type != EdgeContraction::TYPE_NEW)
			{
				continue;
			}

			const Group& group = m_groups[contraction->m_groupIndex];

			if (group.m_attributes[contraction->m_attributeIndex] == HK_NULL)
			{
				return false;
			}
		}
	}

	return true;
}

hkBool32 hkQemSimplifier::isOk() const
{
	hkBool32 ok = m_topology.isOk();
	hkBool32 closed = true;
	if (!m_enableOpenEdgeGeometry)
	{
		closed = m_topology.isClosed();
	}
	hkBool32 contractions = m_contractions.isOk();
	hkBool32 attribs = _areAttributesOk();

	if (ok && contractions && attribs && closed)
	{
		return true;
	}
	return false;
}

void hkQemSimplifier::debugDisplay()
{
	// Min and max
	const hkArray<EdgeContraction*>& contractions = m_contractions.getContents();
	if (contractions.getSize() <= 0)
	{
		return;
	}

	hkSimdReal max = contractions[0]->m_error;
	hkSimdReal min = max;

	for (int i = 0; i < contractions.getSize(); i++)
	{
		EdgeContraction* contraction = contractions[i];
		max.setMax(max, contraction->m_error);
		min.setMin(min, contraction->m_error);
	}

	const hkArray<hkVector4>& positions = m_positions;

	for (int i = 0; i < contractions.getSize(); i++)
	{
		EdgeContraction* contraction = contractions[i];

		// Color
		hkSimdReal val = (contraction->m_error - min) / (max - min);

		hkUint32 intensity; (hkSimdReal_255 - (val * hkSimdReal_255)).storeSaturateInt32((hkInt32*)&intensity);
		// Red varying with error
		hkColor::Argb color = hkColor::fromArgb( 0xff, static_cast<unsigned char>( intensity ), 0, 0 );
		
		if (i != 0)
		{			
			HK_DISPLAY_LINE(positions[contraction->m_start], positions[contraction->m_end], color);
		}

		{
			hkVector4 middle;
			middle.setAdd(m_positions[contraction->m_start], m_positions[contraction->m_end]);
			middle.mul(hkSimdReal_Half);

			hkStringBuf text; text.printf("%f", contraction->m_error.getReal());

			HK_DISPLAY_3D_TEXT(text.cString(), middle, hkColor::GREEN);
		}
	}

	{
		EdgeContraction* contraction = contractions[0];
		HK_DISPLAY_LINE(positions[contraction->m_start], positions[contraction->m_end], hkUint32(hkColor::GREEN));

		hkVector4 p;
		switch (contraction->m_type)
		{
			default:
			case EdgeContraction::TYPE_SELECT_START:
			{
				p = m_positions[contraction->m_start];
				break;
			}
			case EdgeContraction::TYPE_SELECT_END:
			{
				p = m_positions[contraction->m_end];
				break;
			}
			case EdgeContraction::TYPE_NEW:
			{
				m_groups[contraction->m_groupIndex].getPosition(contraction->m_attributeIndex, p);
				break;
			}
		}

		HK_DISPLAY_STAR(p, hkReal(0.25f), hkColor::CYAN);
	}
}

//
//	Sets the boundary edge


HK_FORCE_INLINE void hkQemSimplifier::BoundaryEdge::set(int vA, int vB)
{
	if ( vA > vB )	{	m_start = vB;	m_end = vA;	}
	else			{	m_start = vA;	m_end = vB;	}
}

//
//	Compares two boundary edges

HK_FORCE_INLINE bool hkQemSimplifier::BoundaryEdge::equals(const BoundaryEdge& be) const
{
	return ( (m_start == be.m_start) && (m_end == be.m_end) );
}

//
//	Returns true if the edge contains the given vertex

HK_FORCE_INLINE bool hkQemSimplifier::BoundaryEdge::containsVertex(int v) const
{
	return (m_start == v) || (m_end == v);
}

//
//	Replaces srcVertex with dstVertex

HK_FORCE_INLINE void hkQemSimplifier::BoundaryEdge::replaceVertex(int srcVertex, int dstVertex)
{
	HK_ASSERT(0x1e5d72bc, containsVertex(srcVertex));
	if ( m_start == srcVertex )	{	m_start = dstVertex;	}
	else						{	m_end = dstVertex;		}
}

//
//	Returns true if the edge is degenerate

HK_FORCE_INLINE bool hkQemSimplifier::BoundaryEdge::isDegenerate() const
{
	return (m_start == m_end);
}

//
//	Creates the list of material boundary edges

void hkQemSimplifier::findMaterialBoundaries()
{
	hkArray<EdgeId> edges;
	m_materialBoundaries.setSize(0);

	const int numTris = m_topology.getNumTriangles();
	for (int ti = 0; ti < numTris; ti++)
	{
		const Triangle* tri = m_topology.getTriangle(ti);
		
		// Look at the triangle edges. If any edge has another adjacent triangle with the same material,
		// then this is not a boundary edge
		for (int vi = 0; vi < 3; vi++)
		{
			const int vii = (1 << vi) & 3;
			const int vA = tri->m_vertexIndices[vi];
			const int vB = tri->m_vertexIndices[vii];

			edges.setSize(0);
			m_topology.findAllEdges(vA, vB, edges);

			// Get all triangles adjacent to these edges
			const int numEdges = edges.getSize();
			bool isBoundaryEdge = true;
			for (int ei = 0; ei < numEdges; ei++)
			{
				const Edge e(edges[ei]);
				const Triangle* otherTri = (const Triangle*)e.getTriangle();
				if ( (otherTri != tri) && (otherTri->m_groupIndex == tri->m_groupIndex) )
				{
					isBoundaryEdge = false;
					break;
				}
			}

			// If we found a boundary edge, save it!
			if ( isBoundaryEdge && (findBoundaryEdge(vA, vB) < 0) )
			{
				addBoundaryEdge(vA, vB);
			}
		}
	}
}

//
//	Locates the boundary edge and returns its index, -1 if nothing was found

HK_FORCE_INLINE int hkQemSimplifier::findBoundaryEdge(int srcVertex, int  dstVertex)
{
	BoundaryEdge be;
	be.set(srcVertex, dstVertex);

	for (int k = m_materialBoundaries.getSize() - 1; k >= 0; k--)
	{
		if ( m_materialBoundaries[k].equals(be) )
		{
			return k;
		}
	}

	return -1;
}

//
//	Adds a boundary edge

HK_FORCE_INLINE void hkQemSimplifier::addBoundaryEdge(int srcVertex, int  dstVertex)
{
	BoundaryEdge be;
	be.set(srcVertex, dstVertex);
	m_materialBoundaries.pushBack(be);
}

//
//	Tests whether the given contraction (srcVertex -> dstVertex) preserves the material border

bool hkQemSimplifier::contractionPreservesMaterialBoundary(int srcVertex, int dstVertex)
{
	const hkVector4& point = m_positions[srcVertex];
	hkSimdReal maxDistance;
	maxDistance.setZero();

	// Find all boundary edges containing srcVertex. The boundary error is the distance from srcVertex to the
	// modified boundary edges
	const int numEdges = m_materialBoundaries.getSize();
	for (int ei = 0; ei < numEdges; ei++)
	{
		BoundaryEdge be = m_materialBoundaries[ei];
		if ( be.containsVertex(srcVertex) )
		{
			be.replaceVertex(srcVertex, dstVertex);
			if ( !be.isDegenerate() )
			{
				// We can compute the distance from srcVertex to the new boundary edge
				const hkVector4& lineStart = m_positions[be.m_start];
				const hkVector4& lineEnd = m_positions[be.m_end];

				// Compute projection fraction.
				hkVector4	delta; delta.setSub(lineStart, lineEnd);
				hkVector4	origin; origin.setSub(lineStart, point);
				hkSimdReal	fraction = delta.dot<3>(origin) / delta.lengthSquared<3>();

				// Compute distance and set projectionOut if required.
				hkVector4	projection; projection.setSubMul(origin, delta, fraction);
				hkSimdReal projLength = projection.length<3>();
				maxDistance.setMax(maxDistance, projLength);
			}
		}
	}

	// Compare with position tolerance
	return ( maxDistance < m_materialBoundaryTolerance );
}

//
//	Tests whether the given contraction (srcVertex -> dstVertex) preserves the coplanarity

bool hkQemSimplifier::contractionPreservesCoplanarity(int srcVertex, int dstVertex)
{
	const hkVector4& vP = m_positions[srcVertex];
	hkSimdReal maxDistance; maxDistance.setZero();

	// Find all triangles around srcVertex
	hkArray<EdgeId> eids;
	m_topology.findAllVertexEdges(srcVertex, eids);
	const int numEdges = eids.getSize();
	for (int ei = 0; ei < numEdges; ei++)
	{
		const EdgeId eid = eids[ei];
		const Edge edge(eid);
		const Triangle* tri = edge.getTriangle();

		// Get triangle verts and virtually perform the contraction
		int triVerts[3];
		for (int k = 0; k < 3; k++)
		{
			const int vidx = tri->m_vertexIndices[k];
			triVerts[k] = (vidx == srcVertex) ? dstVertex : vidx;
		}
		if ( (triVerts[0] == triVerts[1]) || (triVerts[1] == triVerts[2]) || (triVerts[2] == triVerts[0]) )
		{
			continue;	// Triangle is degenerate, ignore!
		}

		// Get triangle verts after contraction
		const hkVector4& vA = m_positions[triVerts[0]];
		const hkVector4& vB = m_positions[triVerts[1]];
		const hkVector4& vC = m_positions[triVerts[2]];

		// Compute distance from point to plane ABC
		hkVector4 vAB;		vAB.setSub(vB, vA);
		hkVector4 vAC;		vAC.setSub(vC, vA);
		hkVector4 vN;		vN.setCross(vAB, vAC);
		hkVector4 vAP;		vAP.setSub(vP, vA);
		hkSimdReal dist;	dist.setAbs(vAP.dot<3>(vN));

		const hkSimdReal maxDist = m_coplanarTolerance * vN.length<3>();
		if ( dist.isGreaterEqual(maxDist) )
		{
			return false;
		}
	}

	return true;
}

//
//	Tests whether the given contraction (srcVertex -> dstVertex) preserves the attributes

bool hkQemSimplifier::contractionPreservesAttributes(int srcVertex, int dstVertex)
{
	// Get all unique attributes of srcVertex
	hkArray<Attribute> srcAttribs;
	_calcVertexUniqueAttributes(srcVertex, srcAttribs);

	// Compute the attribs after the contraction
	hkArray<Attribute> dstAttribs;
	{
		const int numAttribs = srcAttribs.getSize();
		dstAttribs.setSize(numAttribs);

		hkInplaceArray<EdgeId, 16> edgeIds;
		m_topology.findAllEdges(srcVertex, dstVertex, edgeIds);
		for (int k = 0; k < numAttribs; k++)
		{
			HK_ON_DEBUG(hkBool32 foundMove = )_findMove(srcAttribs[k], edgeIds, dstAttribs[k]);
			HK_ASSERT(0x72a81602, foundMove);
		}
	}

	hkArray<hkReal> interpAttribute;

	// For all other triangles that will be moved, interpolate their attribs and test the difference
	// If the difference is too big, we should not run the contraction.
	hkArray<EdgeId> eids;
	m_topology.findAllVertexEdges(srcVertex, eids);
	const int numEdges = eids.getSize();
	for (int ei = 0; ei < numEdges; ei++)
	{
		const EdgeId eid = eids[ei];
		const Edge edge(eid);
		const Triangle* tri = edge.getTriangle();

		// Locate the moving vertex
		for (int via = 0; via < 3; via++)
		{
			if ( tri->m_vertexIndices[via] != (hkUint32)srcVertex )
			{
				continue;
			}

			// Determine the other vertices
			const int vib = (1 << via) & 3;
			const int vic = (1 << vib) & 3;

			// Ignore degenerate triangles
			if ( (tri->m_vertexIndices[vib] == (hkUint32)dstVertex) || (tri->m_vertexIndices[vic] == (hkUint32)dstVertex) )
			{
				continue;
			}

			// This triangle will have one of its vertices moved. Determine the barycentric coordinates of the new point vA1 w.r.t. triangle (vA0, vB, vC), 
			// estimate its parameters in the new point and compare with the attribute of the end-point.
			computeInterpolatedAttribute(tri, m_positions[dstVertex], interpAttribute);

			// Locate destination attribute
			int k = dstAttribs.getSize() - 1;
			for (; k >= 0; k--)
			{
				if ( (srcAttribs[k].m_attributeIndex == tri->m_attributeIndices[via]) && (srcAttribs[k].m_groupIndex == tri->m_groupIndex) )
				{
					break;
				}
			}
			HK_ASSERT(0x6cc755b2, k >= 0);

			// If the interpolated attribute is too different from the target attribute, fail!
			const Attribute tgtAttrib = dstAttribs[k];
			if ( !attributeIsApproxEqual(interpAttribute, tgtAttrib) )
			{
				return false;
			}
		}
	}

	return true;
}

//
//	Loads a variable sized attribute

static HK_FORCE_INLINE void HK_CALL loadElements(hkVector4& v, const hkReal* ptr, int numElements)
{
	HK_ALIGN_REAL(hkReal tmp[4]) = { 0, 0, 0, 0 };
	for (int k = numElements - 1; k >= 0; k--)
	{
		tmp[k] = ptr[k];
	}
	v.load<4>(&tmp[0]);
}

//
//	Computes an interpolated attribute for the given vertex relative to the given triangle

void hkQemSimplifier::computeInterpolatedAttribute(const Triangle* tri, hkVector4Parameter vP, hkArray<hkReal>& attribOut)
{
	// Get the triangle vertices
	const hkVector4& vA = m_positions[tri->m_vertexIndices[0]];
	const hkVector4& vB = m_positions[tri->m_vertexIndices[1]];
	const hkVector4& vC = m_positions[tri->m_vertexIndices[2]];
	
	// Compute the barycentric coordinates of P w.r.t. ABC
	hkVector4 vBary;
	hkcdTriangleUtil::calcBarycentricCoordinates(vP, vA, vB, vC, vBary);

	// Get triangle attributes
	const Group& mtlGroup = m_groups[tri->m_groupIndex];
	const hkVector4* aA = mtlGroup.getAttribute(tri->m_attributeIndices[0]);
	const hkVector4* aB = mtlGroup.getAttribute(tri->m_attributeIndices[1]);
	const hkVector4* aC = mtlGroup.getAttribute(tri->m_attributeIndices[2]);

	// Alloc output attribute
	attribOut.setSize(mtlGroup.m_attributeVecSize * sizeof(hkVector4) / sizeof(hkReal));
	hkVector4* aP = (hkVector4*)attribOut.begin();

	// Interpolate each attribute
	for (int k = 0; k < mtlGroup.m_attributeVecSize; k++)
	{
		// Interpolate
		hkVector4 vAttribP;
		vAttribP.setMul(vBary.getComponent<0>(), aA[k]);
		vAttribP.addMul(vBary.getComponent<1>(), aB[k]);
		vAttribP.addMul(vBary.getComponent<2>(), aC[k]);

		// Store
		aP[k] = vAttribP;
	}
}

//
//	Returns true if the given attribute is almost equal with another

bool hkQemSimplifier::attributeIsApproxEqual(const hkArray<hkReal>& attribIn, const Attribute& tgtAttribute)
{
	// Get the group, format, and attribute array
	const Group& mtlGroup		= m_groups[tgtAttribute.m_groupIndex];
	const AttributeFormat& fmt	= mtlGroup.m_fmt;
	const hkReal* myAttribPtr	= attribIn.begin();
	const hkReal* tgtAttribPtr	= (const hkReal*)mtlGroup.getAttribute(tgtAttribute.m_attributeIndex);

	const int numEntries = fmt.m_entries.getSize();
	for (int k = 0, offset = 0; k < numEntries; k++)
	{
		const AttributeEntry& ae = fmt.m_entries[k];
		switch ( ae.m_type )
		{
		case ATTRIB_TYPE_NORMAL:
		case ATTRIB_TYPE_BINORMAL:
		case ATTRIB_TYPE_TANGENT:
			{
				hkVector4 vN1;	loadElements(vN1, &myAttribPtr[offset], ae.m_size);
				hkVector4 vN2;	loadElements(vN2, &tgtAttribPtr[offset], ae.m_size);

				// Compute dot and magnitudes
				hkVector4 vDots;	hkVector4Util::dot3_3vs3(vN1, vN2, vN1, vN1, vN2, vN2, vDots);

#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
				hkVector4 vDots_sqrt; vDots_sqrt.setSqrt(vDots);
				const hkSimdReal len1	= vDots_sqrt.getComponent<1>();		// Length(vN1)
				const hkSimdReal len2	= vDots_sqrt.getComponent<2>();		// Length(vN2)
#else
				const hkSimdReal len1	= vDots.getComponent<1>().sqrt();		// Length(vN1)
				const hkSimdReal len2	= vDots.getComponent<2>().sqrt();		// Length(vN2)
#endif
				const hkSimdReal len12	= len1 * len2;							// Length(vN1) * Length(vN2)
				const hkSimdReal dot12	= vDots.getComponent<0>();				// Dot(vN1, vN2)
				const hkSimdReal relErr	= len12 - dot12;						// Length(vN1) * Length(vN2) - Dot(vN1, vN2)
				hkSimdReal lenErr;	lenErr.setAbs(len1 - len2);

				const hkSimdReal absTol	=	hkSimdReal::fromFloat(	(ae.m_type == ATTRIB_TYPE_NORMAL)		?	m_thresholds.m_thresholds[hkVertexFormat::USAGE_NORMAL]	:
																	( ae.m_type == ATTRIB_TYPE_BINORMAL )	?	m_thresholds.m_thresholds[hkVertexFormat::USAGE_BINORMAL]:
																												m_thresholds.m_thresholds[hkVertexFormat::USAGE_TANGENT]);
				const hkSimdReal relTol = len12 * hkSimdReal::fromFloat(m_thresholds.m_angularThreshold);
				if ( (relErr > relTol) || (lenErr > absTol) )
				{
					return false;
				}
			}
			break;

		case ATTRIB_TYPE_UV:
			{
				hkVector4 vUv1;	loadElements(vUv1, &myAttribPtr[offset], ae.m_size);
				hkVector4 vUv2;	loadElements(vUv2, &tgtAttribPtr[offset], ae.m_size);
				hkVector4 vErr;	vErr.setSub(vUv1, vUv2);
				const hkSimdReal err = vErr.length<3>();
				const hkSimdReal tol = hkSimdReal::fromFloat(m_thresholds.m_thresholds[hkVertexFormat::USAGE_TEX_COORD]);
				if ( err > tol )
				{
					return false;
				}
			}
			break;
		}

		offset += ae.m_size;
	}

	// Attributes are almost equal
	return true;
}

//
//	Propagates a contraction (srcVertex -> dstVertex) to the boundary edges

void hkQemSimplifier::applyContractionOnBoundary(int srcVertex, int dstVertex)
{
	const int numEdges = m_materialBoundaries.getSize();
	for (int k = numEdges - 1; k >= 0; k--)
	{
		BoundaryEdge& be = m_materialBoundaries[k];
		if ( be.containsVertex(srcVertex) )
		{
			be.replaceVertex(srcVertex, dstVertex);
			if ( be.isDegenerate() )
			{
				m_materialBoundaries.removeAt(k);
			}
		}
	}
}

//
//	Sets the format

void hkQemSimplifier::AttributeFormat::set(const hkVertexFormat& vfmt)
{
	const int numEles = vfmt.m_numElements;

	for (int i = 0; i < numEles; i++)
	{
		const hkVertexFormat::Element& elt = vfmt.m_elements[i];

		switch ( elt.m_dataType )
		{
		case hkVertexFormat::TYPE_FLOAT32:
			{
				AttributeEntry ae;
				ae.m_type	= (hkUint8)ATTRIB_TYPE_UNKNOWN;
				ae.m_size	= elt.m_numValues;

				switch ( elt.m_usage )
				{
				case hkVertexFormat::USAGE_NORMAL:		{	ae.m_type	= (elt.m_numValues == 3) ? (hkUint8)ATTRIB_TYPE_NORMAL : ae.m_type;							break;	}
				case hkVertexFormat::USAGE_BINORMAL:	{	ae.m_type	= (elt.m_numValues == 3) ? (hkUint8)ATTRIB_TYPE_BINORMAL : ae.m_type;						break;	}		
				case hkVertexFormat::USAGE_TANGENT:		{	ae.m_type	= (elt.m_numValues == 3) ? (hkUint8)ATTRIB_TYPE_TANGENT : ae.m_type;						break;	}
				case hkVertexFormat::USAGE_POSITION:	{	ae.m_type	= (elt.m_subUsage == 0) ? (hkUint8)ATTRIB_TYPE_POSITION : ae.m_type;						break;	}
				case hkVertexFormat::USAGE_COLOR:		{	ae.m_type	= (elt.m_numValues >= 1 && elt.m_numValues <= 4) ? (hkUint8)ATTRIB_TYPE_COLOR : ae.m_type;	break;	}
				case hkVertexFormat::USAGE_TEX_COORD:	{	ae.m_type	= (hkUint8)ATTRIB_TYPE_UV;																	break;	}
				default:								{																											break;	}
				}
				
				m_entries.pushBack(ae);
			}
			break;

		case hkVertexFormat::TYPE_ARGB32:
			{
				for (int k = 0; k < elt.m_numValues; k++)
				{
					AttributeEntry ae;
					ae.m_type	= ATTRIB_TYPE_COLOR;
					ae.m_size	= 4;

					m_entries.pushBack(ae);
				}
				break;
			}

		default:
			{
				HK_ASSERT(0x3244a3a2, !"Unhandled type");
			}
		}
	}
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
