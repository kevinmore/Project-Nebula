/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h>
#include <Common/GeometryUtilities/Mesh/Utils/NormalCalculator/hkNormalCalculator.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>
#include <Common/GeometryUtilities/Mesh/Utils/FindUniqueIndicesUtil/hkFindUniqueIndicesUtil.h>

static HK_FORCE_INLINE hkUint32 calcEdgeId(int a, int b) { return hkUint32(a) << 16 | hkUint32(b); }

/* static */void HK_CALL hkNormalCalculator::calculateSmoothingGroups(const hkArray<hkVector4>& positions, const hkUint16* triangleList, int numTriangles, hkReal cosSmoothingAngle, hkArray<int>& triangleIndicesOut, hkArray<int>& numTrianglesIndicesOut)
{
	hkSimdReal cosSmoothAngle = hkSimdReal::fromFloat(cosSmoothingAngle);
	if (cosSmoothAngle.isGreaterEqual(hkSimdReal_1))
	{
		// Need a little bit of wiggle to take into account calculation errors
		cosSmoothAngle = hkSimdReal::fromFloat(1.0f - 1e-4f);
	}

    triangleIndicesOut.clear();
    numTrianglesIndicesOut.clear();

    // Edge map, specifies which triangle the edge belongs to (edges are directed)
    hkPointerMap<hkUint32, int> edgeMap;

    // Work out the normal to each triangle
    hkLocalArray<hkVector4> triangleNormals(numTriangles);
    triangleNormals.setSize(numTriangles);

    {
        const hkUint16* triIndices = triangleList;
        for (int i = 0; i < numTriangles; i++, triIndices += 3)
        {
            // Get the indices
            int v0Index = triIndices[0];
            int v1Index = triIndices[1];
            int v2Index = triIndices[2];

            // Get the vertices
            const hkVector4& v0 = positions[v0Index];
            const hkVector4& v1 = positions[v1Index];
            const hkVector4& v2 = positions[v2Index];

            hkVector4 e0; e0.setSub(v1, v0);
            hkVector4 e1; e1.setSub(v2, v0);
            hkVector4& normal = triangleNormals[i];

            normal.setCross(e0, e1);
            if (normal.lengthSquared<3>().isLess(hkSimdReal_Eps))
            {
                // Its degenerate
				normal = hkVector4::getConstant<HK_QUADREAL_1000>();
            }
            else
            {
                normal.normalize<3,HK_ACC_12_BIT,HK_SQRT_SET_ZERO>();
            }

            // Set up the edge map
            edgeMap.insert(calcEdgeId(v0Index, v1Index), i);
            edgeMap.insert(calcEdgeId(v1Index, v2Index), i);
            edgeMap.insert(calcEdgeId(v2Index, v0Index), i);
        }
    }

    // Set the remaining triangle indices
    hkLocalArray<int> remainingTriangleIndices(numTriangles);
    hkLocalArray<bool> remainingTriangles(numTriangles);
    {
        remainingTriangleIndices.setSize(numTriangles);
        remainingTriangles.setSize(numTriangles);
        for (int i = 0; i < numTriangles; i++)
        {
            remainingTriangleIndices[i] = i;
            remainingTriangles[i] = true;
        }
    }

    hkArray<int> triangleStack;
    while (remainingTriangleIndices.getSize() > 0)
    {
        // Start of all triangles we have
        int startIndex = triangleIndicesOut.getSize();

		/// Get an unused index
		int unusedTriIndex = remainingTriangleIndices.back();
		remainingTriangleIndices.popBack();

		// Empty the stack
		triangleStack.clear();

		// Its been processed
		remainingTriangles[unusedTriIndex] = false;

        triangleStack.pushBack(unusedTriIndex);
        triangleIndicesOut.pushBack(unusedTriIndex);

        while (triangleStack.getSize() > 0)
        {
            int triIndex = triangleStack.back();
            triangleStack.popBack();

            const hkVector4& triNormal = triangleNormals[triIndex];
            const hkUint16* triIndices = triangleList + (3 * triIndex);

            // Check all of the edges
            int startVertexIndex = 2;
            for (int i = 0; i < 3; i++)
            {
                int pairTriIndex = edgeMap.getWithDefault(calcEdgeId(triIndices[i], triIndices[startVertexIndex]), -1);

                if (pairTriIndex >= 0 && remainingTriangles[pairTriIndex])
                {
                    // Work out if the
                    if (triNormal.dot<3>(triangleNormals[pairTriIndex]).isGreater(cosSmoothAngle))
                    {
                        // Put on stack
                        triangleStack.pushBack(pairTriIndex);
                        // Remove from remaining list
                        {
                            int index = remainingTriangleIndices.indexOf(pairTriIndex);
                            remainingTriangleIndices.removeAt(index);
                            remainingTriangles[pairTriIndex] = false;
                        }
                        // Add to list of indices
                        triangleIndicesOut.pushBack(pairTriIndex);
                    }
                }
                startVertexIndex = i;
            }
        }

        // Store the amount of indices
        numTrianglesIndicesOut.pushBack(triangleIndicesOut.getSize() - startIndex);
    }
}

/* static */void HK_CALL hkNormalCalculator::calculateSmoothedGeometry(const hkArray<hkVector4>& positionsIn, const hkUint16* triangleVertexIndicesIn, int numTrianglesIn, hkReal cosSmoothingAngle, hkArray<hkVector4>& verticesOut, hkArray<hkVector4>& normalsOut, hkArray<hkUint16>& triangleVertexIndicesOut, hkArray<hkUint16>& originalVertexIndicesOut)
{
    verticesOut.clear();
    normalsOut.clear();
    triangleVertexIndicesOut.clear();
    originalVertexIndicesOut.clear();

    // Need to find the groups they are in
    hkArray<int> triangleIndices;
    hkArray<int> numTriangleIndices;

    hkFindUniqueIndicesUtil uniqueIndicesUtil;

    // Calculate the smoothing groups
    calculateSmoothingGroups(positionsIn, triangleVertexIndicesIn, numTrianglesIn, cosSmoothingAngle, triangleIndices, numTriangleIndices);

    int startIndex = 0;
    for (int i = 0; i < numTriangleIndices.getSize(); i++)
    {
        const int vertexBase = verticesOut.getSize();
		const int numGroupTriangles = numTriangleIndices[i];

		hkUint16* groupVertexIndices = triangleVertexIndicesOut.expandBy(numGroupTriangles * 3);
		hkUint16* originalVertexIndices = originalVertexIndicesOut.expandBy(numGroupTriangles * 3);

		{

			const int* triIndices = triangleIndices.begin() + startIndex;
			uniqueIndicesUtil.initialize(positionsIn.getSize());
			{
				hkUint16* dstIndices = groupVertexIndices;
				for (int j = 0; j < numGroupTriangles; j++)
				{
					const hkUint16* indices = triangleVertexIndicesIn + triIndices[j] * 3;
					for (int k = 0; k < 3; k++)
					{
						// Save the original indices
						*originalVertexIndices++ = indices[k];
						// Save off the new indices
						*dstIndices++ = hkUint16(uniqueIndicesUtil.addIndex(indices[k]));
					}
				}
			}
		}

        // We need to calculate normals for this set of triangles
        const hkArray<int>& remapGroupIndices = uniqueIndicesUtil.m_uniqueIndices;
        const int numGroupVertices = remapGroupIndices.getSize();

        hkVector4* dstVertices = verticesOut.expandBy(numGroupVertices);
        hkVector4* dstNormals = normalsOut.expandBy(numGroupVertices);
        // Copy the vertices over
        for (int j = 0; j < numGroupVertices; j++)
        {
            dstVertices[j] = positionsIn[remapGroupIndices[j]];
        }

        // We want to do the smoothing over this group
        hkArray<hkVector4> groupVertices(dstVertices, numGroupVertices, numGroupVertices);
        hkArray<hkVector4> groupNormals(dstNormals, numGroupVertices, numGroupVertices);

        // Set up the normals
        calculateNormals(groupVertices, groupVertexIndices, numGroupTriangles, groupNormals);

        // Fix up the indices (we need to offset the index)
        for (int j = 0; j < numGroupTriangles * 3; j++)
        {
            groupVertexIndices[j] = hkUint16(groupVertexIndices[j] + vertexBase);
        }

        // Next group set
        startIndex += numGroupTriangles;
    }
}

/* static */void HK_CALL hkNormalCalculator::normalize(hkArray<hkVector4>& normals)
{
    const int numNormals = normals.getSize();
    for (int i = 0; i < numNormals; i++)
    {
        hkVector4& normal = normals[i];
        // Hopefully it will see this is calculated twice
        const hkSimdReal len2 = normal.dot<3>(normal);
        // We need to see if its been set before normalizing
        if (len2.isGreater(hkSimdReal_Eps))
        {
            // Normalize
            normal.normalize<3,HK_ACC_12_BIT,HK_SQRT_SET_ZERO>();
        }
        else
        {
			normal = hkVector4::getConstant<HK_QUADREAL_1000>();
        }
    }
}

/* static */void HK_CALL hkNormalCalculator::calculateNormals(const hkArray<hkVector4>& positions, const hkUint16* triangleList, int numTriangles, hkArray<hkVector4>& normals)
{
    // Zero all of the normals
    normals.setSize(positions.getSize());
    hkString::memClear16(normals.begin(), (sizeof(hkVector4)*normals.getSize())>>4);

    // Epsilon to ensure non zero normal
	const hkReal epsilon = hkReal(1e-5f);
	hkVector4 epsilonX;
    epsilonX.set(epsilon, 0, 0);

    for (int j = 0; j < numTriangles; j++, triangleList += 3)
    {
		// Get the indices
        int v0Index = triangleList[0];
        int v1Index = triangleList[1];
        int v2Index = triangleList[2];

        // Get the vertices
        const hkVector4& v0 = positions[v0Index];
        const hkVector4& v1 = positions[v1Index];
        const hkVector4& v2 = positions[v2Index];

        hkVector4 e0; e0.setSub(v1, v0);
        hkVector4 e1; e1.setSub(v2, v0);
        hkVector4 normal; normal.setCross(e0, e1);

        if (normal.lengthSquared<3>().isLess(hkSimdReal_Eps))
		{
			// Its degenerate
			normal = hkVector4::getConstant<HK_QUADREAL_1000>();
		}
		else
		{
            normal.normalize<3,HK_ACC_12_BIT,HK_SQRT_SET_ZERO>();
		}

		// Work out normalized edges
        hkVector4 e0to1;
        hkVector4 e1to2;
        hkVector4 e2to0;

        e0to1.setSub(v1, v0);
		e0to1.add(epsilonX);
        e0to1.normalize<3,HK_ACC_12_BIT,HK_SQRT_SET_ZERO>();

        e1to2.setSub(v2, v1);
		e1to2.add(epsilonX);
        e1to2.normalize<3,HK_ACC_12_BIT,HK_SQRT_SET_ZERO>();

        e2to0.setSub(v0, v2);
		e2to0.add(epsilonX);
        e2to0.normalize<3,HK_ACC_12_BIT,HK_SQRT_SET_ZERO>();

		// Get the normals
		hkVector4& n0 = normals[v0Index];
		hkVector4& n1 = normals[v1Index];
		hkVector4& n2 = normals[v2Index];

		hkSimdReal d;
         // v0
		d = e2to0.dot<3>(e0to1) + hkSimdReal_1;          // Varies between 0-2 depending on the angle
        n0.addMul(d, normal);

        // v1
        d = e0to1.dot<3>(e1to2) + hkSimdReal_1;          // Varies between 0-2 depending on the angle
        n1.addMul(d, normal);

        // v2
        d = e1to2.dot<3>(e2to0) + hkSimdReal_1;          // Varies between 0-2 depending on the angle
        n2.addMul(d, normal);
    }

    // All of the normals need to be normalized at the end
    normalize(normals);
}


/* static */void HK_CALL hkNormalCalculator::calculateTangentSpaces(const hkArray<hkVector4>& positions, const hkArray<hkVector4>& normals, const hkArray<hkVector4>& texCoords, const hkArray<hkUint16>& triangleIndices, hkArray<hkVector4>& tangents, hkArray<hkVector4>& binormals)
{
	const int numTriangles = triangleIndices.getSize() / 3;
	const int numVertices = positions.getSize();

	// Based on Section 9.8.3 of "Mathematics for 3D Game Programming and Computer Graphics", SE, by Eric Lengyel.
	
	hkLocalBuffer<hkVector4> tan1Array(numVertices * 2);
	hkVector4* tan1 = tan1Array.begin();

	hkVector4 *tan2 = tan1 + numVertices;
	hkString::memClear16(tan1, 2*(sizeof(hkVector4)>>4) * numVertices );

	for (int t = 0; t < numTriangles; ++t)
	{
		const int idx1 = triangleIndices[t*3  ];
		const int idx2 = triangleIndices[t*3+1];
		const int idx3 = triangleIndices[t*3+2];

		const hkVector4& v1 = positions[idx1];
		const hkVector4& v2 = positions[idx2];
		const hkVector4& v3 = positions[idx3];

		const hkVector4& w1 = texCoords[idx1];
		const hkVector4& w2 = texCoords[idx2];
		const hkVector4& w3 = texCoords[idx3];

		hkVector4 xyz1; xyz1.setSub(v2,v1);
		hkVector4 xyz2; xyz2.setSub(v3,v1);

		hkVector4 st1; st1.setSub(w2,w1);
		hkVector4 st2; st2.setSub(w3,w1);

		const hkSimdReal scale = st1.getComponent<0>()*st2.getComponent<1>() - st2.getComponent<0>()*st1.getComponent<1>();
		hkSimdReal absScale; absScale.setAbs(scale);
		hkSimdReal r; r.setSelect(absScale.greater(hkSimdReal_Eps),scale.reciprocal(),hkSimdReal_1);

		hkVector4 sdir;
		sdir.setMul(st2.getComponent<1>(), xyz1);
		sdir.subMul(st1.getComponent<1>(), xyz2);
		sdir.mul(r);

		hkVector4 tdir;
		tdir.setMul(st1.getComponent<0>(), xyz2);
		tdir.subMul(st2.getComponent<0>(), xyz1);
		tdir.mul(r);

		tan1[idx1].add(sdir);
		tan1[idx2].add(sdir);
		tan1[idx3].add(sdir);

		tan2[idx1].add(tdir);
		tan2[idx2].add(tdir);
		tan2[idx3].add(tdir);

	}

	hkVector4 nDnt;
	hkVector4 nCt;

	tangents.setSize(numVertices);
	binormals.setSize(numVertices);

	for (int v = 0; v < numVertices; ++v)
	{
		const hkVector4& n = normals[v];
		const hkVector4& t1 = tan1[v];
		const hkVector4& t2 = tan2[v];

		const hkSimdReal nt1 = t1.dot<3>(t1);
		hkBool32 okT1 = nt1.isGreater(hkSimdReal::fromFloat(0.1f));

		const hkVector4* t = ( okT1? &t1 : &t2 );
		const hkVector4* bt = ( okT1? &t2 : &t1 );

		hkVector4* tangent = &(tangents[v]);
		hkVector4* bitangent = &(binormals[v]);
		if (!okT1)
		{
			tangent = &(binormals[v]);
			bitangent = &(tangents[v]);
		}

		// Gram-Schmidt orthogonalize
		// tangent[a] = (t - n * (n * t)).Normalize();
		hkSimdReal nDt = n.dot<3>(*t);
		nDnt.setMul(nDt,n);
		tangent->setSub(*t, nDnt);
		tangent->normalizeIfNotZero<3>();

		// Calculate handedness
		// tangent[a].w = (n % t * tan2[a] < 0.0F) ? -1.0F : 1.0F;
		nCt.setCross(n, *t);
		const hkSimdReal tw = nCt.dot<3>(*bt);

		// Calculate the bitangent
		if (tw.isLessZero())
		{
			bitangent->setCross(*tangent, n);
		}
		else
		{
			bitangent->setCross(n, *tangent);
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
