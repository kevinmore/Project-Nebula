/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Common/Base/Container/PointerMap/hkPointerMap.h>
#include <Common/Base/Types/Geometry/hkGeometry.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>

#include <Common/Internal/GeometryProcessing/ConvexHull/hkgpConvexHull.h>
#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/hkpShapeCollection.h>
#include <Physics2012/Collide/Util/ShapeCutter/hkpShapeCutterUtil.h>

#include <Common/Base/Types/Color/hkColor.h>
#include <Common/Visualize/hkDebugDisplay.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>

#include <Common/Base/Algorithm/Sort/hkSort.h>

#include <Common/Internal/ConvexHull/hkGeometryUtility.h>

#include <Common/Base/Memory/Allocator/FreeList/hkFreeList.h>

#include <Common/Internal/ConvexHull/Deprecated/hkGeomHull.h>
#include <Common/Internal/ConvexHull/Deprecated/hkGeomConvexHullBuilder.h>

// This
#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesConnectivityUtil.h>
HK_DISABLE_OPTIMIZATION_VS2008_X64
hkpConvexVerticesConnectivity* _findConnectivity( const hkArray<hkVector4>& vertices, const hkArray<hkVector4>& facePlanes, const hkArray<hkVector4>& usedVertices, hkGeomHull& hull )
{
	hkInplaceArray<int,32> originalVertices;
	originalVertices.setSize(usedVertices.getSize());

	if ( vertices.begin() == usedVertices.begin() && vertices.getSize() == usedVertices.getSize() )
	{
		for (int i = 0; i < usedVertices.getSize(); i++ )
		{
			originalVertices[i] = i;
		}
	}
	else
	{
		for (int i = 0; i < usedVertices.getSize(); i++ )
		{
			const hkVector4& newVert = usedVertices[i];

			int closest = -1;
			hkSimdReal closestDistance = hkSimdReal::fromFloat(1e10f);

			for (int j = 0; j < vertices.getSize(); j++ )
			{
				const hkVector4& origVert = vertices[j];

				if (newVert.allExactlyEqual<3>(origVert))
				{
					closest = j;
					closestDistance.setZero();
					break;
				}

				hkVector4 diff;
				diff.setSub(newVert,origVert);

				const hkSimdReal dist = diff.length<3>();

				if (dist < closestDistance)
				{
					closestDistance = dist;
					closest = j;
				}
			}

			// If its too far away something is probably wrong
			HK_ASSERT(0x2098b211,closestDistance.getReal() < 0.1f );

			// Set the closest
			originalVertices[i] = closest;
		}

	}
	// We need to make up the edges, for the faces. We need to find the face that matches with these edges.

	hkArray<hkBool> edgeUsed;
	{
		edgeUsed.setSize(hull.m_edges.getSize());
		for (int i = 0; i < edgeUsed.getSize(); i++)
		{
			edgeUsed[i] = false;
		}
	}

	hkInplaceArray<hkpConvexVerticesConnectivityUtil::FaceEdge*,64> faces;
	faces.setSize(facePlanes.getSize(),HK_NULL);

	hkInplaceArray<int,8> faceIndices;

	hkFreeList edgeFreeList(sizeof(hkpConvexVerticesConnectivityUtil::FaceEdge),sizeof(void*),2048);
	for (int i = 0; i < hull.m_edges.getSize(); i++)
	{
		if (edgeUsed[i])
		{
			// If this edge has been visited we are done
			continue;
		}

		faceIndices.clear();

		// Get the edge
		{
			hkGeomEdge* startEdge = &hull.m_edges[i];
			hkGeomEdge* edge = startEdge;
			do
			{
				// Save out the index
				faceIndices.pushBack(edge->m_vertex);
				// Mark next
				edgeUsed[edge->m_next] = true;
				// Get the next edge
				edge = &hull.m_edges[edge->m_next];
			}
			while (edge != startEdge);
		}

		// Find the plane the indices are all in
		const int numIndices = faceIndices.getSize();

		int faceIndex = 0;
		if (numIndices > 0)
		{
			// Look up the face index - will be the face

			hkSimdReal smallestDistance = hkSimdReal::fromFloat(1e10f);
			int bestFaceIndex = -1;
			for (int j = 0; j < facePlanes.getSize(); j++)
			{
				const hkVector4& facePlane = facePlanes[j];

				hkSimdReal minDistance; minDistance.setZero();
				for (int k = 0; k < numIndices; k++)
				{
					const hkSimdReal dist = facePlane.dot4xyz1(usedVertices[faceIndices[k]]);
					hkSimdReal absDist; absDist.setAbs(dist);			// Make +ve

					if (absDist > minDistance)
					{
						minDistance = absDist;
					}
				}

				if (minDistance < smallestDistance)
				{
					bestFaceIndex = j;
					smallestDistance = minDistance;
				}
			}

			HK_ASSERT(0xd8279a22,bestFaceIndex >= 0);
			//if (smallestDistance > 0.1f)
			{
				// Looks like its not possible to get connectivity 
				//HK_WARN(0x44342, "Unable to find hkpConvexVerticesShape connectivity - vertices don't lie in plane");
				//return false;
			}

			//HK_ASSERT(0xd8279a21, smallestDistance < 1e-1f);

			faceIndex = bestFaceIndex;
		}
		// Add the edges
		int start = faceIndices.back();
		for (int j = 0; j < numIndices; j++)
		{
			hkpConvexVerticesConnectivityUtil::FaceEdge* edge = (hkpConvexVerticesConnectivityUtil::FaceEdge*)edgeFreeList.alloc();

			int end = faceIndices[j];

			edge->m_startIndex = end;
			edge->m_endIndex = start;

			// Add the edge to the list
			edge->m_next = faces[faceIndex];
			faces[faceIndex] = edge;

			// Next ones start is this ones end
			start = end;
		}
	}

	hkpConvexVerticesConnectivity* out = new hkpConvexVerticesConnectivity();

	hkArray<int> next;
	next.setSize(vertices.getSize());

	for (int i = 0; i < faces.getSize(); i++)
	{
		{
			hkpConvexVerticesConnectivityUtil::FaceEdge* edge = faces[i];

			if (edge == HK_NULL)
			{
				// Output degenerate face info
				out->m_numVerticesPerFace.pushBack(0);
				continue;
			}

			/// Remove all of the pairs
			hkpConvexVerticesConnectivityUtil::FaceEdge** prev = &faces[i];
			while (edge)
			{
				// Find if there is a pair
				hkpConvexVerticesConnectivityUtil::FaceEdge* pair = edge->m_next;
				hkpConvexVerticesConnectivityUtil::FaceEdge** pairPrev = &edge->m_next;

				// Search for the reverse edge
				while (pair && (pair->m_startIndex != edge->m_endIndex || pair->m_endIndex != edge->m_startIndex))
				{
					pairPrev = &pair->m_next;
					pair = pair->m_next;
				}

				if (pair)
				{
					// We found a match -> we need to remove both edges
					if (edge->m_next == pair)
					{
						// If they are contiguous, just rip them both out
						*prev = pair->m_next;
						edge = pair->m_next;
					}
					else
					{
						// Move along
						edge = edge->m_next;
						// Remove the edge
						*prev = edge;
						// Remove the pair
						*pairPrev = pair->m_next;
					}
				}
				else
				{
					// Go to the next edge
					prev = &edge->m_next;
					edge = edge->m_next;
				}
			}
		}

		{
			int startVertexIndex = out->m_vertexIndices.getSize();

			// We need to put the edges in order to output
			hkpConvexVerticesConnectivityUtil::FaceEdge* edge = faces[i];
			if (edge)
			{
				while (edge)
				{
					next[edge->m_startIndex] = edge->m_endIndex;
					edge = edge->m_next;
				}

				int startIndex = faces[i]->m_startIndex;
				int index = startIndex;

				do
				{
					// Output this index
					out->m_vertexIndices.pushBack(hkUint16(originalVertices[index]));
					// Next
					int newIndex = next[index];
					next[index] = startIndex;	// make sure we do not run into an endless loop
					index = newIndex;
				}
				while (index != startIndex);
			}

			/// Save the amount of indices
			out->m_numVerticesPerFace.pushBack(hkUint8(out->m_vertexIndices.getSize() - startVertexIndex));
		}
	}

	return out;
}

HK_RESTORE_OPTIMIZATION_VS2008_X64

hkGeometry* hkpConvexVerticesConnectivityUtil::createGeometry(const hkpConvexVerticesShape* shape,const hkpConvexVerticesConnectivity* con)
{
    HK_ASSERT(0x3002423,con && shape);

    hkGeometry* geom = new hkGeometry;

    shape->getOriginalVertices(geom->m_vertices);

    /// Use the connectivity to triangulate the surface
    const int numFaces = con->getNumFaces();
    int faceStart = 0;
    for (int i = 0; i < numFaces; i++)
    {
        const int numFaceIndices = con->m_numVerticesPerFace[i];

        int startIndex = con->m_vertexIndices[faceStart];
        for (int j = 0; j < numFaceIndices-2; j++)
        {
            hkGeometry::Triangle tri;

            // Fan
			tri.set(startIndex, con->m_vertexIndices[faceStart+j+1], con->m_vertexIndices[faceStart+j+2]);
            
            // Add the triangle
            geom->m_triangles.pushBack(tri);
        }

        faceStart += numFaceIndices;
    }

    return geom;
}

hkGeometry* hkpConvexVerticesConnectivityUtil::createGeometry(const hkpConvexVerticesShape* shape,const hkpConvexVerticesConnectivity* con, const hkTransform& transform)
{
	HK_ASSERT(0x3246323,con && shape);

	hkGeometry* geom = new hkGeometry;

	shape->getOriginalVertices(geom->m_vertices);

	// Iterate through the vertices and transform them
	hkArray<hkVector4>& vertices = geom->m_vertices;
	for (int j = 0; j < vertices.getSize(); ++j)
	{
		vertices[j]._setTransformedPos(transform, vertices[j]);
	}

	/// Use the connectivity to triangulate the surface
	const int numFaces = con->getNumFaces();
	int faceStart = 0;
	for (int i = 0; i < numFaces; i++)
	{
		const int numFaceIndices = con->m_numVerticesPerFace[i];

		int startIndex = con->m_vertexIndices[faceStart];
		for (int j = 0; j < numFaceIndices-2; j++)
		{
			hkGeometry::Triangle tri;

			// Fan
			tri.set(startIndex, con->m_vertexIndices[faceStart+j+1], con->m_vertexIndices[faceStart+j+2]);
			
			// Add the triangle
			geom->m_triangles.pushBack(tri);
		}

		faceStart += numFaceIndices;
	}

	return geom;
}



const hkpConvexVerticesShape* HK_CALL hkpConvexVerticesConnectivityUtil::cut(const hkpConvexVerticesShape* shape, const hkVector4& plane, hkReal convexRadius, hkReal minVolume)
{
	// Correct plane by convex radius
	hkVector4 planeShrunkenByConvexRadius = plane;
	planeShrunkenByConvexRadius( 3 ) += convexRadius;

	ensureConnectivity( shape );

	const hkpConvexVerticesConnectivity* conn = shape->getConnectivity();
	HK_ASSERT2(0x3403a23, conn, "Convex vertices shape without connectivity is not supported.\n");

    hkLocalArray<hkVector4> inVertices(shape->getNumCollisionSpheres());
    shape->getOriginalVertices(inVertices);
	const int numInVertices = inVertices.getSize();

	hkArray<VertexInfo>::Temp vertexInfo; vertexInfo.setSize(numInVertices);

	const hkArray<hkVector4>& inPlanes = shape->getPlaneEquations();

	hkArray<hkVector4> newVertices; newVertices.reserve(numInVertices);

    for (int i = 0; i < numInVertices; i++)
    {
        VertexInfo& info = vertexInfo[i];

        const hkVector4& vertexPos = inVertices[i];
        info.m_side = planeShrunkenByConvexRadius.dot4xyz1(vertexPos).getReal();

        if (info.m_side > 0.0f)
        {
            info.m_outside = 1;
        }
        else
        {
            info.m_outside = 0;
			info.m_newIndex = newVertices.getSize();
			newVertices.pushBackUnchecked(vertexPos);
        }
    }

    if (newVertices.isEmpty())
    {
        // Nothing was produced
        return HK_NULL;
    }

    if (newVertices.getSize() == inVertices.getSize())
    {
        // Lets not copy
        shape->addReference();
        return shape;
    }

	hkpConvexVerticesConnectivity* connOut = new hkpConvexVerticesConnectivity;

    // Planes for output
    hkInplaceArray<hkVector4,32> outPlanes;

    // Key is 16/16 bit concat of the indices. To make edges unique the order always goes from smaller to
    // larger index
    hkPointerMap<hkUint32,hkUint32> vertexMap;
	vertexMap.reserve( numInVertices * 2 );

    hkInplaceArray<Edge,32> capEdges;
    {
        const int numFaces = conn->getNumFaces();
        int faceStart = 0;
		int planeIndex = 0;
        for (int fi = 0; fi < numFaces; planeIndex++, fi++)
        {
            const int numFaceIndices = conn->m_numVerticesPerFace[fi];
			if ( numFaceIndices < 3 )		// if the face has no area, than there is no plane equation stored
			{
				planeIndex--;
				continue;
			}

            int startIndices = connOut->m_vertexIndices.getSize();

            // Look for a vertex that is inside-> that way we can ensure if we hit a cut then we will have a start
            // vertex
            int numInside = 0;
            int cur = -1;
            for (int j = 0; j < numFaceIndices; j++)
            {
                int index = conn->m_vertexIndices[faceStart + j];
                const VertexInfo& info = vertexInfo[index];
                if (info.m_outside == 0)
                {
                    // Look if its inside
                    cur = j;
					numInside ++;
                }
            }

            if (numInside == numFaceIndices)
            {
                // They are all inside -> copy the face
                for (int j = 0; j < numFaceIndices; j++)
                {
					int index = conn->m_vertexIndices[faceStart + j];
                    const VertexInfo& info = vertexInfo[index];
                    connOut->m_vertexIndices.pushBack(hkUint16(info.m_newIndex));
                }
                connOut->m_numVerticesPerFace.pushBack(hkUint8(numFaceIndices));
				// planes
				outPlanes.pushBack(inPlanes[planeIndex]);

				// Update the face start
				faceStart += numFaceIndices;
                continue;
            }

            if (cur < 0)
            {
                // They are all outside

				// Update the face start
				faceStart += numFaceIndices;
                continue;
            }

            int edgeStart = -1;
            int startIndex =  conn->m_vertexIndices[faceStart + cur];
            for (int j = 0; j < numFaceIndices; j++)
            {
                // Increment cur, and make it wrap around
                cur = (cur == numFaceIndices-1)?0:cur+1;

                // End index
                int endIndex = conn->m_vertexIndices[faceStart + cur];

				// get the vertex info for these indices
                const VertexInfo& startInfo = vertexInfo[startIndex];
                const VertexInfo& endInfo = vertexInfo[endIndex];

				// If they are different we know we have a cut
                if ((startInfo.m_outside ^ endInfo.m_outside) == 0)
                {
                    if (startInfo.m_outside == 0)
                    {
                        // Both inside
                        connOut->m_vertexIndices.pushBack(hkUint16(startInfo.m_newIndex));
                    }
                    else
                    {
                        // Both outside - so ignore
                    }
                }
                else
                {
                    // Need to find the new index
                    hkUint32 key = (startIndex<endIndex) ? ((startIndex<<16)|(endIndex))  :  ((endIndex<<16)|(startIndex));
                    hkUint32 vertexIndex;
                    // Do the lookup

                    hkPointerMap<hkUint32,hkUint32>::Iterator iter = vertexMap.findKey(key);
                    if (vertexMap.isValid(iter))
                    {
                        vertexIndex = vertexMap.getValue(iter);
                    }
                    else
                    {
                        // Work out the new vertex
                        hkVector4 newVertex;
                        newVertex.setSub(inVertices[endIndex],inVertices[startIndex]);
                        hkReal interp =  startInfo.m_side/( startInfo.m_side - endInfo.m_side);
						newVertex.mul(hkSimdReal::fromFloat(interp));
                        newVertex.add(inVertices[startIndex]);

                        // Add it
                        vertexIndex = newVertices.getSize();
                        newVertices.pushBack(newVertex);
                        // Add the key
                        vertexMap.insert(key,vertexIndex);
                    }

                    if (startInfo.m_outside == 0)
                    {
                        /// inside->outside => start vertex for cap face

                        // Need to add this vertex
                        connOut->m_vertexIndices.pushBack(hkUint16(startInfo.m_newIndex));
                        // ... and the interp vertex
                        connOut->m_vertexIndices.pushBack(hkUint16(vertexIndex));

                        HK_ASSERT(0xbd838dd3,edgeStart <0);
                        edgeStart = vertexIndex;
                    }
                    else
                    {
                        /// outside->inside => end vertex for cap face

                        connOut->m_vertexIndices.pushBack(hkUint16(vertexIndex));

                        HK_ASSERT(0x223324f,edgeStart >=0);
                        // Add an edge

                        Edge edge;
                        edge.m_end = edgeStart;
                        edge.m_start = vertexIndex;

                        capEdges.pushBack(edge);
                        // mark as invalid
                        edgeStart = -1;
                    }
                }
                // next
                startIndex = endIndex;
            }

			// Need to see if anything was output
			const int finalIndices = connOut->m_vertexIndices.getSize();
			int numOutFaceIndices = finalIndices - startIndices;

			if (numOutFaceIndices)
			{
				if (numOutFaceIndices <= 2 )
				{
					// Remove the indices
					connOut->m_vertexIndices.setSize(startIndices);
				}
				else
				{
					// add the face
					outPlanes.pushBack(inPlanes[planeIndex]);
					// Set the amount of indices
					connOut->m_numVerticesPerFace.pushBack(hkUint8(numOutFaceIndices));
				}
			}

			///
			faceStart += numFaceIndices;
        }
    }

    // If the shape is not strictly convex, or there are numerical inaccuracies it is possible
    // that multiple capping faces are produced
    // We want to find each connected ring and output it as a face
    // Doing this will mean there will likely be connected - but degenerate volumes
    // additionally the vertex ordering may not be correct wrt to the direction of the normal of the
    // surface.
    //
    // Thus after you a cut you could find
    // o More than one connected contained surfaces - although only one of them can be thought of as correct
    // o The plane normal may be facing in the opposite direction to the normal defined by the direction of the
    //   edges making up the face
    // o A face may not be convex - it can only be slightly non-convex say due to numerical inaccuracies or that
    //   the input data is non convex

	bool capEdgesAreOK = false;

	int capEdgesSize = capEdges.getSize();
    while (capEdges.getSize() >= 3)
    {
        int finalIndices = connOut->m_vertexIndices.getSize();

        Edge* edge = &capEdges[0];
        int startIndex = edge->m_start;
        int nextIndex = edge->m_end;

        capEdges.removeAt(0);

        connOut->m_vertexIndices.pushBack(hkUint16(startIndex));

        while (startIndex != nextIndex)
        {
            // Look for an edge which start index is the next index
            for (int i = 0; i < capEdges.getSize(); i++)
            {
                const Edge& capEdge = capEdges[i];
                if (capEdge.m_start == nextIndex)
                {
                    // output next
                    connOut->m_vertexIndices.pushBack(hkUint16(nextIndex));
                    nextIndex = capEdge.m_end;

                    // Remove edge
                    capEdges.removeAt(i);
                    goto TRY_NEXT_EDGE;
                }
            }
			// edge not found, drop last edge
			if ( connOut->m_vertexIndices.getSize() == 0)
			{
				goto START_NEW_EDGE;
			}
			nextIndex = connOut->m_vertexIndices.back();
			connOut->m_vertexIndices.popBack();
TRY_NEXT_EDGE:;
        }
        // Set the face
		{
			int numCapVertices = connOut->m_vertexIndices.getSize() - finalIndices;
			if (numCapVertices == capEdgesSize )
			{
				capEdgesAreOK = true;
			}
			connOut->m_numVerticesPerFace.pushBack(hkUint8(numCapVertices));
		}

        // Add the plane
        outPlanes.pushBack(plane);
START_NEW_EDGE:;
    }

	hkpConvexVerticesShape* shapeOut;
	if ( capEdgesAreOK )
	{
		// If the volume is too small we'll say nothing was produced
		hkReal volume = calculateVolume(newVertices,connOut);
		if (volume < minVolume)
		{
			connOut->removeReference();
			return HK_NULL;
		}
		connOut->m_numVerticesPerFace.optimizeCapacity( 0,true);

		// Create the shape
		shapeOut = new hkpConvexVerticesShape(hkStridedVertices(newVertices.begin(), newVertices.getSize()), outPlanes, shape->getRadius() );
	    shapeOut->setConnectivity(connOut, false);
		HK_ASSERT(0x5c85538b, shapeOut->getConnectivity()->getNumFaces() <= shapeOut->getPlaneEquations().getSize() );
	}
	else
	{
		// rebuild convex vertices shape from scratch 
		hkpConvexVerticesShape::BuildConfig config;
		config.m_createConnectivity = true;
		config.m_shrinkByConvexRadius = false;
		config.m_maxRelativeShrink = 0.0f;
		config.m_maxShrinkingVerticesDisplacement = 0.0f;

		shapeOut = new hkpConvexVerticesShape( hkStridedVertices(newVertices.begin(), newVertices.getSize()), config );
		if ( shapeOut->m_numVertices < 3 )
		{	// if the shape degenerates to a single line, return nothing
			shapeOut->removeReference();
			connOut->removeReference();
			return HK_NULL;
		}

		// expand plane equations
		shapeOut->setRadiusUnchecked( shape->getRadius() );
		{
			for (int pe = 0; pe < shapeOut->m_planeEquations.getSize(); pe++ )
			{
				shapeOut->m_planeEquations[pe](3) -= shape->getRadius();
			}
		}
	}

	// cleanup temp stuff
	connOut->removeReference();

	// Copy original shape user data
	if ( shape && shapeOut )
	{
		shapeOut->m_userData = shape->m_userData;
	}

    return shapeOut;
}

hkResult hkpConvexVerticesConnectivityUtil::ensureConnectivity(const hkpConvexVerticesShape* shape)
{
    if (shape->getConnectivity())
    {
        return HK_SUCCESS;
    }

	hkpConvexVerticesConnectivity* conn = findConnectivity(shape);

	if (!conn)
	{
		return HK_FAILURE;
	}

	const_cast< hkpConvexVerticesShape* >( shape )->setConnectivity(conn);
	conn->removeReference();

	return HK_SUCCESS;

}


hkResult hkpConvexVerticesConnectivityUtil::ensureConnectivityAll(const hkpShape* shapeIn)
{
    switch (shapeIn->getType())
    {
        case hkcdShapeType::COLLECTION:
		case hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_COLLECTION:
        {
            const hkpShapeCollection* collection = static_cast<const hkpShapeCollection*>(shapeIn);

			hkpShapeKey key = collection->getFirstKey();
            while (key != HK_INVALID_SHAPE_KEY)
            {
                // Get the child
                hkpShapeBuffer shapeBuffer;
                const hkpShape* child = collection->getChildShape(key, shapeBuffer);

                // Recurse
                hkResult res = ensureConnectivityAll(child);
				if ( res == HK_FAILURE )
				{
					return res;
				}
                // Next
                key = collection->getNextKey(key);
            }
            break;
        }
        case hkcdShapeType::CONVEX_VERTICES:
        {
            const hkpConvexVerticesShape* shape = static_cast<const hkpConvexVerticesShape*>(shapeIn);
            if (shape->getConnectivity() == HK_NULL)
            {
                // Set up connectivity
                hkpConvexVerticesConnectivity* conn = findConnectivity(shape);

				if ( !conn )
				{
					return HK_FAILURE;
				}
                const_cast< hkpConvexVerticesShape* >( shape )->setConnectivity(conn);
                conn->removeReference();
            }
            break;
        }
        default:
        {
            break;
        }
    }
	return HK_SUCCESS;
}

hkpConvexVerticesConnectivity* hkpConvexVerticesConnectivityUtil::findConnectivity(const hkpConvexVerticesShape* shape)
{
	hkArray<hkVector4> vertices;
	shape->getOriginalVertices(vertices);
	
    const hkArray<hkVector4>& facePlanes = shape->getPlaneEquations();

    hkArray<hkVector4> usedVertices;
    hkGeomHull hull;

    hkGeomConvexHullBuilder::generateConvexHull( vertices.begin(), vertices.getSize(), hull, usedVertices, HK_GEOM_CONVEXHULL_MODE_FAST); 

	return _findConnectivity( vertices, facePlanes, usedVertices, hull );
}


#if 0
void hkpConvexVerticesConnectivityUtil::drawArrow(const hkVector4& start,const hkVector4& end,int lineCol,int headCol)
{
    hkVector4 dir;
    dir.setSub4(end,start);

            // Check that we have a valid direction
	if (dir.lengthSquared3() < HK_REAL_EPSILON)
	{
		return;
	}

    hkVector4 normal;
    normal = dir;
    normal.normalize3();
    const hkReal twoPi = 3.1415926f*2.0f;
    const int numSegments = 20;
    hkQuaternion quat(normal,twoPi/numSegments);

    HK_DISPLAY_LINE(start,end,lineCol);

    // get an orthogonal vector
    hkVector4 ort; hkVector4Util::calculatePerpendicularVector( normal, ort );
    for (int i=0;i<numSegments;i++)
    {
        ort.setRotatedDir(quat,ort);

        // Work out the start point
        hkVector4 nextStart;
        nextStart.setMul4(0.05f,ort);
        nextStart.addMul4(-0.1f,normal);
        nextStart.add4(end);

        HK_DISPLAY_LINE(nextStart,end,headCol);
    }
}


void hkpConvexVerticesConnectivityUtil::drawAsLines(const hkTransform& trans,const hkpConvexVerticesShape* shape,const hkpConvexVerticesConnectivity* conn)
{
    hkArray<hkVector4> vertices;
    shape->getOriginalVertices(vertices);

    for (int i = 0; i < vertices.getSize(); i++)
    {
        hkVector4 pos;
        pos.setTransformedPos(trans,vertices[i]);
        vertices[i] = pos;
    }

    // Lets go through the vertices
    for (int i = 0; i < vertices.getSize(); i++)
    {
        HK_DISPLAY_STAR(vertices[i], 0.05f, hkColor::WHITE);
    }

    const hkArray<hkVector4>& facePlanes = shape->getPlaneEquations();

    const int numFaces = conn->getNumFaces();
    int faceStart = 0;
    for (int i = 0; i < numFaces; i++)
    {
        const int numFaceIndices = conn->m_numVerticesPerFace[i];

        if (numFaceIndices <= 0)
        {
            continue;
        }

        // Fan out the shape
        const hkUint16* indices = &conn->m_vertexIndices[faceStart];
        int prev = indices[numFaceIndices - 1];

        hkVector4 pushOut;
        pushOut.setMul4(0.1f,facePlanes[i]);

        for (int j = 0; j < numFaceIndices; j++)
        {
			int next = indices[j];

            hkVector4 v0; v0.setAdd4(vertices[prev],pushOut);
            hkVector4 v1; v1.setAdd4(vertices[next],pushOut);

            // Draw the line
            drawArrow(v0,v1,hkColor::RED,hkColor::RED);

            // set prev
            prev = next;
        }

        ///
        faceStart += numFaceIndices;
    }
}
#endif

hkReal hkpConvexVerticesConnectivityUtil::calculateVolume(const hkpConvexVerticesShape* shape,const hkpConvexVerticesConnectivity* conn)
{
    hkArray<hkVector4> vertices;
    shape->getOriginalVertices(vertices);

    return calculateVolume(vertices,conn);
}

hkReal hkpConvexVerticesConnectivityUtil::calculateVolume(const hkArrayBase<hkVector4>& vertices,const hkpConvexVerticesConnectivity* conn)
{
    hkSimdReal volume; volume.setZero();

    const int numFaces = conn->getNumFaces();
    int faceStart = 0;
    for (int i = 0; i < numFaces; i++)
    {
        const int numFaceIndices = conn->m_numVerticesPerFace[i];

		if (numFaceIndices <= 0)
		{
			continue;
		}

        // Fan out the shape
        const hkUint16* indices = &conn->m_vertexIndices[faceStart];

        int i0 = indices[0];
		const hkVector4& v0 = vertices[i0];

        for (int j = 1; j < numFaceIndices-1; j++)
        {
            int i1 = indices[j];
            int i2 = indices[j + 1];

            hkVector4 d0,d1;
            const hkVector4& v1 = vertices[i1];
            const hkVector4& v2 = vertices[i2];

            d0.setSub(v1,v0);
            d1.setSub(v2,v0);

            const hkSimdReal area = d0.getComponent<0>() * d1.getComponent<1>() - d0.getComponent<1>() * d1.getComponent<0>();
			volume.addMul(area, v0.getComponent<2>() + v1.getComponent<2>() + v2.getComponent<2>() );
        }

        ///
        faceStart += numFaceIndices;
    }

	hkSimdReal actualVolume; actualVolume.setDiv(volume, hkSimdReal_6);
    return actualVolume.getReal();
}

#if 0
// This is the old implementation - it may still be useful

hkBool hkpConvexVerticesConnectivityUtil::findConnectivity(const hkpConvexVerticesShape* shape,hkpConvexVerticesConnectivity& out) const
{
    hkArray<hkVector4> vertices;
    shape->getOriginalVertices(vertices);
    const hkArray<hkVector4>& planes = shape->getPlaneEquations();

    hkBool res = setPlanesVertices(planes,vertices,out);

	{
		hkReal vol = calculateVolume(shape,&out);

		hkMassProperties massProps;
		hkpInertiaTensorComputer::computeShapeVolumeMassProperties(shape, 1.0f, massProps);

		HK_ASSERT(0x7bba5465,hkMath::fabs(massProps.m_volume - vol)<0.1f);
	}

	return res;
}

hkBool hkpConvexVerticesConnectivityUtil::setPlanesVertices(const hkArray<hkVector4>& planes,const hkArray<hkVector4>& vertices,hkpConvexVerticesConnectivity& out) const
{
    out.clear();

    // Go through all of the faces
    hkArray<int> vertexIndices;
    hkArray<int> faceIndices;
    for (int i = 0; i < planes.getSize(); i++)
    {
        const hkVector4& plane = planes[i];
        vertexIndices.clear();
        faceIndices.clear();

        int numVertices = findVerticesOnPlane(plane,vertices,vertexIndices);
		if (numVertices<3)
		{
            HK_WARN_ALWAYS(0xabba3ecf,"Couldn't find sufficient vertices to make a plane");
            return false;
		}

        HK_ASSERT2(0xffeeddcc,numVertices>=3,"There must be at least 3 vertices to define a plane");

        int curIndex = 0;
        for (int j = 0; j < vertexIndices.getSize(); j++)
        {
            // Need to find the vertex which all these vertices will be
            int nextIndex = findConnected(curIndex,vertices,vertexIndices,plane);
            HK_ASSERT(0x24234234,nextIndex != curIndex);

            if (nextIndex < 0)
            {
                HK_ASSERT2(0x34524122,false,"Couldn't connect up");
                return false;
            }

            // We've found an index
            faceIndices.pushBack(vertexIndices[curIndex]);
            curIndex = nextIndex;
        }

        // add the face
        out.addFace(&faceIndices[0],faceIndices.getSize());
    }
    return true;
}


// Okay -> I'm going to look for the edge which all of the other vertices are behind

int hkpConvexVerticesConnectivityUtil::findConnected(int index,const hkArray<hkVector4>& vertices,const hkArray<int>& indices,const hkVector4& normal) const
{
    int startIndex = indices[index];
    const hkVector4& start = vertices[startIndex];

    for (int i = 0; i < indices.getSize(); i++)
    {
        if (i == index)
        {
            // A line to itself would be just silly
            continue;
        }

        int endIndex = indices[i];
        const hkVector4& end = vertices[endIndex];

        // Calculate the plane
        hkVector4 edge;
        edge.setSub4(end,start);

        if (edge.lengthSquared3() < m_vertexThreshold * m_vertexThreshold)
		{
			HK_WARN_ALWAYS(0xabba102e,"Vertices too close together to work out face construction");
			return -1;
		}

		hkReal edgeLen = edge.normalizeWithLength3();

        hkVector4 testNormal;
        testNormal.setCross(edge,normal);
        testNormal.normalize3();

        hkVector4 testPlane = testNormal;
		testPlane(3) = -testNormal.dot3(start);

		// We could check that both points are on the plane

        HK_ASSERT2(0x3454abde,_onPlane(testPlane,start) && _onPlane(testPlane,end),"They both should be on the plane!");

        // Okay found the plane. If all of the other vertices are behind this plane we have done it

        hkBool foundVertex = false;
        for (int j = 0; j < indices.getSize(); j++)
        {
            if (j == i || j == index)
            {
                continue;
            }

            const hkVector4& cur = vertices[indices[j]];
            hkReal dist = testPlane.dot4xyz1(cur);

            if (dist > 0.0f)
            {
                if (dist < m_onPlaneThreshold)
				{
					/// Hmm could be described as on the plane.
					/// Is it in the direction of the edge?
					hkVector4 test;
                    test.setSub4(cur,start);

					hkReal along = edge.dot3(test);
					// Well if its 'on' the edge and further along we could still be on
					if (along > edgeLen)
					{
						continue;
					}
				}

                // Found a vertex on the other side of the plane -> we failed
                foundVertex = true;
                break;
            }
        }

        if (!foundVertex)
        {
            // If no vertex could be found on the other side then we are done
            return i;
        }
    }

	//HK_ASSERT2(0x123409ab,false,"I couldn't find a vertex.. something is wrong..");
	// If couldn't find one then somethings gone wrong
	return -1;
}

int hkpConvexVerticesConnectivityUtil::findNumVerticesOnPlane(const hkVector4& plane,const hkArray<hkVector4>& vertices)
{
    int numVertices =0;
    for (int i = 0; i < vertices.getSize(); i++)
    {
        const hkVector4& vert = vertices[i];
        if (_onPlane(plane,vert))
        {
            numVertices++;
        }
    }
    return numVertices;
}

int hkpConvexVerticesConnectivityUtil::findVerticesOnPlane(const hkVector4& plane,const hkArray<hkVector4>& vertices,hkArray<int>& indicesOut) const
{
    int numVertices =0;
    for (int i = 0; i < vertices.getSize(); i++)
    {
        const hkVector4& vert = vertices[i];
        if (_onPlane(plane,vert))
        {
            indicesOut.pushBack(i);
            numVertices++;
        }
    }
    return numVertices;
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
