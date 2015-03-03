/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>

#include <Physics2012/Collide/Shape/Convex/Sphere/hkpSphereShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/List/hkpListShape.h>
#include <Physics2012/Collide/Shape/Misc/Transform/hkpTransformShape.h>
#include <Physics2012/Collide/Filter/Null/hkpNullCollisionFilter.h>

#include <Physics2012/Collide/Query/Collector/PointCollector/hkpAllCdPointCollector.h>
#include <Physics2012/Collide/Query/Collector/PointCollector/hkpClosestCdPointCollector.h>

#include <Common/GeometryUtilities/Mesh/Utils/MeshSectionLockSet/hkMeshSectionLockSet.h>
#include <Common/GeometryUtilities/Mesh/Utils/VertexBufferUtil/hkMeshVertexBufferUtil.h>
#include <Common/GeometryUtilities/Mesh/hkMeshShape.h>


// This
#include <Physics2012/Collide/Util/ShapeSkinning/hkpShapeSkinningUtil.h>


/* static */int HK_CALL hkpShapeSkinningUtil::findClosestShape(hkpCollisionDispatcher* collisionDispatcher, const hkArray<hkTransform>& shapeTransforms, const hkArray<const hkpShape*>& shapes, hkReal maxDistance, const hkVector4& point)
{
    hkSimdReal minDist = hkSimdReal_Max;
    int minShapeIndex = -1;

    const int numShapes = shapes.getSize();
    HK_ASSERT(0x2963cb69, shapeTransforms.getSize() == numShapes);

	// Create the sphere shape we are going to use
    hkpSphereShape sphereShape(maxDistance);

	// Set up the collision input
	hkpNullCollisionFilter shapeFilter;
	hkpCollisionInput collisionInput;
	collisionInput.m_filter = &shapeFilter;
    collisionInput.m_dispatcher = collisionDispatcher;
	collisionInput.setTolerance( 0.0f);

    hkTransform sphereTransform; sphereTransform.setIdentity();
    sphereTransform.getTranslation() = point;

    hkpCdBody sphereBody(&sphereShape, &sphereTransform);

    hkpClosestCdPointCollector pointCollector;

    for (int i = 0; i < numShapes; i++)
    {
        const hkpShape* shape = shapes[i];

        // Get the appropriate function
        hkpCollisionDispatcher::GetClosestPointsFunc getClosestPoints = collisionDispatcher->getGetClosestPointsFunc(sphereShape.getType(), shape->getType());
        HK_ASSERT2(0x3432432c, getClosestPoints, "We need a function to find the distance");
        if (!getClosestPoints)
        {
            continue;
        }

        hkpCdBody boneBody(shape, &shapeTransforms[i]);

        // Okay lets work out what we got
        pointCollector.reset();

        // Collect the points...
        getClosestPoints(sphereBody, boneBody, collisionInput, pointCollector);

        // Find the closest distance
        if (pointCollector.hasHit())
        {
            hkSimdReal distance = pointCollector.getHit().m_contact.getDistanceSimdReal();

            if (distance < minDist)
            {
                minShapeIndex = i;
                minDist = distance;
            }
        }
    }

    return minShapeIndex;
}

/* static */void hkpShapeSkinningUtil::calculateSkinning(const Input& input, hkArray<Entry>& entries)
{
	const int numBones = input.m_numBones;
	const int numBonesPerVertex = input.m_bonesPerVertex;

	const int numValues = numBonesPerVertex * input.m_numVertices;
	entries.setSize(numValues);

	// Reset them all
	hkFindVertexWeightsUtil::reset(entries.begin(), numBonesPerVertex, input.m_numVertices);

	// Create the sphere shape we are going to use
	hkpSphereShape sphereShape(0.0f);

	// Set up the collision input

	hkpNullCollisionFilter shapeFilter;
	hkpCollisionInput collisionInput;
	collisionInput.m_filter = &shapeFilter;
	collisionInput.m_dispatcher = input.m_collisionDispatcher;
	collisionInput.setTolerance( input.m_maxOutside );


	{
		// Set up the bodies with the transform
		hkTransform sphereTransform;
		sphereTransform.setIdentity();

		hkSimdReal maxInside; maxInside.load<1>(&input.m_maxInside);
		hkSimdReal maxOutside; maxOutside.load<1>(&input.m_maxOutside);

		Entry* vertexEntries = entries.begin();

		hkpAllCdPointCollector pointCollector;
		// Okay - for each vertex I need to find the bones that it collides with
		const int numVertices = input.m_numVertices;
		for (int i = 0; i < numVertices; i++, vertexEntries += numBonesPerVertex)
		{
			const hkVector4& vertexPosition = input.m_vertexPositions[i];
			sphereTransform.getTranslation() = vertexPosition;

			hkpCdBody sphereBody(&sphereShape, &sphereTransform);

			int numVertexValues = 0;
			hkReal maxDistSq = hkFindVertexWeightsUtil::getMaxValue(vertexEntries, numBonesPerVertex);

			for (int j = 0; j < numBones; j++)
			{
				const hkpShape* boneShape = input.m_shapes[j];

				// Get the appropriate function
				hkpCollisionDispatcher::GetClosestPointsFunc getClosestPoints = input.m_collisionDispatcher->getGetClosestPointsFunc(sphereShape.getType(), boneShape->getType());
				HK_ASSERT2(0x3432432d, getClosestPoints, "We need a function to find the distance");
				if (!getClosestPoints)
				{
					continue;
				}

				hkpCdBody boneBody(boneShape, &input.m_transforms[j]);

				// Okay lets work out what we got
				pointCollector.reset();

				// Collect the points...
				getClosestPoints(sphereBody, boneBody, collisionInput, pointCollector);

				// Find the closest distance
				if (pointCollector.hasHit())
				{
					// If its not hit, then we just have to tie to a single bone

					const hkArray<hkpRootCdPoint>& hits = pointCollector.getHits();
					const int numHits = hits.getSize();
					for (int k = 0; k < numHits; k++)
					{
						const hkpRootCdPoint& point = hits[k];

						hkSimdReal distance = point.m_contact.getDistanceSimdReal();
						if (distance.isLessZero())
						{
							distance.setMax(-maxInside, distance);
						}
						else
						{
							if (distance > maxOutside)
							{
								continue;
							}
						}
						distance.add(maxInside);

						// We want it squared for the rest of the algorithm
						const hkSimdReal distSq = distance * distance;
						if (distSq < hkSimdReal::fromFloat(maxDistSq))
						{
							numVertexValues = hkFindVertexWeightsUtil::insertEntry(vertexEntries, numVertexValues, numBonesPerVertex, distance.getReal(), j);
							maxDistSq = hkFindVertexWeightsUtil::getMaxValue(vertexEntries, numBonesPerVertex);
						}
					}
				}
			}
		}
	}
}


/* static */hkResult hkpShapeSkinningUtil::setSkinningValues(const Input& inputIn, hkMeshVertexBuffer* buffer)
{
    Input input = inputIn;

    hkVertexFormat vertexFormat;
    buffer->getVertexFormat(vertexFormat);

    input.m_bonesPerVertex = hkSkinningUtil::findNumBoneIndices(vertexFormat);
    if (input.m_bonesPerVertex <= 0)
    {
        return HK_FAILURE;
    }


    // Skin it
    hkArray<hkVector4> vertexPositions;
    HK_ON_DEBUG( hkResult res = ) hkMeshVertexBufferUtil::getElementVectorArray(buffer, hkVertexFormat::USAGE_POSITION, 0, vertexPositions);
    HK_ASSERT(0xd8279a20, res == HK_SUCCESS);

    input.m_vertexPositions = vertexPositions.begin();
    input.m_numVertices = vertexPositions.getSize();

    // Work out the skinning value
    hkArray<hkpShapeSkinningUtil::Entry> entries;
    hkpShapeSkinningUtil::calculateSkinning(input, entries);

    // Set the values
    hkSkinningUtil::setSkinningValues(entries, buffer, input.m_maxInside + input.m_maxOutside);

    return HK_SUCCESS;
}


/* static */hkResult hkpShapeSkinningUtil::setSkinningValues(const Input& inputIn, hkMeshShape* meshShape)
{
    hkMeshSectionLockSet sectionSet;
    sectionSet.addMeshSections(meshShape, hkMeshShape::ACCESS_VERTEX_BUFFER);

    hkArray<hkMeshVertexBuffer*> uniqueVertexBuffers;
    sectionSet.findUniqueVertexBuffers(uniqueVertexBuffers);

    const int numBuffers = uniqueVertexBuffers.getSize();
    for (int i = 0; i < numBuffers; i++)
    {
        hkMeshVertexBuffer* buffer = uniqueVertexBuffers[i];
        setSkinningValues(inputIn, buffer);
    }
    return HK_SUCCESS;
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
