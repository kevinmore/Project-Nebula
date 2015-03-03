/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Collide/Query/hknpCollisionQueryDispatcher.h>
#include <Physics/Physics/Extensions/ShapeProcessing/ShapeSkinning/hknpShapeSkinningUtil.h>
#include <Physics/Physics/Collide/Shape/Convex/Sphere/hknpSphereShape.h>
#include <Physics/Physics/Collide/Shape/hknpShapeQueryInterface.h>
#include <Physics/Physics/Collide/Query/Collector/hknpAllHitsCollector.h>
#include <Physics/Physics/Collide/Query/Collector/hknpClosestHitCollector.h>

#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>
#include <Common/GeometryUtilities/Mesh/Utils/MeshSectionLockSet/hkMeshSectionLockSet.h>
#include <Common/GeometryUtilities/Mesh/Utils/VertexBufferUtil/hkMeshVertexBufferUtil.h>
#include <Common/GeometryUtilities/Mesh/hkMeshShape.h>

//
//	Constructor

hknpShapeSkinningUtil::Input::Input()
:	m_maxInside(1.0f)
,	m_maxOutside(1.0f)
,	m_shapes(HK_NULL)
,	m_transforms(HK_NULL)
,	m_numBones(0)
,	m_bonesPerVertex(4)
,	m_vertexPositions(HK_NULL)
,	m_numVertices(0)
{}

//
//	Find a list of shapes and transforms, finds the one which is closest. If non are in maxDistance range returns -1.

int HK_CALL hknpShapeSkinningUtil::findClosestShape(const hkArray<hkTransform>& shapeTransforms, const hkArray<const hknpShape*>& shapes, hkReal maxDistance, hkVector4Parameter point)
{
	HK_ASSERT(0x2963cb69, shapeTransforms.getSize() == shapes.getSize());

	// Create the sphere shape we are going to use
	hkVector4 vZero;			vZero.setZero();
	hkRefPtr<hknpShape> sphere;	sphere.setAndDontIncrementRefCount(hknpSphereShape::createSphereShape(vZero, 0.0f));

	// Set up the collision input
	hknpCollisionQueryDispatcher dispatcher;
	hknpCollisionQueryContext queryContext;		queryContext.m_dispatcher = &dispatcher;
	hknpQueryFilterData targetShapeFilterData;

	// Set up the sphere transform
	hkTransform sphereTransform;
	sphereTransform.setIdentity();
	sphereTransform.setTranslation(point);

	// For each shape, track the closest distance
	hknpClosestHitCollector collector;
	hknpShapeQueryInfo queryShapeInfo;	queryShapeInfo.m_shapeToWorld = &sphereTransform;
	hknpClosestPointsQuery	query		(*sphere, maxDistance);
	hkReal minDist						= HK_REAL_MAX;
	int minShapeIndex					= -1;

	for (int i = shapes.getSize() - 1; i >= 0; i--)
	{
		const hknpShape* boneShape				= shapes[i];
		hknpShapeQueryInfo targetShapeInfo;		targetShapeInfo.m_shapeToWorld	= &shapeTransforms[i];

		// Collect the points
		collector.reset();
		hknpShapeQueryInterface::getClosestPoints(&queryContext, query, queryShapeInfo, *boneShape, targetShapeFilterData, targetShapeInfo, &collector);

		// Find the closest distance
		if ( collector.hasHit() )
		{
			const hkReal distance = collector.getHits()->m_fraction;

			if ( distance < minDist )
			{
				minShapeIndex = i;
				minDist = distance;
			}
		}
	}

    return minShapeIndex;
}

//
//	Calculates the weights and bone indices based on the input. There will be m_bonesPerVertex * m_numVertices entries - each m_bonesPerVertex run
//	for a vertex. A bone index in an entry = -1 means that a bone close enough could not be found.

void HK_CALL hknpShapeSkinningUtil::calculateSkinning(const Input& input, hkArray<Entry>& entries)
{
	const int numBones			= input.m_numBones;
	const int numBonesPerVertex	= input.m_bonesPerVertex;
	const int numValues			= numBonesPerVertex * input.m_numVertices;
	entries.setSize(numValues);

	// Reset them all
	hkFindVertexWeightsUtil::reset(entries.begin(), numBonesPerVertex, input.m_numVertices);

	// Create the sphere shape we are going to use
	hkVector4 vZero;			vZero.setZero();
	hkRefPtr<hknpShape> sphere;	sphere.setAndDontIncrementRefCount(hknpSphereShape::createSphereShape(vZero, 0.0f));

	// Set up the collision input
	hknpCollisionQueryDispatcher dispatcher;
	hknpCollisionQueryContext queryContext;		queryContext.m_dispatcher = &dispatcher;
	hknpQueryFilterData targetShapeFilterData;

	{
		// Set up the bodies with the transform
		hkTransform sphereTransform;		sphereTransform.setIdentity();
		const hkReal maxInside				= input.m_maxInside;
		const hkReal maxOutside				= input.m_maxOutside;
		Entry* HK_RESTRICT vertexEntries	= entries.begin();

		// For each vertex, find the bones that it collides with
		hknpAllHitsCollector collector;
		hknpShapeQueryInfo queryShapeInfo;	queryShapeInfo.m_shapeToWorld = &sphereTransform;
		const int numVertices				= input.m_numVertices;

		for (int i = 0; i < numVertices; i++, vertexEntries += numBonesPerVertex)
		{
			const hkVector4& vertexPosition		= input.m_vertexPositions[i];
			sphereTransform.getTranslation()	= vertexPosition;
			hknpClosestPointsQuery	query		(*sphere, input.m_maxOutside);
			int numVertexValues					= 0;
			hkReal maxDistSq					= hkFindVertexWeightsUtil::getMaxValue(vertexEntries, numBonesPerVertex);

			for (int j = 0; j < numBones; j++)
			{
				const hknpShape* boneShape				= input.m_shapes[j];
				hknpShapeQueryInfo targetShapeInfo;		targetShapeInfo.m_shapeToWorld	= &input.m_transforms[j];

				// Collect the points
				collector.reset();
				hknpShapeQueryInterface::getClosestPoints(&queryContext, query, queryShapeInfo, *boneShape, targetShapeFilterData, targetShapeInfo, &collector);

				// Find the closest distance
				if ( collector.hasHit() )
				{
					// If it's not hit, then we just have to tie to a single bone
					const hknpCollisionResult* hits = collector.getHits();
					const int numHits = collector.getNumHits();

					for (int k = 0; k < numHits; k++)
					{
						hkReal distance = hits[k].m_fraction;
						if ( distance < 0.0f )
						{
							if ( distance < -maxInside )
							{
								distance = -maxInside;
							}
						}
						else
						{
							if ( distance > maxOutside )
							{
								continue;
							}
						}
						distance += maxInside;

						// We want it squared for the rest of the algorithm
						const hkReal distSq = distance * distance;
						if ( distSq < maxDistSq )
						{
							numVertexValues = hkFindVertexWeightsUtil::insertEntry(vertexEntries, numVertexValues, numBonesPerVertex, distance, j);
							maxDistSq = hkFindVertexWeightsUtil::getMaxValue(vertexEntries, numBonesPerVertex);
						}
					}
				}
			}
		}
	}
}

//
//	Set the skinning values on a vertex buffer

hkResult HK_CALL hknpShapeSkinningUtil::setSkinningValues(const Input& inputIn, hkMeshVertexBuffer* buffer)
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
    hkArray<Entry> entries;
    calculateSkinning(input, entries);

    // Set the values
    hkSkinningUtil::setSkinningValues(entries, buffer, input.m_maxInside + input.m_maxOutside);

    return HK_SUCCESS;
}

//
//	Set the skinning values on a shape

hkResult hknpShapeSkinningUtil::setSkinningValues(const Input& inputIn, hkMeshShape* meshShape)
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
