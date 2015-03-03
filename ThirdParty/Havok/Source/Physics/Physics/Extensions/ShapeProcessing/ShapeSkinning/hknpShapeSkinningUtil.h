/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_SHAPE_SKINNING_UTIL_H
#define HKNP_SHAPE_SKINNING_UTIL_H

#include <Common/GeometryUtilities/Mesh/Utils/FindVertexWeightsUtil/hkFindVertexWeightsUtil.h>

class hknpConvexShape;
class hknpShape;

/// Utility class that works out bone indices and weights from an array of physics shapes acting as bones.
class hknpShapeSkinningUtil
{
	public:

        HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_PHYSICS, hknpShapeSkinningUtil);

		typedef hkFindVertexWeightsUtil::Entry Entry;

	public:

		/// Input data
        struct Input
        {
    		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_PHYSICS, hknpShapeSkinningUtil::Input);

			/// Constructor
			Input();

			hkReal m_maxInside;									///< The maximum penetration distance before all distances are equated to being 'at' the bone
			hkReal m_maxOutside;								///< The maximum outside distance. If a shape is further away than this it will not be considered
			const hknpShape* const* HK_RESTRICT m_shapes;		///< A shape per bone (must be at least m_numBones of these)
			const hkTransform* HK_RESTRICT m_transforms;		///< A transform to place a shape per bone (must be at least m_numBones of these)
			int m_numBones;										///< the number of bones
			int m_bonesPerVertex;								///< The number of bones per vertex
			hkVector4* HK_RESTRICT m_vertexPositions;			///< The positions of the vertices
			int m_numVertices;									///< The number of vertices
        };

		/// Find a list of shapes and transforms, finds the one which is closest. If non are in maxDistance range returns -1.
		static int HK_CALL findClosestShape(const hkArray<hkTransform>& shapeTransforms, const hkArray<const hknpShape*>& shapes, hkReal maxDistance, hkVector4Parameter point);

		/// Calculates the weights and bone indices based on the input. There will be m_bonesPerVertex * m_numVertices entries - each m_bonesPerVertex run
		/// for a vertex. A bone index in an entry = -1 means that a bone close enough could not be found.
		static void HK_CALL calculateSkinning(const Input& input, hkArray<Entry>& entries);

		/// Set the skinning values on a vertex buffer
		static hkResult HK_CALL setSkinningValues(const Input& inputIn, hkMeshVertexBuffer* buffer);

		/// Set the skinning values on a shape
		static hkResult HK_CALL setSkinningValues(const Input& inputIn, hkMeshShape* meshShape);
};

#endif // HKNP_SHAPE_SKINNING_UTIL_H

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
