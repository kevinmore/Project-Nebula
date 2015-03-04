/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_SYSTEM_SCALING_UTIL_H
#define HKNP_SYSTEM_SCALING_UTIL_H

class hknpPhysicsSceneData;
class hknpPhysicsSystemData;

#include <Physics/Physics/Extensions/ShapeProcessing/ShapeScaling/hknpShapeScalingUtil.h>

/// This utility class uniformly scales physics systems.
/// Note that this is not meant to be used at runtime, but in the tool chain and preprocess stages.
class hknpPhysicsSystemDataUtil
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_PHYSICS, hknpPhysicsSystemDataUtil);

		typedef hknpShapeScalingUtil::ShapePair	ShapePair;

	public:

		/// Scales all the rigid bodies in the given physics system data
		static void HK_CALL scaleSystem(hknpPhysicsSceneData* physicsSceneData, hknpPhysicsSystemData* physicsSystemData, hkSimdRealParameter uniformScale, hkArray<ShapePair>& doneShapes);

		/// Scales all the rigid bodies in the given physics scene data
		static void HK_CALL scaleScene(hknpPhysicsSceneData* physicsData, hkSimdRealParameter uniformScale, hkArray<ShapePair>* doneShapes = HK_NULL);
};

#endif // HKNP_SYSTEM_SCALING_UTIL_H

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
