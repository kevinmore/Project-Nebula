/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_SHAPE_SCALING_UTIL_H
#define HKNP_SHAPE_SCALING_UTIL_H

class hknpShape;

/// This utility class scales physics shapes.
/// Note that this is not meant to be used at runtime, but in the tool chain and preprocess stages.
class hknpShapeScalingUtil
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_PHYSICS, hknpShapeScalingUtil);

	public:

		// Pair of (original, scaled) physics shapes
		struct ShapePair
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_PHYSICS, hknpShapeScalingUtil::ShapePair);

			/// Old shape.
			hkRefPtr<const hknpShape> m_originalShape;

			/// New shape. Null if not new.
			hkRefPtr<hknpShape> m_newShape;
		};

		/// This will try to scale the given shape by the given amount.
		/// If you are doing multiple scale calls and you have shared shapes,
		/// you might want to provide an array to persist between calls
		/// to stop the shapes from being shrunk more than once.
		/// This function is recursive.
		static hknpShape* HK_CALL scaleShape(const hknpShape* shape, hkVector4Parameter vScale, hkArray<ShapePair>* doneShapes = HK_NULL);
};

#endif // HKNP_SHAPE_SCALING_UTIL_H

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
