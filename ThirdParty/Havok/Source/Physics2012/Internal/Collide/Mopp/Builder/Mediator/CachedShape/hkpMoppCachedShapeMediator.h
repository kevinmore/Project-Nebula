/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HK_COLLIDE2_MOPP_CACHED_SHAPE_MEDIATOR_H
#define HK_COLLIDE2_MOPP_CACHED_SHAPE_MEDIATOR_H

#include <Physics2012/Internal/Collide/Mopp/Builder/Mediator/hkpMoppMediator.h>

class hkpShapeContainer;
class hkpConvexShape;

	/// Primitive Mediator Implementation that caches each primitive's maximum/minimum extent for all supplied splitting axes.
	///
	/// This version of the shape primitive mediator only calculates a primitive's the maximum/minimum extent along all splitting
	/// axes only once upfront and caches the result. This will improve the runtime performance of the mediator but also increases
	/// the mediator's memory consumption: for each of the shape's primitive an additional 108 bytes block is allocated.
class hkpMoppCachedShapeMediator : public hkpMoppMediator 
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_MOPP );

		hkpMoppCachedShapeMediator(const hkpShapeContainer* shape);		
		~hkpMoppCachedShapeMediator();												
		
	public:
		void setSplittingPlaneDirections(const hkpMoppSplittingPlaneDirection* directions, int numDirections);

		// returns the number of compiler primitives
		int  getNumPrimitives();											

		void splitPrimitive(const hkpMoppCompilerPrimitive& primtiveIn, const hkVector4 &direction, hkReal planeOffset, int treeDepth, hkpMoppCompilerPrimitive* primitiveOut);
		void getPrimitives(hkpMoppCompilerPrimitive* primitivesOut);
		void projectPrimitives(const hkVector4& direction, int directionIndex, hkpMoppCompilerPrimitive* primitiveArray, int numPrimitives, hkReal* absMinOut, hkReal* absMaxOut); 
		void findExtents(const hkVector4& direction, int directionIndex, const hkpMoppCompilerPrimitive* primitiveArray, int numPrimitives, hkReal* absMinOut, hkReal* absMaxOut);
		int  getPrimitiveProperties(const hkpMoppCompilerPrimitive &primitiveIn, hkpPrimitiveProperty propertiesOut[hkpMoppCode::MAX_PRIMITIVE_PROPERTIES]);
		
	private:
		void addConvexShape(const hkpConvexShape* convexShape, hkpShapeKey key, const hkpMoppSplittingPlaneDirection* directions, int numDirections);

		HK_FORCE_INLINE void projectPrimitive(const hkpMoppCompilerPrimitive& primtiveIn, int directionIndex, hkReal* minimum, hkReal* maximum);

	protected:

		struct hkpConvexShapeData
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_MOPP, hkpMoppCachedShapeMediator::hkpConvexShapeData );

			hkpShapeKey m_key;
			hkpMoppExtent m_extents[13];
		};

		hkArray<hkpConvexShapeData> m_arrayConvexShapeData;
		const hkpShapeContainer* m_shapeCollection;
		int m_numChildShapes;
};


#endif // HK_COLLIDE2_MOPP_CACHED_SHAPE_MEDIATOR_H

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
