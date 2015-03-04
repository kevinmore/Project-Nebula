/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

// PrimitiveMediator definition and implementation

#ifndef HK_COLLIDE2_MOPP_SHAPE_MEDIATOR_H
#define HK_COLLIDE2_MOPP_SHAPE_MEDIATOR_H

#include <Physics2012/Internal/Collide/Mopp/Builder/Mediator/hkpMoppMediator.h>

class hkpShapeContainer;

#define HK_MOPP_SHAPE_MEDIATOR_MAX_SHAPES 0x10000

/// Shape primitive mediator definition.
/// Note on splitting primitives
class hkpMoppShapeMediator: public hkpMoppMediator 
{
	public:
		HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_BASE);

		// Mediator constructor 
		hkpMoppShapeMediator(const hkpShapeContainer* shape);		
		// Shape destructor
		~hkpMoppShapeMediator();												
		
	public:

		void setSplittingPlaneDirections(const hkpMoppSplittingPlaneDirection* directions, int numDirections);

		// returns the number of compiler primitives
		int  getNumPrimitives();											

		void splitPrimitive( const hkpMoppCompilerPrimitive &primtiveIn, const hkVector4 &direction, hkReal planeOffset, int treeDepth, hkpMoppCompilerPrimitive* primitiveOut );
		void getPrimitives(hkpMoppCompilerPrimitive* primitivesOut);
		void projectPrimitives(const hkVector4 &direction, int directionIndex, hkpMoppCompilerPrimitive* primitiveArray, int numPrimitives, hkReal* absMinOut, hkReal* absMaxOut); 
		void findExtents(const hkVector4 &direction, int directionIndex, const hkpMoppCompilerPrimitive* primitiveArray, int numPrimitives, hkReal* absMinOut, hkReal* absMaxOut);
		int  getPrimitiveProperties( const hkpMoppCompilerPrimitive &primitiveIn, hkpPrimitiveProperty propertiesOut[hkpMoppCode::MAX_PRIMITIVE_PROPERTIES]);

	private:
		HK_FORCE_INLINE void projectPrimitive( const hkpMoppCompilerPrimitive &primtiveIn, const hkVector4 &direction, int directionIndex, hkReal* minimum, hkReal* maximum );

	protected:
		const hkpShapeContainer* m_shape;
		int   m_numChildShapes;

};


#endif // HK_COLLIDE2_MOPP_SHAPE_MEDIATOR_H

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
