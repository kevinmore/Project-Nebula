/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

// PrimitiveMediator definition and implementation

#ifndef HK_COLLIDE2_MOPP_MEDIATOR_H
#define HK_COLLIDE2_MOPP_MEDIATOR_H

#include <Physics2012/Internal/Collide/Mopp/Builder/Splitter/hkpMoppSplitTypes.h>
#include <Physics2012/Internal/Collide/Mopp/Code/hkpMoppCode.h>

/// the Mediator is the interface to set of convex objects
/// the mediator can use two integer ids to identify
/// each primitive.
/// The main job of the mediator is to project a set of primitives
/// onto a straight and return the max and min value of this
/// projection.
class hkpMoppMediator : public hkReferencedObject
{
public:
HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_BASE);
	virtual ~hkpMoppMediator(){;}

	/// Tell mediator about the splitting planes
	virtual void setSplittingPlaneDirections(const hkpMoppSplittingPlaneDirection* directions, int numDirections) = 0;

	/// Returns the total number of primitives handles by the mediator
	virtual int  getNumPrimitives() = 0;

	/// Gets the primitiveId and primitiveIi2 in the out array.
	/// Each primitive can be represented by two integer numbers
	virtual void getPrimitives(hkpMoppCompilerPrimitive *primitiveArrayOut) = 0;

	/// Fill the m_extent members in the primitiveArray
	virtual void projectPrimitives(const hkVector4 &direction, int directionIndex, hkpMoppCompilerPrimitive *primitiveArray, int numPrimitives, hkReal *absMinOut, hkReal *absMaxOut) = 0;

	/// Only get the min max values
	virtual void findExtents(const hkVector4 &direction, int directionIndex, const hkpMoppCompilerPrimitive *primitiveArray, int numPrimitives, hkReal *absMinOut, hkReal *absMaxOut) = 0;

	/// Split a primitive and store the result into *primitiveOut
	virtual void splitPrimitive( const hkpMoppCompilerPrimitive &primtiveIn, const hkVector4 &direction, hkReal planeOffset, int treeDepth, hkpMoppCompilerPrimitive *primitiveOut ) = 0;

	/// Get additional properties connected to the primitive.
	/// Returns the number of properties.
	/// \note The maximum number of properties is hkpMoppCode::MAX_PRIMITIVE_PROPERTIES
	virtual int getPrimitiveProperties( const hkpMoppCompilerPrimitive &primitiveIn, hkpPrimitiveProperty propertiesOut[hkpMoppCode::MAX_PRIMITIVE_PROPERTIES]) = 0;
};

#endif // HK_COLLIDE2_MOPP_MEDIATOR_H

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
