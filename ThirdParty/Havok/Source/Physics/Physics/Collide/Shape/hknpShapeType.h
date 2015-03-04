/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_SHAPE_TYPES_H
#define HKNP_SHAPE_TYPES_H


/// The concrete type of a shape.
/// Mainly used for shape visualization and debugging.
struct hknpShapeType
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpShapeType );
	HK_DECLARE_REFLECTION();

	enum Enum
	{
		// Convex shapes
		CONVEX					,	///< general convex without face info
		CONVEX_POLYTOPE			,	///< general convex with face info
		SPHERE					,
		CAPSULE					,
		TRIANGLE				,

		// Composite shapes
		COMPRESSED_MESH			,
		EXTERN_MESH				,
		STATIC_COMPOUND			,
		DYNAMIC_COMPOUND		,

		// Height field shapes
		HEIGHT_FIELD			,
		COMPRESSED_HEIGHT_FIELD ,

		// Wrapper shapes
		SCALED_CONVEX			,
		MASKED_COMPOSITE		,

		// User defined shapes
		USER_0					,
		USER_1					,
		USER_2					,
		USER_3					,

		NUM_SHAPE_TYPES			,
		INVALID
	};
};


/// A type defining how a shape should be treated during collision detection.
/// This determines which type of collision cache is created when a pair of bodies overlap in the broad phase,
/// which in turn determines which set of hknpShape interface methods are called in the narrow phase.
struct hknpCollisionDispatchType
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpCollisionDispatchType );
	HK_DECLARE_REFLECTION();

	enum Enum
	{
		/// Collision detection will use the convex interface methods of hknpShape to retrieve the vertices and faces
		/// of the convex shape.
		CONVEX,

		/// Collision detection will use the composite interface methods of hknpShape to retrieve the shape keys and
		/// leaf shapes of the composite shape, then perform further collision detection using those leaf shapes.
		COMPOSITE,

		/// Collision detection will use the getSignedDistances() function to collide only the getSupportVertices()
		/// of the other shape with this shape.
		DISTANCE_FIELD,

		/// A user type.
		/// Users must register a collision cache creation function with the hknpCollisionDispatcher in order for
		/// shapes using this type to have any effect.
		USER,

		/// End of dispatch types.
		NUM_TYPES
	};
};


#endif // HKNP_SHAPE_TYPES_H

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
