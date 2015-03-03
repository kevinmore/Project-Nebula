/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_SHAPE_COLLECTOR_H
#define HKNP_SHAPE_COLLECTOR_H

#include <Physics/Physics/Collide/Shape/Convex/Triangle/hknpTriangleShape.h>
#include <Physics/Physics/Collide/Shape/TagCodec/hknpShapeTagCodec.h>


/// Input/output structure for hknpShape::getLeafShape().
struct hknpShapeCollector
{
	public:

#if !defined(HK_PLATFORM_SPU)

		#define HKNP_SHAPE_COLLECTOR_BUFFER_SIZE HKNP_SHAPE_BUFFER_SIZE

#else

		#define HKNP_SHAPE_COLLECTOR_BUFFER_SIZE (HKNP_SHAPE_BUFFER_SIZE + HKNP_COMPOUND_HIERARCHY_BUFFER)

#endif

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpShapeCollector );

		/// Constructor.
		/// You need to pass in a thread local reference triangle shape.
		/// You can use hknpShapeCollectorWithInplaceTriangle for convenience instead.
		HK_FORCE_INLINE hknpShapeCollector( hknpTriangleShape* triangleShapePrototype );

		/// Everytime you are using this collector, you have to call reset() first.
		/// \a transform : this should be the world transform of the shape you call getLeafShape() on.
		HK_FORCE_INLINE void reset( const hkTransform& transform );

		/// Check if reset() has been called.
		HK_FORCE_INLINE void checkForReset();

	public:

		/// The "shape to world" output transform.
		/// Note: A shape's getLeafShape() implementation may or may not modify this parameter.
		/// If it modifies it, it must set m_transformModifiedFlag.
		hkTransform m_transformOut;

		/// A scale to apply to the returned shape.
		/// A shape's getLeafShape() should bake this into the returned shape and reset it to identity.
		hkVector4 m_scaleOut;

		HK_PAD_ON_SPU( hkBool32 ) m_transformModifiedFlag;

		/// Shape output.
		HK_PAD_ON_SPU( const hknpShape* ) m_shapeOut;

		/// The Shape Tag. See hknpShapeTagCodec for more information on how data is encoded in this value.
		/// This value is only the shape's local tag. To retrieve the global tag data (taking the full shape hierarchy
		/// into account) you need to call hknpShapeTagCodec::decode() using m_shapeTagPath.
		HK_PAD_ON_SPU( hknpShapeTag ) m_shapeTagOut;

		/// A pointer to a triangle shape prototype, which has the topology data filled in but no vertex or plane values.
		/// This can be filled with vertex data and returned.
		/// This triangle must not be shared by different threads.
		HK_PAD_ON_SPU( hknpTriangleShape* ) m_triangleShapePrototype;

		/// The parent shape for the output shape. HK_NULL if the returned shape is at root level.
		HK_PAD_ON_SPU( const hknpShape* ) m_parentShape;

		/// The path of shape tags (and context) from the hierarchy's root shape down to \a m_shapeOut,
		/// excluding \a m_shapeOut itself.
		hknpShapeTagPath m_shapeTagPath;

		/// Local buffer where non-triangle shapes may be created to be returned in m_shapeOut.
		HK_ALIGN16( hkUint8 m_shapeBuffer[ HKNP_SHAPE_COLLECTOR_BUFFER_SIZE ] );

#if defined(HK_PLATFORM_SPU)

		/// A pointer into m_shapeBuffer. This will advance on SPU when descending into a shape hierarchy.
		/// Available on SPU only.
		hkUint8* m_shapeBufferPtr;

		/// The remaining space in the m_shapeBuffer.
		/// Available on SPU only.
		int m_shapeBufferSize;

#endif
};


#if !defined(HK_PLATFORM_SPU)

/// Convenient version of hknpShapeCollector with an inplace triangle shape.
struct hknpShapeCollectorWithInplaceTriangle : public hknpShapeCollector
{
	public:

		hknpShapeCollectorWithInplaceTriangle();

		~hknpShapeCollectorWithInplaceTriangle();

	public:

		hknpInplaceTriangleShape m_inplaceTriangleShape;
};

#endif	// !HK_PLATFORM_SPU


#include <Physics/Physics/Collide/Shape/hknpShapeCollector.inl>


#endif // HKNP_SHAPE_COLLECTOR_H

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
