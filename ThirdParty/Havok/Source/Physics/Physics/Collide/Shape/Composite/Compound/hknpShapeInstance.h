/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_SHAPE_INSTANCE_H
#define HKNP_SHAPE_INSTANCE_H

#include <Physics/Physics/Collide/Shape/hknpShape.h>


/// An instance of a shape with its own transform and shape tag, for use in composite shapes.
struct hknpShapeInstance
{
	public:

		enum Flags
		{
			// Build-time flags.
			HAS_TRANSLATION		=	1<<1,	///< Has a non-zero translation.
			HAS_ROTATION		=	1<<2,	///< Has a non-identity rotation.
			HAS_SCALE			=	1<<3,	///< Has a non-identity scale.
			FLIP_ORIENTATION	=	1<<4,	///< Scale has an odd number of sign changes.
			SCALE_SURFACE		=	1<<5,	///< Scale surfaces rather than vertices. See hknpShape::ScaleMode.

			// Run-time flags.
			IS_ENABLED			=	1<<6,	///< Set if this instance is enabled (true upon creation).

			DEFAULT_FLAGS		=	IS_ENABLED
		};

		enum
		{
			NUM_PADDING_BYTES = 32,			// Number of padding bytes in m_padding
			EMPTY_FLAG_PADDING_BYTE = 4		// Position of the byte used to flag a shape instance as empty in the free list array
		};

	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpShapeInstance );
		HK_DECLARE_REFLECTION();

#if !defined( HK_PLATFORM_SPU )

		/// Default constructor.
		HK_FORCE_INLINE hknpShapeInstance();

		/// Construct from a shape and a transform.
		HK_FORCE_INLINE hknpShapeInstance( const hknpShape* shape, const hkTransform& transform = hkTransform::getIdentity() );

		/// Serialization constructor.
		HK_FORCE_INLINE hknpShapeInstance( hkFinishLoadedObjectFlag flag );

		/// Set the shape for this instance.
		void setShape( const hknpShape* shape );

		/// Set the transform for this instance.
		void setTransform( const hkTransform& transform );

		/// Set the scale for this instance.
		void setScale( hkVector4Parameter scale, hknpShape::ScaleMode mode = hknpShape::SCALE_SURFACE );

		/// Set the shape tag for this instance.
		HK_FORCE_INLINE void setShapeTag( hknpShapeTag shapeTag );

		/// Enable or disable this instance.
		HK_FORCE_INLINE void setEnabled( bool isEnabled );

		/// Set the number of bytes that need to be transferred to bring the instance from main to local memory on
		/// non-unified memory model platforms (SPU, GPU).
		HK_FORCE_INLINE void setShapeMemorySize( int size );

		/// For internal use only.
		/// Set the tree node index of this instance.
		HK_FORCE_INLINE void setLeafIndex( int index );

#endif // !HK_PLATFORM_SPU

		/// Get the flags for this instance.
		HK_FORCE_INLINE int getFlags() const;

		/// Get the shape for this instance.
		HK_FORCE_INLINE const hknpShape* getShape() const;

		/// Get the transform for this instance.
		HK_FORCE_INLINE const hkTransform& getTransform() const;

		/// Get the scale for this instance.
		HK_FORCE_INLINE const hkVector4& getScale() const;

		/// Get the scale mode for this instance.
		HK_FORCE_INLINE hknpShape::ScaleMode getScaleMode() const;

		/// Get the shape tag for this instance.
		HK_FORCE_INLINE hknpShapeTag getShapeTag() const;

		/// Get whether this instance is enabled.
		HK_FORCE_INLINE bool isEnabled() const;

		/// Get the number of bytes that need to be transferred to bring the instance from main to local memory on
		/// non-unified memory model platforms (SPU, GPU).
		HK_FORCE_INLINE int getShapeMemorySize() const;

		/// For internal use only.
		/// Get the tree node index of this instance.
		HK_FORCE_INLINE int getLeafIndex() const;

		/// Get the full transform (including scale) for this instance.
		void getFullTransform( hkTransform& transformOut ) const;

		/// Get the full inverse transform (including scale) for this instance.
		void getFullTransformInverse( hkTransform& transformOut ) const;

		/// Compute the AABB of this instance (in compound shape shape).
		void calculateAabb( hkAabb& aabbOut ) const;

		// Operations for shape instances in a free list array.
		HK_FORCE_INLINE static void setEmpty( hknpShapeInstance& shapeInstance, hkUint32 next );
		HK_FORCE_INLINE static hkUint32 getNext( const hknpShapeInstance& shapeInstance );
		HK_FORCE_INLINE static hkBool32 isEmpty( const hknpShapeInstance& shapeInstance );

	protected:

		/// Set the flags for this instance.
		HK_FORCE_INLINE void setFlags( int flags );

	protected:

		hkTransform					m_transform;
		hkVector4					m_scale;
		hkRefPtr<const hknpShape>	m_shape;
		hknpShapeTag				m_shapeTag;
		hkUint8						m_padding[NUM_PADDING_BYTES];
};

/// An identifier for shape instances in a compound shape.
HK_DECLARE_HANDLE( hknpShapeInstanceId, hkInt16, 0x7fff );

#include <Physics/Physics/Collide/Shape/Composite/Compound/hknpShapeInstance.inl>

#endif // HKNP_SHAPE_INSTANCE_H

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
