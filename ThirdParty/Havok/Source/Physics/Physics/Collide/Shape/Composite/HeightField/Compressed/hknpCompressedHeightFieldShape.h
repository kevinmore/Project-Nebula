/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_COMPRESSED_HEIGHT_FIELD_SHAPE_H
#define HKNP_COMPRESSED_HEIGHT_FIELD_SHAPE_H

#include <Physics/Physics/Collide/Shape/Composite/HeightField/hknpHeightFieldShape.h>


/// A height field shape which stores the heights in a simple 16bit format.
/// hknpCompressedHeightFieldShape is also used as a simple height field implementation on the SPU.
class hknpCompressedHeightFieldShape : public hknpHeightFieldShape
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
		HK_DECLARE_REFLECTION();

		/// Constructor.
		hknpCompressedHeightFieldShape( const hknpHeightFieldShapeCinfo& cinfo,
			hknpHeightFieldShape* hf );

		/// Constructor.
		hknpCompressedHeightFieldShape( const hknpHeightFieldShapeCinfo& cinfo,
			const hkArray<hkUint16>& samples,  hkReal quantizationOffset, hkReal quantizationScale,
			hkArray<hknpShapeTag>* shapeTags = HK_NULL );

		/// Constructor.
		hknpCompressedHeightFieldShape( const hknpHeightFieldShapeCinfo& cinfo,
			const hkArray<hkReal>& samples, hkArray<hknpShapeTag>* shapeTags = HK_NULL );

		/// Serialization constructor.
		hknpCompressedHeightFieldShape( hkFinishLoadedObjectFlag f );

		/// Decompress a 16bit value to full float.
		HK_FORCE_INLINE hkReal decompress( hkUint16 comp ) const;

		/// Compress a full float value to 16bit.
		HK_FORCE_INLINE hkUint16 compress( hkReal uncomp ) const;

		//
		// hknpHeightFieldShape implementation
		//

		HK_ON_CPU( virtual ) void getQuadInfoAt( int x, int z,
			hkVector4* heightOut, hknpShapeTag* shapeTagOut, hkBool32 *triangleFlipOut ) const HK_ON_CPU( HK_OVERRIDE );

		//
		// hknpShape implementation
		//

		virtual hknpShapeType::Enum getType() const HK_OVERRIDE { return hknpShapeType::COMPRESSED_HEIGHT_FIELD; }

		virtual int calcSize() const HK_OVERRIDE;

	protected:

		HKNP_DECLARE_SHAPE_VTABLE_UTIL_CONSTRUCTOR( hknpCompressedHeightFieldShape, hknpHeightFieldShape );

	public:

		
		hkArray<hkUint16> m_storage;
		hkArray<hknpShapeTag> m_shapeTags;
		hkBool m_triangleFlip;

		// Quantization information: the uncompressed (real) value is computed by
		//		Uncompressed = scale*compressed + offset
		// Scale has a factor of 2^16 - 1 baked in
		hkReal m_offset;
		hkReal m_scale;
};

#include <Physics/Physics/Collide/Shape/Composite/HeightField/Compressed/hknpCompressedHeightFieldShape.inl>


#endif // HKNP_COMPRESSED_HEIGHT_FIELD_SHAPE_H

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
