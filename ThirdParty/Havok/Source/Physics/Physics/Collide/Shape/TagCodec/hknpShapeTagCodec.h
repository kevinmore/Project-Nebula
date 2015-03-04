/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_SHAPE_TAG_CODEC_H
#define HKNP_SHAPE_TAG_CODEC_H

#include <Physics/Physics/Collide/Query/hknpCollisionQuery.h>

class hknpMaterialPalette;
struct hknpShapeTagPathEntry;


/// The Shape Tag Codec base class.
/// Every Shape Tag Codec should provide an encoding scheme to store 3 values in the shape tag:
/// - a material id.
/// - a collision filter info.
/// - some arbitrary user data.
/// The exact encoding (and thus the decision how many bits are spent on what value) is up to the codec to decide.
class hknpShapeTagCodec : public hkReferencedObject
{
	//+version(1)

	public:

		/// Codec types
		enum CodecType
		{
			NULL_CODEC,
			MATERIAL_PALETTE_CODEC,
			UFM_CODEC,
			USER_CODEC
		};

		/// The runtime/collision context of a (to be decoded) shape tag.
		struct Context
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_PHYSICS, Context);

			hknpCollisionQueryType::Enum		m_queryType;		///< Information on the calling context.

			HK_PAD_ON_SPU( const hknpBody* )	m_body;				///< The tagged shape's governing body. Can be HK_NULL.
			HK_PAD_ON_SPU( const hknpShape* )	m_rootShape;		///< The tagged shape's topmost ancestor. Identical to the shape in m_body (if available).
			HK_PAD_ON_SPU( const hknpShape* )	m_parentShape;		///< The tagged shape's parent shape. It is required by codecs that make use of hknpCompositeShape::m_shapeTagCodecInfo for decoding
			HK_PAD_ON_SPU( hknpShapeKey )		m_shapeKey;			///< The tagged shape's full shape key.
			HK_PAD_ON_SPU( const hknpShape* )	m_shape;			///< The tagged shape. Can be HK_NULL.

			HK_PAD_ON_SPU( const hknpBody* )	m_partnerBody;		///< The (colliding) partner shape's governing body. Can be HK_NULL.
			HK_PAD_ON_SPU( const hknpShape* )	m_partnerRootShape;	///< The (colliding) partner shape's topmost ancestor. Identical to the shape in m_partnerBody (if available).
			HK_PAD_ON_SPU( hknpShapeKey )		m_partnerShapeKey;	///< The (colliding) partner shape's full shape key.
			HK_PAD_ON_SPU( const hknpShape* )	m_partnerShape;		///< The (colliding) partner shape. Can be HK_NULL.
		};

	public:

		HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_PHYSICS);
		HK_DECLARE_REFLECTION();

		HK_FORCE_INLINE hknpShapeTagCodec(CodecType type) : m_type(type) {}

		/// This method can be used to decode the \a shapeTag into the 3 output parameters (\a collisionFilterInfo,
		/// \a materialId and \a userData).
		/// If available, you can base your decoding algorithm on additional collision partner information stored in
		/// the optional \a context parameter.
		/// While descending down a shape hierarchy the 3 output parameters will be preset (before calling this method)
		/// with values coming from the next higher (shape hierarchy) level.
		virtual void decode(
			hknpShapeTag shapeTag, const Context* context,
			hkUint32* collisionFilterInfo, hknpMaterialId* materialId, hkUint64* userData ) const = 0;

		/// This method decodes a path of shape tags (passed in \a shapeTagPath), followed by a final shape tag
		/// (\a finalTag).
		/// For more information on the decoding of one single tag, see the other decode() method above.
		HK_AUTO_INLINE void decode(
			const hknpShapeTagPathEntry* shapeTagPath, int numShapeTagPathEntries, hknpShapeTag finalTag, const Context* context,
			hkUint32* collisionFilterInfo, hknpMaterialId* materialId, hkUint64* userData ) const
		{
			decodeImpl( shapeTagPath, numShapeTagPathEntries, finalTag, context, collisionFilterInfo, materialId, userData );
		}

		/// The (inlined) implementation of decode(). See above for details.
		HK_FORCE_INLINE void decodeImpl(
			const hknpShapeTagPathEntry* shapeTagPath, int numShapeTagPathEntries, hknpShapeTag finalTag, const Context* context,
			hkUint32* collisionFilterInfo, hknpMaterialId* materialId, hkUint64* userData ) const;

	public:

		/// Codec type
		hkEnum<CodecType, hkUint8> m_type;
};


#include <Physics/Physics/Collide/Shape/TagCodec/hknpShapeTagCodec.inl>

#endif	// HKNP_SHAPE_TAG_CODEC_H

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
