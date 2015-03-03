/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_NULL_SHAPE_TAG_CODEC_H
#define HKNP_NULL_SHAPE_TAG_CODEC_H

#include <Physics/Physics/Collide/Shape/TagCodec/hknpShapeTagCodec.h>


/// The default NULL Shape Tag Codec.
/// This codec does not define any encoding scheme and thus will not be able to decode any data from the shape tag.
/// For debugging purposes it will assert when asked to decode any shape tag that has been explicitly set by the user.
class hknpNullShapeTagCodec : public hknpShapeTagCodec
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );

		/// Constructor.
		hknpNullShapeTagCodec();

		// hknpShapeTagCodec implementation.
		virtual void decode( hknpShapeTag shapeTag, const Context* context,
			hkUint32* collisionFilterInfo, hknpMaterialId* materialId, hkUint64* userData ) const HK_OVERRIDE;
};


#endif	// HKNP_NULL_SHAPE_TAG_CODEC_H

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
