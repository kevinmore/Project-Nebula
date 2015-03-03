/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Collide/Shape/TagCodec/Null/hknpNullShapeTagCodec.h>


hknpNullShapeTagCodec::hknpNullShapeTagCodec() : hknpShapeTagCodec(NULL_CODEC)
{
}

void hknpNullShapeTagCodec::decode( hknpShapeTag shapeTag, const Context* context,
	hkUint32* collisionFilterInfo, hknpMaterialId* materialId, hkUint64* userData ) const
{
#if defined (HK_DEBUG)
	if (shapeTag != HKNP_INVALID_SHAPE_TAG)
	{
		HK_WARN_ONCE( 0xaf4eded2, "You cannot use the 'Null' codec to decode a shape tag." );
	}
#endif
}

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
