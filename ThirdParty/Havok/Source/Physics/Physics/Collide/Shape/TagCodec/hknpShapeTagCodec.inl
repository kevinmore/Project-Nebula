/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


void hknpShapeTagCodec::decodeImpl(
	const hknpShapeTagPathEntry* shapeTagPath, int numShapeTagPathEntries,
	hknpShapeTag finalTag, const Context* context, hkUint32* collisionFilterInfo,
	hknpMaterialId* materialId, hkUint64* userData ) const
{
	Context tmpContext;
	Context* tmpContextPtr = HK_NULL;
	if ( context != HK_NULL )
	{
		tmpContext = *context;
		tmpContextPtr = &tmpContext;
	}

	for (int i = 0; i < numShapeTagPathEntries; i++)
	{
		const hknpShapeTagPathEntry* shapeTagPathEntry = &shapeTagPath[i];

		tmpContext.m_parentShape	= shapeTagPathEntry->m_parentShape;
		tmpContext.m_shapeKey		= shapeTagPathEntry->m_shapeKey;
		tmpContext.m_shape			= shapeTagPathEntry->m_shape;

		decode(shapeTagPathEntry->m_shapeTag, tmpContextPtr, collisionFilterInfo, materialId, userData);
	}

	decode(finalTag, context, collisionFilterInfo, materialId, userData);
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
