/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


HK_FORCE_INLINE bool hknpBodyQuality::operator==( const hknpBodyQuality& other ) const
{
	return 0 == hkString::memCmp( (const hkUint32*)this, (const hkUint32*)&other, sizeof(*this) );
}

HK_FORCE_INLINE const hknpBodyQuality* hknpBodyQuality::combineBodyQualities(
	const hknpBodyQuality* qA, const hknpBodyQuality* qB, Flags* flagsOut )
{
	if( qA->m_priority >= qB->m_priority )
	{
		*flagsOut = qB->m_supportedFlags.get() & qA->m_requestedFlags.get();
		return qA;
	}
	else
	{
		*flagsOut = qA->m_supportedFlags.get() & qB->m_requestedFlags.get();
		return qB;
	}
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
