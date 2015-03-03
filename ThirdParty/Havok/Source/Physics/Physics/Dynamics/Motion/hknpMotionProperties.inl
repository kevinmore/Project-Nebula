/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


HK_FORCE_INLINE hknpMotionProperties::hknpMotionProperties( hknpMotionPropertiesId::Preset preset )
:	m_isExclusive(false)
{
	setPreset(preset);
}

HK_FORCE_INLINE bool hknpMotionProperties::operator==( const hknpMotionProperties& other ) const
{
	return 0 == hkString::memCmpUint32( (const hkUint32*)this, (const hkUint32*)&other, sizeof(*this)/sizeof(hkUint32) );
}

HK_FORCE_INLINE void hknpMotionProperties::checkConsistency() const
{
	HK_ASSERT( 0x9e14f801, m_maxLinearSpeed >= 0.0f );
	HK_ASSERT( 0x9e14f802, m_maxAngularSpeed >= 0.0f );
	HK_ASSERT( 0x9e14f803, m_linearDamping >= 0.0f );
	HK_ASSERT( 0x9e14f804, m_angularDamping >= 0.0f );
	HK_ASSERT( 0x9e14f805, m_solverStabilizationSpeedThreshold >= 0.0f );
	HK_ASSERT( 0x9e14f806, m_solverStabilizationSpeedReduction >= 0.0f );
	HK_ASSERT( 0x9e14f807, m_maxDistSqrd >= 0.0f );
	HK_ASSERT( 0x9e14f808, m_maxRotSqrd >= 0.0f );
}


HK_FORCE_INLINE void hknpMotionProperties::FreeListArrayOperations::setEmpty( hknpMotionProperties& mp, hkUint32 next )
{
	// Make sure all members have serializable values
	::new (reinterpret_cast<hkPlacementNewArg*>(&mp)) hknpMotionProperties();

	// Use an invalid but serializable value to mark the object as empty
	mp.m_numDeactivationFrequencyPasses = (hkUint8)-1;

	// Store next empty element index in a data member where it will be serialized properly
	(hkUint32&)mp.m_isExclusive = next;
}

HK_FORCE_INLINE hkUint32 hknpMotionProperties::FreeListArrayOperations::getNext( const hknpMotionProperties& mp )
{
	return (const hkUint32&)mp.m_isExclusive;
}

HK_FORCE_INLINE hkBool32 hknpMotionProperties::FreeListArrayOperations::isEmpty( const hknpMotionProperties& mp )
{
	return mp.m_numDeactivationFrequencyPasses == (hkUint8)-1;
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
