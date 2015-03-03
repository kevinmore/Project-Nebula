/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

HK_FORCE_INLINE const hkTransform& hknpVehicleInstance::getChassisTransform( void ) const
{
	return m_world->getBody(m_body).getTransform();
}

HK_FORCE_INLINE const hknpMotion& hknpVehicleInstance::getChassisMotion( void ) const
{
	const hknpBody& body = m_world->getBody(m_body);
	return m_world->getMotion(body.m_motionId);
}

HK_FORCE_INLINE hknpMotion& hknpVehicleInstance::accessChassisMotion( void )
{
	const hknpBody& body = m_world->getBody(m_body);
	return m_world->accessMotionUnchecked(body.m_motionId);	
}

HK_FORCE_INLINE hkUint8 hknpVehicleInstance::getNumWheels () const
{
	return m_data->m_numWheels;
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
