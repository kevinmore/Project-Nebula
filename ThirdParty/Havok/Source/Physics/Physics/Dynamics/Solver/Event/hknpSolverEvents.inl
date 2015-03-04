//
/*
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


HK_FORCE_INLINE void hknpContactSolverEvent::initialize( const hknpModifier::SolverCallbackInput& input )
{
	HK_ON_CPU( m_contactJacobian = input.m_contactJacobian; )
	HK_ON_SPU( m_contactJacobian = input.m_contactJacobianInMainMemory; )
	m_manifoldIndex = (hkUint8)input.m_manifoldIndex;
}

#if !defined(HK_PLATFORM_SPU)

HK_FORCE_INLINE const hknpContactJacobianTypes::HeaderData& hknpContactSolverEvent::getJacobianHeader() const
{
	HK_ASSERT( 0x1df342b, m_contactJacobian );
	return m_contactJacobian->m_manifoldData[m_manifoldIndex];
}

HK_FORCE_INLINE int hknpContactSolverEvent::getNumContactPoints() const
{
	return m_contactJacobian->m_manifoldData[m_manifoldIndex].m_numContactPoints;
}

#endif


HK_FORCE_INLINE void hknpContactImpulseEvent::setContinuedEventsEnabled( bool areEnabled ) const
{
	if( m_contactJacobian )
	{
		hknpManifoldSolverInfo* msi = &(m_contactJacobian->m_manifoldData[m_manifoldIndex].m_collisionCacheInMainMemory->m_manifoldSolverInfo);
		HK_ASSERT( 0x5dc8fe58, msi );
		if( areEnabled )
		{
			msi->m_flags.orWith( hknpManifoldSolverInfo::RAISE_CONTACT_IMPULSE_CONTINUED_EVENTS );
		}
		else
		{
			msi->m_flags.clear( hknpManifoldSolverInfo::RAISE_CONTACT_IMPULSE_CONTINUED_EVENTS );
		}
	}
}

HK_FORCE_INLINE void hknpContactImpulseEvent::setFinishedEventsEnabled( bool areEnabled ) const
{
	if( m_contactJacobian )
	{
		hknpManifoldSolverInfo* msi = &(m_contactJacobian->m_manifoldData[m_manifoldIndex].m_collisionCacheInMainMemory->m_manifoldSolverInfo);
		HK_ASSERT( 0x2c768570, msi );
		if( areEnabled )
		{
			msi->m_flags.orWith( hknpManifoldSolverInfo::RAISE_CONTACT_IMPULSE_FINISHED_EVENTS );
		}
		else
		{
			msi->m_flags.clear( hknpManifoldSolverInfo::RAISE_CONTACT_IMPULSE_FINISHED_EVENTS );
		}
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
