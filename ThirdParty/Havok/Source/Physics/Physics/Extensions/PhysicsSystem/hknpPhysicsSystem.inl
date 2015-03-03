/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

HK_FORCE_INLINE const hknpPhysicsSystemData* hknpPhysicsSystem::getData() const
{
	return m_data;
}

HK_FORCE_INLINE const hknpWorld* hknpPhysicsSystem::getWorld() const
{
	return m_world;
}

HK_FORCE_INLINE hknpWorld* hknpPhysicsSystem::accessWorld()
{
	return m_world;
}

HK_FORCE_INLINE const hkArray< hknpBodyId >& hknpPhysicsSystem::getBodyIds() const
{
	return m_bodyIds;
}

HK_FORCE_INLINE hkArray< hknpBodyId >& hknpPhysicsSystem::accessBodyIds()
{
	return m_bodyIds;
}

HK_FORCE_INLINE const hkArray< hknpConstraint* >& hknpPhysicsSystem::getConstraints() const
{
	return m_constraints;
}

HK_FORCE_INLINE hkArray< hknpConstraint* >& hknpPhysicsSystem::accessConstraints()
{
	return m_constraints;
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
