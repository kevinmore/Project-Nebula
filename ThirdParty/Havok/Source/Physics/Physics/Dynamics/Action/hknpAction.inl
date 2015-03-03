/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


HK_FORCE_INLINE hknpAction::hknpAction( hkUlong userData )
:	m_userData( userData )
{
}


HK_FORCE_INLINE hknpAction::hknpAction( class hkFinishLoadedObjectFlag flag )
:	hkReferencedObject( flag )
{
}


HK_FORCE_INLINE void hknpAction::addLink(
	const hknpMotion* motionA, const hknpMotion* motionB, hknpCdPairWriter* HK_RESTRICT pairWriter )
{
	if ( !motionA->isStatic() && !motionB->isStatic() )
	{
		addLinkUnchecked( motionA, motionB, pairWriter );
	}
}


HK_FORCE_INLINE hknpMotion* hknpAction::getMotion( hknpWorld* world, const hknpBody& body )
{
	return &world->m_motionManager.accessMotionBuffer()[ body.m_motionId.value() ];
}


HK_FORCE_INLINE hknpUnaryAction::hknpUnaryAction( class hkFinishLoadedObjectFlag flag )
:	hknpAction( flag )
{
}


HK_FORCE_INLINE hknpAction::ApplyActionResult hknpUnaryAction::getAndCheckBodies(
	hknpWorld* world, const hknpBody*& bodyOut )
{
	if ( !world->isBodyValid( m_body ) ) { bodyOut = HK_NULL; return RESULT_REMOVE; }

	const hknpBody* bodyA = &world->getBody( m_body );

	if ( bodyA->isInactive() ) { bodyOut = HK_NULL; return RESULT_DEACTIVATE; }

	bodyOut = bodyA;

	return RESULT_OK;
}


HK_FORCE_INLINE hknpUnaryAction::hknpUnaryAction( hknpBodyId idA, hkUlong userData )
:	hknpAction( userData )
{
	m_body = idA;
}


HK_FORCE_INLINE hknpBinaryAction::hknpBinaryAction( hknpBodyId idA, hknpBodyId idB, hkUlong userData )
:	hknpAction( userData )
{
	m_bodyA = idA;
	m_bodyB = idB;
}


HK_FORCE_INLINE hknpBinaryAction::hknpBinaryAction( class hkFinishLoadedObjectFlag flag )
:	hknpAction( flag )
{
}


HK_FORCE_INLINE hknpAction::ApplyActionResult hknpBinaryAction::getAndCheckBodies(
	hknpWorld* world, const hknpBody*& bodyAOut, const hknpBody*& bodyBOut )
{
	if ( !world->isBodyValid( m_bodyA ) ) { bodyAOut = HK_NULL; bodyBOut = HK_NULL; return RESULT_REMOVE; }
	if ( !world->isBodyValid( m_bodyB ) ) { bodyAOut = HK_NULL; bodyBOut = HK_NULL; return RESULT_REMOVE; }

	const hknpBody* bodyA = &world->getBody( m_bodyA );
	if ( bodyA->isInactive() ) { bodyAOut = HK_NULL; bodyBOut = HK_NULL; return RESULT_DEACTIVATE; }

	const hknpBody* bodyB = &world->getBody( m_bodyB );
	if ( bodyB->isInactive() ) { bodyAOut = HK_NULL; bodyBOut = HK_NULL; return RESULT_DEACTIVATE; }

	bodyAOut = bodyA;
	bodyBOut = bodyB;

	return RESULT_OK;
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
